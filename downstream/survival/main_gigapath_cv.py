import os
import sys
import time
import pickle as pkl
import h5py
import gc
import math
import pandas as pd
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, SequentialSampler
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import numpy as np
from sksurv.metrics import concordance_index_censored

torch.manual_seed(0)
np.random.seed(0)

GIAPATH_PATH = os.path.join(os.path.dirname(__file__), "..", "..", "..", "prov-gigapath")
sys.path.insert(0, GIGAPATH_PATH)

try:
    from gigapath import slide_encoder
    GIGAPATH_AVAILABLE = True
except ImportError:
    GIGAPATH_AVAILABLE = False

GIGAPATH_DIM = 1536
DEFAULT_LR_SEARCH = [1e-4, 5e-4, 1e-3, 2e-3]
DEFAULT_LATENT_DIM_SEARCH = [512, 768, 1024]

class SurvivalHead(nn.Module):
    def __init__(
        self,
        input_dim=1536,
        latent_dim=768,
        feat_layer='11',
        n_classes=4,  
        model_arch="gigapath_slide_enc12l768d",
        pretrained="hf_hub:prov-gigapath/prov-gigapath",
        freeze=False,
        **kwargs,
    ):
        super(SurvivalHead, self).__init__()

        self.feat_layer = [eval(x) for x in feat_layer.split("-")]
        self.feat_dim = len(self.feat_layer) * latent_dim
        self.slide_encoder = slide_encoder.create_model(pretrained, model_arch, in_chans=input_dim, **kwargs)

        if freeze:
            print("Freezing Pretrained GigaPath model")
            for name, param in self.slide_encoder.named_parameters():
                param.requires_grad = False
            print("Done")

        self.classifier = nn.Sequential(
            nn.Linear(self.feat_dim, n_classes)
        )

    def forward(self, images: torch.Tensor, coords: torch.Tensor) -> torch.Tensor:
    
        if len(images.shape) == 2:
            images = images.unsqueeze(0)
        assert len(images.shape) == 3

        img_enc = self.slide_encoder.forward(images, coords, all_layer_embed=True)
        img_enc = [img_enc[i] for i in self.feat_layer]
        img_enc = torch.cat(img_enc, dim=-1)

        h = img_enc.reshape([-1, img_enc.size(-1)])
        logits = self.classifier(h)
        return logits

class NLLSurvLoss(nn.Module):
    def __init__(self, alpha=0.0, eps=1e-7, reduction='mean'):
        super().__init__()
        self.alpha = alpha
        self.eps = eps
        self.reduction = reduction

    def forward(self, logits, times, censorships):
        y = times.long()
        c = censorships.long()

        hazards = torch.sigmoid(logits)
        S = torch.cumprod(1 - hazards, dim=1)
        S_padded = torch.cat([torch.ones_like(c), S], 1)

        s_prev = torch.gather(S_padded, dim=1, index=y).clamp(min=self.eps)
        h_this = torch.gather(hazards, dim=1, index=y).clamp(min=self.eps)
        s_this = torch.gather(S_padded, dim=1, index=y + 1).clamp(min=self.eps)

        uncensored_loss = -(1 - c) * (torch.log(s_prev) + torch.log(h_this))
        censored_loss = -c * torch.log(s_this)
        neg_l = censored_loss + uncensored_loss

        if self.alpha is not None:
            loss = (1 - self.alpha) * neg_l + self.alpha * uncensored_loss

        if self.reduction == 'mean':
            loss = loss.mean()
            censored_loss = censored_loss.mean()
            uncensored_loss = uncensored_loss.mean()
        elif self.reduction == 'sum':
            loss = loss.sum()
            censored_loss = censored_loss.sum()
            uncensored_loss = uncensored_loss.sum()

        return {'loss': loss, 'uncensored_loss': uncensored_loss, 'censored_loss': censored_loss}


class CoxLoss(nn.Module):
    def __init__(self):
        super().__init__()
    
    def forward(self, logits, times, censorships):
        lrisks = logits
        survival_times = times
        event_indicators = (1 - censorships).float()
        num_uncensored = torch.sum(event_indicators, 0)
        if num_uncensored.item() == 0:
            return {'loss': torch.sum(lrisks) * 0}

        survival_times = survival_times.squeeze(1)
        event_indicators = event_indicators.squeeze(1)
        lrisks = lrisks.squeeze(1)

        sindex = torch.argsort(-survival_times)
        survival_times = survival_times[sindex]
        event_indicators = event_indicators[sindex]
        lrisks = lrisks[sindex]

        log_risk_stable = torch.logcumsumexp(lrisks, 0)
        likelihood = lrisks - log_risk_stable
        uncensored_likelihood = likelihood * event_indicators
        logL = -torch.sum(uncensored_likelihood)
        return {'loss': logL / num_uncensored}


def compute_discretization(df, survival_time_col, censorship_col, n_label_bins=4, label_bins=None, pid_col=None):
    df_unique = df[~df[pid_col].duplicated()]

    if label_bins is not None:
        assert len(label_bins) == n_label_bins + 1
        q_bins = label_bins
    else:
        uncensored_df = df_unique[df_unique[censorship_col] == 0]
        _, q_bins = pd.qcut(uncensored_df[survival_time_col], q=n_label_bins, retbins=True, labels=False)
        q_bins[-1] = 1e6
        q_bins[0] = -1e-6

    disc_labels, q_bins = pd.cut(
        df[survival_time_col], bins=q_bins,
        retbins=True, labels=False,
        include_lowest=True
    )

    disc_labels.name = 'disc_label'
    return disc_labels, q_bins


class GigaPathSurvivalDataset(Dataset):
    def __init__(
            self,
            embeds_root,
            df,
            fold_names,
            survival_time_col,
            censorship_col,
            pid_col,
            n_label_bins=4,
            label_bins=None,
            max_tiles=10000,
            shuffle_tiles=False):

        self.embeds_root = embeds_root
        self.df = df.copy()
        self.fold_names = fold_names
        self.max_tiles = max_tiles
        self.shuffle_tiles = shuffle_tiles

        self.survival_time_col = survival_time_col
        self.censorship_col = censorship_col
        self.n_label_bins = n_label_bins
        self.label_bins = label_bins

        if self.n_label_bins > 0:
            disc_labels, label_bins = compute_discretization(
                df=self.df,
                survival_time_col=self.survival_time_col,
                censorship_col=self.censorship_col,
                n_label_bins=self.n_label_bins,
                label_bins=self.label_bins,
                pid_col=pid_col)
            self.df = self.df.join(disc_labels)
            self.label_bins = label_bins
            self.target_col = disc_labels.name

        self.pid_col = pid_col
        self.df = self.df.set_index(pid_col, drop=False)

    def get_pids(self):
        return self.fold_names

    def __len__(self):
        return len(self.fold_names)

    def get_label_bins(self):
        return self.label_bins

    def read_h5_file(self, h5_path):
        """Read features and coordinates from H5 file"""
        with h5py.File(h5_path, 'r') as f:
            features = np.array(f['features'])
            coords = np.array(f['coords'])
        return features, coords

    def __getitem__(self, idx):
        slide_name = self.fold_names[idx]
        curr_row = self.df.loc[slide_name]
        censorship = curr_row[self.censorship_col]
        time = curr_row[self.survival_time_col]
        target = curr_row[self.target_col]

        # Try H5 format first
        h5_path = os.path.join(self.embeds_root, f'{slide_name}.h5')
        if os.path.exists(h5_path):
            features, coords = self.read_h5_file(h5_path)
        else:
            npy_path = os.path.join(self.embeds_root, f'{slide_name}.npy')
            features = np.load(npy_path, allow_pickle=True)
            n_tiles = features.shape[0]
            grid_size = int(np.ceil(np.sqrt(n_tiles)))
            coords = np.array([[i // grid_size * 256, i % grid_size * 256]
                              for i in range(n_tiles)])

        features = torch.from_numpy(features.astype(np.float32))
        coords = torch.from_numpy(coords.astype(np.float32))

        if self.shuffle_tiles:
            indices = torch.randperm(len(features))
            features = features[indices]
            coords = coords[indices]

        if features.size(0) > self.max_tiles:
            features = features[:self.max_tiles]
            coords = coords[:self.max_tiles]

        return {
            'imgs': features,
            'coords': coords,
            'survival_time': torch.tensor([time]),
            'censorship': torch.tensor([censorship]),
            'label': torch.tensor([target]),
            'slide_id': slide_name
        }

def pad_tensors(imgs, coords):
    """Pad tensors to the same length for batching"""
    max_len = max([t.size(0) for t in imgs])
    padded_imgs = []
    padded_coords = []
    masks = []

    for i in range(len(imgs)):
        tensor = imgs[i]
        coord = coords[i]
        N_i = tensor.size(0)

        padded_tensor = torch.zeros(max_len, tensor.size(1))
        padded_coord = torch.zeros(max_len, 2)
        mask = torch.zeros(max_len)

        padded_tensor[:N_i] = tensor
        padded_coord[:N_i] = coord
        mask[:N_i] = torch.ones(N_i)

        padded_imgs.append(padded_tensor)
        padded_coords.append(padded_coord)
        masks.append(mask)

    padded_imgs = torch.stack(padded_imgs)
    padded_coords = torch.stack(padded_coords)
    masks = torch.stack(masks).bool()

    return padded_imgs, padded_coords, masks

def survival_collate_fn(samples):
    """Collate function for survival dataset"""
    image_list = [s['imgs'] for s in samples]
    coord_list = [s['coords'] for s in samples]
    survival_time_list = [s['survival_time'] for s in samples]
    censorship_list = [s['censorship'] for s in samples]
    label_list = [s['label'] for s in samples]
    slide_id_list = [s['slide_id'] for s in samples]

    survival_times = torch.stack(survival_time_list)
    censorships = torch.stack(censorship_list)
    labels = torch.stack(label_list)
    pad_imgs, pad_coords, pad_mask = pad_tensors(image_list, coord_list)

    return {
        'imgs': pad_imgs,
        'coords': pad_coords,
        'pad_mask': pad_mask,
        'survival_time': survival_times,
        'censorship': censorships,
        'label': labels,
        'slide_id': slide_id_list
    }

class AverageMeter:
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def param_groups_lrd(model, weight_decay=0.05, layer_decay=0.75):
    """Layer-wise learning rate decay for GigaPath model"""
    param_groups = {}

    try:
        num_layers = model.slide_encoder.encoder.num_layers + 1
    except:
        num_layers = 12 + 1

    layer_scales = list(layer_decay ** (num_layers - i) for i in range(num_layers + 1))

    for n, p in model.named_parameters():
        if not p.requires_grad:
            continue

        if p.ndim == 1:
            g_decay = "no_decay"
            this_decay = 0.
        else:
            g_decay = "decay"
            this_decay = weight_decay

        layer_id = get_layer_id(n, num_layers)
        group_name = f"{n}_{layer_id}_{g_decay}"

        if group_name not in param_groups:
            this_scale = layer_scales[layer_id]
            param_groups[group_name] = {
                "lr_scale": this_scale,
                "weight_decay": this_decay,
                "params": [],
            }

        param_groups[group_name]["params"].append(p)

    return list(param_groups.values())


def get_layer_id(name, num_layers):
    """Get layer ID for learning rate decay"""
    if 'cls_token' in name or 'pos_embed' in name:
        return 0
    elif name.startswith('patch_embed') or name.startswith('slide_encoder.patch_embed'):
        return 0
    elif 'slide_encoder.encoder.layers' in name:
        parts = name.split('.')
        for i, part in enumerate(parts):
            if part == 'layers' and i + 1 < len(parts):
                try:
                    return int(parts[i + 1]) + 1
                except ValueError:
                    pass
        return num_layers
    else:
        return num_layers


def adjust_learning_rate(optimizer, epoch, args):
    """Cosine annealing with warmup"""
    if epoch < args.warmup_epochs:
        lr = args.lr * epoch / args.warmup_epochs if args.warmup_epochs > 0 else args.lr
    else:
        lr = args.min_lr + (args.lr - args.min_lr) * 0.5 * \
            (1. + math.cos(math.pi * (epoch - args.warmup_epochs) / (args.max_epochs - args.warmup_epochs)))

    for param_group in optimizer.param_groups:
        if "lr_scale" in param_group:
            param_group["lr"] = lr * param_group["lr_scale"]
        else:
            param_group["lr"] = lr
    return lr


def process_surv(logits, label, censorship, loss_fn):
    """Process survival predictions and compute loss"""
    results_dict = {'logits': logits}
    log_dict = {}

    if isinstance(loss_fn, NLLSurvLoss):
        surv_loss_dict = loss_fn(logits, label, censorship)
        hazards = torch.sigmoid(logits)
        survival = torch.cumprod(1 - hazards, dim=1)
        risk = -torch.sum(survival, dim=1).unsqueeze(dim=1)
        results_dict.update({
            'hazards': hazards,
            'survival': survival,
            'risk': risk
        })
    elif isinstance(loss_fn, CoxLoss):
        surv_loss_dict = loss_fn(logits, label, censorship)
        risk = torch.exp(logits)
        results_dict['risk'] = risk

    loss = surv_loss_dict['loss']
    log_dict['surv_loss'] = surv_loss_dict['loss'].item()
    log_dict.update({
        k: v.item() for k, v in surv_loss_dict.items() if isinstance(v, torch.Tensor)
    })
    results_dict.update({'loss': loss})

    return results_dict, log_dict


def save_checkpoint(model, optimizer, epoch, val_loss, train_loss, c_index_train, c_index_val, args):
    """Save training checkpoint"""
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'val_loss': val_loss,
        'val_c_index': c_index_val,
        'train_loss': train_loss,
        'train_c_index': c_index_train,
    }

    checkpoint_path = os.path.join(args.results_dir,
                                   f'checkpoint_epoch_{epoch}_val_{val_loss:.4f}_c_ind_{c_index_val:.4f}.pth')
    torch.save(checkpoint, checkpoint_path)

    log_path = os.path.join(args.results_dir, 'training_val_log.csv')
    if not os.path.exists(log_path):
        with open(log_path, 'w') as f:
            f.write('epoch,train_loss,c_index_train,val_loss,c_index_val\n')

    with open(log_path, 'a') as f:
        f.write(f'{epoch},{train_loss:.4f},{c_index_train:.4f},{val_loss:.4f},{c_index_val:.4f}\n')


def train_step(model, loader, optimizer, loss_fn, epoch, args):
    model.train()
    meters = {}
    all_risk_scores, all_censorships, all_event_times = [], [], []

    progress_bar = tqdm(enumerate(loader), total=len(loader), ncols=110)
    progress_bar.set_description(f'Epoch {epoch}/{args.max_epochs}')

    for batch_idx, batch in progress_bar:
        if batch_idx % args.gc == 0:
            adjust_learning_rate(optimizer, batch_idx / len(loader) + epoch, args)

        images = batch['imgs'].to(args.device)
        coords = batch['coords'].to(args.device)
        labels = batch['label'].to(args.device)
        event_times = batch['survival_time'].to(args.device)
        censorships = batch['censorship'].to(args.device)

        with torch.cuda.amp.autocast(enabled=args.fp16):
            logits = model(images, coords)
            out, log_dict = process_surv(logits, labels, censorships, loss_fn)

            if out['loss'] is None:
                continue

            loss = out['loss'] / args.gc

        loss.backward()

        if (batch_idx + 1) % args.gc == 0:
            optimizer.step()
            optimizer.zero_grad()

        all_risk_scores.append(out['risk'].detach().cpu().numpy())
        all_censorships.append(censorships.cpu().numpy())
        all_event_times.append(event_times.cpu().numpy())

        for key, val in log_dict.items():
            if key not in meters:
                meters[key] = AverageMeter()
            meters[key].update(val=val, n=len(images))

        progress_bar.set_postfix({"loss": meters['surv_loss'].avg if 'surv_loss' in meters else 0})

    all_risk_scores = np.concatenate(all_risk_scores).squeeze()
    all_censorships = np.concatenate(all_censorships).squeeze()
    all_event_times = np.concatenate(all_event_times).squeeze()

    if len(all_risk_scores.shape) > 1:
        all_risk_scores = all_risk_scores[:, 0]

    c_index = concordance_index_censored(
        (1 - all_censorships).astype(bool), all_event_times, all_risk_scores, tied_tol=1e-08)[0]

    results = {k: meter.avg for k, meter in meters.items()}
    results['c_index'] = c_index
    results['loss'] = results.get('surv_loss', 0)

    print(f'Train - Loss: {results["loss"]:.4f}, C-index: {c_index:.4f}')

    del all_risk_scores, all_censorships, all_event_times
    gc.collect()
    torch.cuda.empty_cache()

    return results


def validate_step(model, loader, loss_fn, args):
    model.eval()
    meters = {}
    all_risk_scores, all_censorships, all_event_times = [], [], []

    with torch.no_grad():
        for batch in tqdm(loader, desc='Validating', ncols=110):
            images = batch['imgs'].to(args.device)
            coords = batch['coords'].to(args.device)
            labels = batch['label'].to(args.device)
            event_times = batch['survival_time'].to(args.device)
            censorships = batch['censorship'].to(args.device)

            with torch.cuda.amp.autocast(enabled=args.fp16):
                logits = model(images, coords)
                out, log_dict = process_surv(logits, labels, censorships, loss_fn)

            all_risk_scores.append(out['risk'].detach().cpu().numpy())
            all_censorships.append(censorships.cpu().numpy())
            all_event_times.append(event_times.cpu().numpy())

            for key, val in log_dict.items():
                if key not in meters:
                    meters[key] = AverageMeter()
                meters[key].update(val=val, n=len(images))

    all_risk_scores = np.concatenate(all_risk_scores).squeeze()
    all_censorships = np.concatenate(all_censorships).squeeze()
    all_event_times = np.concatenate(all_event_times).squeeze()

    if len(all_risk_scores.shape) > 1:
        all_risk_scores = all_risk_scores[:, 0]

    c_index = concordance_index_censored(
        (1 - all_censorships).astype(bool), all_event_times, all_risk_scores, tied_tol=1e-08)[0]

    results = {k: meter.avg for k, meter in meters.items()}
    results['c_index'] = c_index
    results['loss'] = results.get('surv_loss', 0)

    print(f'Val - Loss: {results["loss"]:.4f}, C-index: {c_index:.4f}')

    del all_risk_scores, all_censorships, all_event_times
    gc.collect()
    torch.cuda.empty_cache()

    return results

def run_hyperparam_search_gigapath_survival(train_loader, val_loader, args, fold, lr_search, latent_dim_search):

    best_val_cindex = 0
    best_lr = args.lr
    best_latent_dim = args.latent_dim
    hyperparam_results = []

    if args.loss_fn == 'nll':
        loss_fn = NLLSurvLoss(alpha=args.nll_alpha)
    elif args.loss_fn == 'cox':
        loss_fn = CoxLoss()

    for lr in lr_search:
        for latent_dim in latent_dim_search:
            print(f'  Trying lr={lr}, latent_dim={latent_dim}')

            model = SurvivalHead(
                input_dim=GIGAPATH_DIM,
                latent_dim=latent_dim,
                feat_layer=args.feat_layer,
                n_classes=args.n_label_bins,
                model_arch=args.model_arch,
                pretrained=args.pretrained,
                freeze=args.freeze
            ).to(args.device)

            param_groups = param_groups_lrd(model, args.weight_decay, args.layer_decay)
            optimizer = torch.optim.AdamW(param_groups, lr=lr)

            search_epochs = min(args.max_epochs, 3)  # Fewer epochs for GigaPath due to compute cost
            best_epoch_cindex = 0

            for epoch in range(search_epochs):
                model.train()
                for batch_idx, batch in enumerate(train_loader):
                    images = batch['imgs'].to(args.device)
                    coords = batch['coords'].to(args.device)
                    labels = batch['label'].to(args.device)
                    censorships = batch['censorship'].to(args.device)

                    with torch.cuda.amp.autocast(enabled=args.fp16):
                        logits = model(images, coords)
                        out, _ = process_surv(logits, labels, censorships, loss_fn)
                        if out['loss'] is None:
                            continue
                        loss = out['loss'] / args.gc

                    loss.backward()

                    if (batch_idx + 1) % args.gc == 0:
                        optimizer.step()
                        optimizer.zero_grad()

                val_results = validate_step(model, val_loader, loss_fn, args)
                if val_results['c_index'] > best_epoch_cindex:
                    best_epoch_cindex = val_results['c_index']

            print(f'    Best val C-index: {best_epoch_cindex:.4f}')
            hyperparam_results.append({
                'lr': lr,
                'latent_dim': latent_dim,
                'val_cindex': best_epoch_cindex
            })

            if best_epoch_cindex > best_val_cindex:
                best_val_cindex = best_epoch_cindex
                best_lr = lr
                best_latent_dim = latent_dim

            del model, optimizer
            torch.cuda.empty_cache()

    return best_lr, best_latent_dim, best_val_cindex, hyperparam_results


def train_cycle(train_loader, val_loader, args, fold=None, lr=None, latent_dim=None):
    fold_metrics = {}

    lr_to_use = lr if lr is not None else args.lr
    latent_dim_to_use = latent_dim if latent_dim is not None else args.latent_dim

    model = SurvivalHead(
        input_dim=GIGAPATH_DIM,
        latent_dim=latent_dim_to_use,
        feat_layer=args.feat_layer,
        n_classes=args.n_label_bins,
        model_arch=args.model_arch,
        pretrained=args.pretrained,
        freeze=args.freeze
    ).to(args.device)

    param_groups = param_groups_lrd(model, args.weight_decay, args.layer_decay)
    optimizer = torch.optim.AdamW(param_groups, lr=lr_to_use)

    if args.loss_fn == 'nll':
        loss_fn = NLLSurvLoss(alpha=args.nll_alpha)
    elif args.loss_fn == 'cox':
        loss_fn = CoxLoss()
    else:
        raise ValueError(f"Unknown loss function: {args.loss_fn}")

    writer = SummaryWriter(args.writer_dir, flush_secs=15)

    best_c_index = 0
    best_epoch = 0

    for epoch in range(args.max_epochs):
        fold_metrics[epoch] = {}

        print(f'\n--- Fold {fold}, Epoch {epoch}/{args.max_epochs} ---')
        train_results = train_step(model, train_loader, optimizer, loss_fn, epoch, args)
        fold_metrics[epoch]['train'] = train_results

        for k, v in train_results.items():
            writer.add_scalar(f'train/{k}', v, epoch)

        val_results = validate_step(model, val_loader, loss_fn, args)
        fold_metrics[epoch]['val'] = val_results

        for k, v in val_results.items():
            writer.add_scalar(f'val/{k}', v, epoch)

        save_checkpoint(
            model, optimizer, epoch,
            val_results['loss'],
            train_results['loss'],
            train_results['c_index'],
            val_results['c_index'],
            args
        )

        if val_results['c_index'] > best_c_index:
            best_c_index = val_results['c_index']
            best_epoch = epoch
            torch.save(model.state_dict(), os.path.join(args.results_dir, 'best_model.pt'))

    fold_metrics['best_epoch'] = best_epoch
    fold_metrics['best_c_index'] = best_c_index

    writer.close()
    return fold_metrics

def main(args):
    if not GIGAPATH_AVAILABLE:
        raise ImportError("GigaPath modules not available. Please ensure prov-gigapath is in the path.")

    with open(args.splits, 'rb') as f:
        cv_folds = pkl.load(f)

    cross_fold_metrics = {}
    hyperparam_summary = {}
    all_c_indices = []

    if args.hyperparam_search:
        lr_search = [float(x) for x in args.lr_search.split(',')] if args.lr_search else DEFAULT_LR_SEARCH_GIGAPATH
        latent_dim_search = [int(x) for x in args.latent_dim_search.split(',')] if args.latent_dim_search else DEFAULT_LATENT_DIM_SEARCH
        print(f'Hyperparameter search enabled:')
        print(f'  Learning rates: {lr_search}')
        print(f'  Latent dimensions: {latent_dim_search}')
    else:
        lr_search = [args.lr]
        latent_dim_search = [args.latent_dim]

    for fold in range(args.fold_count):
        print(f'Starting Fold {fold}')

        save_folder = os.path.join(args.save_root, args.exp_name, f'fold_{fold}')
        os.makedirs(save_folder, exist_ok=True)

        args.results_dir = os.path.join(save_folder, 'results')
        args.writer_dir = os.path.join(save_folder, 'writer')
        os.makedirs(args.results_dir, exist_ok=True)
        os.makedirs(args.writer_dir, exist_ok=True)

        train_names = cv_folds[fold]['train']['train_ids']
        test_names = cv_folds[fold]['test']['test_ids']

        df = pd.read_csv(args.labels, index_col=0)

        train_dataset = GigaPathSurvivalDataset(
            args.bag_root,
            df,
            train_names,
            survival_time_col=args.survival_time_col,
            censorship_col=args.censorship_col,
            pid_col=args.pid_col,
            n_label_bins=args.n_label_bins,
            max_tiles=args.max_tiles,
            shuffle_tiles=args.shuffle_tiles
        )
        test_dataset = GigaPathSurvivalDataset(
            args.bag_root,
            df,
            test_names,
            survival_time_col=args.survival_time_col,
            censorship_col=args.censorship_col,
            pid_col=args.pid_col,
            n_label_bins=args.n_label_bins,
            label_bins=train_dataset.get_label_bins(),
            max_tiles=args.max_tiles,
            shuffle_tiles=False
        )

        train_loader = DataLoader(
            train_dataset,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=args.num_workers,
            collate_fn=survival_collate_fn,
            pin_memory=True
        )
        test_loader = DataLoader(
            test_dataset,
            batch_size=1,
            sampler=SequentialSampler(test_dataset),
            num_workers=args.num_workers,
            collate_fn=survival_collate_fn,
            pin_memory=True
        )

        if args.hyperparam_search:
            print(f'Running hyperparameter search for fold {fold}...')
            best_lr, best_latent_dim, best_search_cindex, hp_results = run_hyperparam_search_gigapath_survival(
                train_loader, test_loader, args, fold, lr_search, latent_dim_search
            )
            print(f'Best hyperparameters: lr={best_lr}, latent_dim={best_latent_dim}, val_cindex={best_search_cindex:.4f}')
            hyperparam_summary[fold] = {
                'best_lr': best_lr,
                'best_latent_dim': best_latent_dim,
                'search_results': hp_results
            }
        else:
            best_lr = args.lr
            best_latent_dim = args.latent_dim

        print(f'Training with lr={best_lr}, latent_dim={best_latent_dim}')
        fold_metrics = train_cycle(train_loader, test_loader, args, fold=fold, lr=best_lr, latent_dim=best_latent_dim)
        fold_metrics['best_lr'] = best_lr
        fold_metrics['best_latent_dim'] = best_latent_dim

        with open(os.path.join(args.results_dir, f'fold_{fold}_{args.task}_gigapath.pkl'), 'wb') as f:
            pkl.dump(fold_metrics, f)

        cross_fold_metrics[fold] = fold_metrics
        all_c_indices.append(fold_metrics['best_c_index'])

    print('Cross-Validation Results Summary (GigaPath Native Slide Encoder - Survival)')
    print(f'Task: {args.task}')
    print(f'Model: {args.model_arch}')
    print(f'Loss: {args.loss_fn}')
    print(f'Feature layers: {args.feat_layer}')
    print(f'Pretrained: {args.pretrained}')
    print(f'Fold C-indices: {all_c_indices}')
    print(f'Mean C-index: {np.mean(all_c_indices):.4f} +/- {np.std(all_c_indices):.4f}')
    if hyperparam_summary:
        print('Hyperparameter search summary:')
        for fold_idx, hp_info in hyperparam_summary.items():
            print(f'  Fold {fold_idx}: lr={hp_info["best_lr"]}, latent_dim={hp_info["best_latent_dim"]}')
    print(f'{"="*60}\n')

    cross_fold_metrics['summary'] = {
        'mean_c_index': np.mean(all_c_indices),
        'std_c_index': np.std(all_c_indices),
        'all_c_indices': all_c_indices
    }
    cross_fold_metrics['hyperparam_summary'] = hyperparam_summary

    with open(os.path.join(args.save_root, f'cross_val_results_{args.task}_gigapath.pkl'), 'wb') as f:
        pkl.dump(cross_fold_metrics, f)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='GigaPath Native Slide Encoder Cross-Validation Training for Survival')

    parser.add_argument('--save_root', type=str, required=True)
    parser.add_argument('--exp_name', type=str, default='gigapath_survival')
    parser.add_argument('--splits', type=str, required=True)
    parser.add_argument('--labels', type=str, required=True)
    parser.add_argument('--bag_root', type=str, required=True)

    parser.add_argument('--task', type=str, default='bcr',
                        choices=['bcr', 'plco_lung', 'plco_breast', 'tcga_prad', 'tcga_ucec', 'tcga_brca'])
    parser.add_argument('--fold_count', type=int, default=5)

    parser.add_argument('--model_arch', type=str, default='gigapath_slide_enc12l768d',
                        choices=['gigapath_slide_enc12l768d', 'gigapath_slide_enc24l1024d', 'gigapath_slide_enc12l1536d'])
    parser.add_argument('--pretrained', type=str, default='hf_hub:prov-gigapath/prov-gigapath')
    parser.add_argument('--latent_dim', type=int, default=768)
    parser.add_argument('--feat_layer', type=str, default='11')
    parser.add_argument('--freeze', action='store_true')

    parser.add_argument('--max_tiles', type=int, default=10000)
    parser.add_argument('--shuffle_tiles', action='store_true')

    parser.add_argument('--loss_fn', type=str, default='nll',
                        choices=['nll', 'cox'])
    parser.add_argument('--nll_alpha', type=float, default=0)
    parser.add_argument('--n_label_bins', type=int, default=4)
    parser.add_argument('--survival_time_col', type=str, default='days_to_event')
    parser.add_argument('--censorship_col', type=str, default='bcr')
    parser.add_argument('--pid_col', type=str, default='slide_name')

    parser.add_argument('--lr', type=float, default=2e-3)
    parser.add_argument('--min_lr', type=float, default=1e-6)
    parser.add_argument('--weight_decay', type=float, default=0.05)
    parser.add_argument('--layer_decay', type=float, default=0.95)
    parser.add_argument('--max_epochs', type=int, default=5)
    parser.add_argument('--warmup_epochs', type=int, default=1)
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--gc', type=int, default=32)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--fp16', action='store_true')

    parser.add_argument('--gpu', type=str, default='0')

    parser.add_argument('--hyperparam_search', action='store_true')
    parser.add_argument('--lr_search', type=str, default=None)
    parser.add_argument('--latent_dim_search', type=str, default=None)

    args = parser.parse_args()

    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    args.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    main(args)
