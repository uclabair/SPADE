import os
import sys
import time
import pickle
import h5py
import pandas as pd
import argparse
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler, SequentialSampler
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

import numpy as np
from sklearn.metrics import (
    roc_auc_score,
    precision_score,
    recall_score,
    f1_score,
    auc,
    confusion_matrix,
    precision_recall_curve
)

import pickle as pkl

torch.manual_seed(0)
np.random.seed(0)

GIGAPATH_PATH = os.path.join(os.path.dirname(__file__), '..', '..', '..', 'prov-gigapath')
sys.path.insert(0, GIGAPATH_PATH)

try:
    from gigapath.classification_head import ClassificationHead
    from gigapath import slide_encoder
    GIGAPATH_AVAILABLE = True
except ImportError:
    GIGAPATH_AVAILABLE = False
    print("Warning: GigaPath modules not found. Make sure prov-gigapath is in the path.")

GIGAPATH_DIM = 1536
DEFAULT_LR_SEARCH_GIGAPATH = [1e-4, 5e-4, 1e-3, 2e-3]
DEFAULT_LATENT_DIM_SEARCH = [512, 768, 1024]

class_count_mapping = {
    'cptac_lung': 2,  
    'camelyon16': 3,
    'panda': 6,
    'plco_lung': 2,
    'ovarian': 5,
    'tcga_brca': 2,
    'tcga_prad': 4,
    'plco_breast': 3,
    'brca_gene': 2,
    'crc_gene': 2
}

BINARY_TASKS = ['cptac_lung', 'plco_lung', 'tcga_brca', 'brca_gene', 'crc_gene']

class GigaPathSlideDatasetCV(Dataset):

    def __init__(self, names, labels, feat_path, max_tiles=10000, shuffle_tiles=False):
        self.names = names
        self.labels = labels
        self.feat_path = feat_path
        self.max_tiles = max_tiles
        self.shuffle_tiles = shuffle_tiles

    def __len__(self):
        return len(self.names)

    def read_h5_file(self, h5_path):
        with h5py.File(h5_path, 'r') as f:
            features = np.array(f['features'])
            coords = np.array(f['coords'])
        return features, coords

    def __getitem__(self, index):
        slide_id = self.names[index]
        label = self.labels[index]

        h5_path = os.path.join(self.feat_path, f'{slide_id}.h5')
        if os.path.exists(h5_path):
            features, coords = self.read_h5_file(h5_path)
        else:
            
            npy_path = os.path.join(self.feat_path, f'{slide_id}.npy')
            features = np.load(npy_path, allow_pickle=True)
            # Generate grid coordinates if not available
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
            'labels': torch.tensor(label),
            'slide_id': slide_id
        }


def pad_tensors(imgs, coords):
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


def slide_collate_fn(samples):
    image_list = [s['imgs'] for s in samples]
    coord_list = [s['coords'] for s in samples]
    label_list = [s['labels'] for s in samples]
    slide_id_list = [s['slide_id'] for s in samples]

    labels = torch.stack(label_list)
    pad_imgs, pad_coords, pad_mask = pad_tensors(image_list, coord_list)

    return {
        'imgs': pad_imgs,
        'coords': pad_coords,
        'pad_mask': pad_mask,
        'labels': labels,
        'slide_id': slide_id_list
    }

def get_sampler(labels_train):
    """Create weighted sampler for class imbalance"""
    labels_array = np.array(labels_train)
    class_idx, class_sample_count = np.unique(labels_array, return_counts=True)
    class_weights = 1 / torch.Tensor(class_sample_count)
    sample_weights = class_weights[labels_array]
    sampler = WeightedRandomSampler(
        sample_weights, num_samples=len(labels_train), replacement=True
    )
    return sampler


def calculate_metrics(preds, y_true, task_type='binary'):
    preds = np.array(preds)
    y_true = np.array(y_true)

    if task_type == 'binary':
        if len(preds.shape) > 1 and preds.shape[1] == 2:
            preds = preds[:, 1]  
        best_threshold = 0.5
        results_arr = (preds > best_threshold).astype(int)
        auc_ = roc_auc_score(y_true, preds)
        precision = precision_score(y_true, results_arr, zero_division=0)
        recall = recall_score(y_true, results_arr, zero_division=0)
        f1 = f1_score(y_true, results_arr, zero_division=0)

        cm = confusion_matrix(y_true, results_arr)
        if cm.shape == (2, 2):
            tn, fp, fn, tp = cm.ravel()
            sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
            specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
        else:
            sensitivity, specificity = 0, 0

        precision_curve, recall_curve, _ = precision_recall_curve(y_true, preds)
        auprc = auc(recall_curve, precision_curve)

        return {
            'auc': auc_,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'auprc': auprc,
            'sensitivity': sensitivity,
            'specificity': specificity
        }
    else:  
        results = np.argmax(preds, axis=1)
        auc_ = roc_auc_score(y_true, preds, multi_class='ovr', average='macro')
        precision = precision_score(y_true, results, average='macro', zero_division=0)
        recall = recall_score(y_true, results, average='macro', zero_division=0)
        f1 = f1_score(y_true, results, average='macro', zero_division=0)

        return {
            'auc': auc_,
            'precision_macro': precision,
            'recall_macro': recall,
            'f1_macro': f1
        }

def param_groups_lrd(model, weight_decay=0.05, layer_decay=0.75):
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
    if epoch < args.warmup_epochs:
        lr = args.lr * epoch / args.warmup_epochs
    else:
        lr = args.min_lr + (args.lr - args.min_lr) * 0.5 * \
            (1. + math.cos(math.pi * (epoch - args.warmup_epochs) / (args.epochs - args.warmup_epochs)))

    for param_group in optimizer.param_groups:
        if "lr_scale" in param_group:
            param_group["lr"] = lr * param_group["lr_scale"]
        else:
            param_group["lr"] = lr
    return lr


def train_one_epoch(train_loader, model, optimizer, criterion, epoch, args):
    model.train()
    total_loss = 0
    all_preds = []
    all_labels = []

    progress_bar = tqdm(enumerate(train_loader), total=len(train_loader), ncols=110)
    progress_bar.set_description(f'Epoch {epoch}/{args.epochs}')

    for batch_idx, batch in progress_bar:
        if batch_idx % args.gc == 0 and args.lr_scheduler == 'cosine':
            adjust_learning_rate(optimizer, batch_idx / len(train_loader) + epoch, args)

        images = batch['imgs'].to(args.device)
        coords = batch['coords'].to(args.device)
        labels = batch['labels'].to(args.device)

        with torch.cuda.amp.autocast(enabled=args.fp16):
            logits = model(images, coords)

            if args.task_type == 'binary':
                labels = labels.long()
            else:
                labels = labels.long()

            loss = criterion(logits, labels)
            loss = loss / args.gc

        loss.backward()

        if (batch_idx + 1) % args.gc == 0:
            optimizer.step()
            optimizer.zero_grad()

        total_loss += loss.item() * args.gc

        with torch.no_grad():
            if args.task_type == 'binary':
                probs = torch.softmax(logits, dim=1).cpu().numpy()
            else:
                probs = torch.softmax(logits, dim=1).cpu().numpy()
            all_preds.extend(probs)
            all_labels.extend(labels.cpu().numpy())

        progress_bar.set_postfix({"loss": total_loss / (batch_idx + 1)})

    avg_loss = total_loss / len(train_loader)
    metrics = calculate_metrics(all_preds, all_labels, args.task_type)
    metrics['loss'] = avg_loss

    return metrics


def evaluate(loader, model, criterion, args):
    model.eval()
    total_loss = 0
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for batch in tqdm(loader, desc='Evaluating', ncols=110):
            images = batch['imgs'].to(args.device)
            coords = batch['coords'].to(args.device)
            labels = batch['labels'].to(args.device)

            with torch.cuda.amp.autocast(enabled=args.fp16):
                logits = model(images, coords)

                if args.task_type == 'binary':
                    labels = labels.long()
                else:
                    labels = labels.long()

                loss = criterion(logits, labels)

            total_loss += loss.item()

            if args.task_type == 'binary':
                probs = torch.softmax(logits, dim=1).cpu().numpy()
            else:
                probs = torch.softmax(logits, dim=1).cpu().numpy()

            all_preds.extend(probs)
            all_labels.extend(labels.cpu().numpy())

    avg_loss = total_loss / len(loader)
    metrics = calculate_metrics(all_preds, all_labels, args.task_type)
    metrics['loss'] = avg_loss

    return metrics


def run_hyperparam_search_gigapath(train_loader, val_loader, args, fold, lr_search, latent_dim_search):

    best_val_auc = 0
    best_lr = args.lr
    best_latent_dim = args.latent_dim
    hyperparam_results = []

    n_classes = class_count_mapping[args.task]

    for lr in lr_search:
        for latent_dim in latent_dim_search:
            print(f'  Trying lr={lr}, latent_dim={latent_dim}')

            model = ClassificationHead(
                input_dim=GIGAPATH_DIM,
                latent_dim=latent_dim,
                feat_layer=args.feat_layer,
                n_classes=n_classes,
                model_arch=args.model_arch,
                pretrained=args.pretrained,
                freeze=args.freeze
            ).to(args.device)

            param_groups = param_groups_lrd(model, args.weight_decay, args.layer_decay)
            optimizer = torch.optim.AdamW(param_groups, lr=lr)
            criterion = nn.CrossEntropyLoss()

            search_epochs = min(args.epochs, 3)  # Fewer epochs for GigaPath due to compute cost
            best_epoch_auc = 0

            for epoch in range(search_epochs):
                model.train()
                for batch_idx, batch in enumerate(train_loader):
                    images = batch['imgs'].to(args.device)
                    coords = batch['coords'].to(args.device)
                    labels = batch['labels'].to(args.device).long()

                    with torch.cuda.amp.autocast(enabled=args.fp16):
                        logits = model(images, coords)
                        loss = criterion(logits, labels) / args.gc

                    loss.backward()

                    if (batch_idx + 1) % args.gc == 0:
                        optimizer.step()
                        optimizer.zero_grad()

                val_metrics = evaluate(val_loader, model, criterion, args)
                if val_metrics['auc'] > best_epoch_auc:
                    best_epoch_auc = val_metrics['auc']

            print(f'    Best val AUC: {best_epoch_auc:.4f}')
            hyperparam_results.append({
                'lr': lr,
                'latent_dim': latent_dim,
                'val_auc': best_epoch_auc
            })

            if best_epoch_auc > best_val_auc:
                best_val_auc = best_epoch_auc
                best_lr = lr
                best_latent_dim = latent_dim

            del model, optimizer
            torch.cuda.empty_cache()

    return best_lr, best_latent_dim, best_val_auc, hyperparam_results


def train_fold(train_loader, val_loader, args, fold, lr=None, latent_dim=None):
    fold_metrics = {}

    n_classes = class_count_mapping[args.task]

    lr_to_use = lr if lr is not None else args.lr
    latent_dim_to_use = latent_dim if latent_dim is not None else args.latent_dim

    model = ClassificationHead(
        input_dim=GIGAPATH_DIM,
        latent_dim=latent_dim_to_use,
        feat_layer=args.feat_layer,
        n_classes=n_classes,
        model_arch=args.model_arch,
        pretrained=args.pretrained,
        freeze=args.freeze
    ).to(args.device)

    param_groups = param_groups_lrd(model, args.weight_decay, args.layer_decay)
    optimizer = torch.optim.AdamW(param_groups, lr=lr_to_use)

    criterion = nn.CrossEntropyLoss()

    best_auc = 0
    best_epoch = 0

    for epoch in range(args.epochs):
        fold_metrics[epoch] = {}

        print(f'\n--- Fold {fold}, Epoch {epoch}/{args.epochs} ---')
        train_metrics = train_one_epoch(train_loader, model, optimizer, criterion, epoch, args)
        fold_metrics[epoch]['train'] = train_metrics
        print(f'Train Loss: {train_metrics["loss"]:.4f}, Train AUC: {train_metrics["auc"]:.4f}')

        val_metrics = evaluate(val_loader, model, criterion, args)
        fold_metrics[epoch]['val'] = val_metrics
        print(f'Val Loss: {val_metrics["loss"]:.4f}, Val AUC: {val_metrics["auc"]:.4f}')

        if val_metrics['auc'] > best_auc:
            best_auc = val_metrics['auc']
            best_epoch = epoch
            save_path = os.path.join(args.save_folder, f'best_model_fold_{fold}.pt')
            torch.save(model.state_dict(), save_path)

        ckpt_path = os.path.join(args.save_folder, f'checkpoint_fold_{fold}_epoch_{epoch}_auc_{val_metrics["auc"]:.4f}.pt')
        torch.save({
            'model': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'epoch': epoch,
            'val_auc': val_metrics['auc']
        }, ckpt_path)

    fold_metrics['best_epoch'] = best_epoch
    fold_metrics['best_val_auc'] = best_auc

    return fold_metrics


def main(args):
    if not GIGAPATH_AVAILABLE:
        raise ImportError("GigaPath modules not available. Please ensure prov-gigapath is in the path.")

    with open(args.cv_split_file, 'rb') as f:
        cv_folds = pkl.load(f)

    cross_fold_metrics = {}
    hyperparam_summary = {}
    all_val_aucs = []

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

        args.save_folder = os.path.join(args.save_root, f'fold_{fold}')
        os.makedirs(args.save_folder, exist_ok=True)

        train_names = cv_folds[fold]['train']['train_ids']
        train_labels = cv_folds[fold]['train']['train_labels']
        test_names = cv_folds[fold]['test']['test_ids']
        test_labels = cv_folds[fold]['test']['test_labels']

        train_dataset = GigaPathSlideDatasetCV(
            train_names, train_labels, args.feat_path,
            max_tiles=args.max_tiles, shuffle_tiles=args.shuffle_tiles
        )
        test_dataset = GigaPathSlideDatasetCV(
            test_names, test_labels, args.feat_path,
            max_tiles=args.max_tiles, shuffle_tiles=False
        )

        train_loader = DataLoader(
            train_dataset,
            batch_size=args.batch_size,
            sampler=get_sampler(train_labels),
            num_workers=args.num_workers,
            collate_fn=slide_collate_fn,
            pin_memory=True
        )
        test_loader = DataLoader(
            test_dataset,
            batch_size=1,
            sampler=SequentialSampler(test_dataset),
            num_workers=args.num_workers,
            collate_fn=slide_collate_fn,
            pin_memory=True
        )

        if args.hyperparam_search:
            print(f'Running hyperparameter search for fold {fold}...')
            best_lr, best_latent_dim, best_search_auc, hp_results = run_hyperparam_search_gigapath(
                train_loader, test_loader, args, fold, lr_search, latent_dim_search
            )
            print(f'Best hyperparameters: lr={best_lr}, latent_dim={best_latent_dim}, val_auc={best_search_auc:.4f}')
            hyperparam_summary[fold] = {
                'best_lr': best_lr,
                'best_latent_dim': best_latent_dim,
                'search_results': hp_results
            }
        else:
            best_lr = args.lr
            best_latent_dim = args.latent_dim

        print(f'Training with lr={best_lr}, latent_dim={best_latent_dim}')
        fold_metrics = train_fold(train_loader, test_loader, args, fold, lr=best_lr, latent_dim=best_latent_dim)
        fold_metrics['best_lr'] = best_lr
        fold_metrics['best_latent_dim'] = best_latent_dim
        cross_fold_metrics[fold] = fold_metrics
        all_val_aucs.append(fold_metrics['best_val_auc'])

        with open(os.path.join(args.save_folder, f'fold_{fold}_metrics.pkl'), 'wb') as f:
            pkl.dump(fold_metrics, f)

    print('Cross-Validation Results Summary (GigaPath Native Slide Encoder)')
    print(f'Task: {args.task}')
    print(f'Model: {args.model_arch}')
    print(f'Feature layers: {args.feat_layer}')
    print(f'Pretrained: {args.pretrained}')
    print(f'Fold AUCs: {all_val_aucs}')
    print(f'Mean AUC: {np.mean(all_val_aucs):.4f} +/- {np.std(all_val_aucs):.4f}')
    if hyperparam_summary:
        print('Hyperparameter search summary:')
        for fold_idx, hp_info in hyperparam_summary.items():
            print(f'  Fold {fold_idx}: lr={hp_info["best_lr"]}, latent_dim={hp_info["best_latent_dim"]}')

    cross_fold_metrics['summary'] = {
        'mean_auc': np.mean(all_val_aucs),
        'std_auc': np.std(all_val_aucs),
        'all_aucs': all_val_aucs
    }
    cross_fold_metrics['hyperparam_summary'] = hyperparam_summary

    with open(os.path.join(args.save_root, f'cross_val_results_{args.task}_gigapath.pkl'), 'wb') as f:
        pkl.dump(cross_fold_metrics, f)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='GigaPath Native Slide Encoder Cross-Validation Training')

    parser.add_argument('--cv_split_file', type=str, required=True,
                        help='Path to cross-validation splits pickle file')
    parser.add_argument('--feat_path', type=str, required=True,
                        help='Path to GigaPath feature folder (H5 files with features and coords)')
    parser.add_argument('--save_root', type=str, required=True,
                        help='Path to save results')

    parser.add_argument('--task', type=str, required=True,
                        choices=['cptac_lung', 'camelyon16', 'panda', 'plco_lung',
                                'ovarian', 'tcga_brca', 'tcga_prad', 'plco_breast',
                                'brca_gene', 'crc_gene'],
                        help='Downstream classification task')
    parser.add_argument('--task_type', type=str, required=True,
                        choices=['binary', 'multi'],
                        help='Task type: binary or multi-class')
    parser.add_argument('--fold_count', type=int, default=5,
                        help='Number of cross-validation folds')

    parser.add_argument('--model_arch', type=str, default='gigapath_slide_enc12l768d',
                        choices=['gigapath_slide_enc12l768d', 'gigapath_slide_enc24l1024d', 'gigapath_slide_enc12l1536d'],
                        help='GigaPath slide encoder architecture')
    parser.add_argument('--pretrained', type=str, default='hf_hub:prov-gigapath/prov-gigapath',
                        help='Path to pretrained GigaPath weights')
    parser.add_argument('--latent_dim', type=int, default=768,
                        help='Latent dimension of slide encoder')
    parser.add_argument('--feat_layer', type=str, default='11',
                        help='Which transformer layers to extract features from (e.g., "5-11" for layers 5 and 11)')
    parser.add_argument('--freeze', action='store_true',
                        help='Freeze pretrained slide encoder weights')

    parser.add_argument('--max_tiles', type=int, default=10000,
                        help='Maximum number of tiles per slide')
    parser.add_argument('--shuffle_tiles', action='store_true',
                        help='Shuffle tiles during training')

    parser.add_argument('--lr', type=float, default=2e-3,
                        help='Base learning rate')
    parser.add_argument('--min_lr', type=float, default=1e-6,
                        help='Minimum learning rate for cosine annealing')
    parser.add_argument('--weight_decay', type=float, default=0.05,
                        help='Weight decay')
    parser.add_argument('--layer_decay', type=float, default=0.95,
                        help='Layer-wise learning rate decay')
    parser.add_argument('--epochs', type=int, default=5,
                        help='Number of training epochs')
    parser.add_argument('--warmup_epochs', type=int, default=1,
                        help='Number of warmup epochs')
    parser.add_argument('--batch_size', type=int, default=1,
                        help='Batch size')
    parser.add_argument('--gc', type=int, default=32,
                        help='Gradient accumulation steps')
    parser.add_argument('--num_workers', type=int, default=4,
                        help='Number of data loading workers')
    parser.add_argument('--lr_scheduler', type=str, default='cosine',
                        help='Learning rate scheduler')
    parser.add_argument('--fp16', action='store_true',
                        help='Use FP16 training')

    parser.add_argument('--gpu', type=str, default='0',
                        help='GPU device ID')

    parser.add_argument('--hyperparam_search', action='store_true',
                        help='Enable hyperparameter search for lr and latent dimensions')
    parser.add_argument('--lr_search', type=str, default=None,
                        help='Comma-separated list of learning rates to search (e.g., "1e-4,5e-4,1e-3")')
    parser.add_argument('--latent_dim_search', type=str, default=None,
                        help='Comma-separated list of latent dimensions to search (e.g., "512,768,1024")')

    args = parser.parse_args()

    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    args.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    main(args)
