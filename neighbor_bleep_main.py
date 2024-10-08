import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from models import CLIPModelAttn, CLIPModel_Neighbor
#from utils import AvgMeter
from PIL import Image
import torchvision.transforms.functional as TF
import random
from dataset import STDataset_PreLoad, STDataset_Neighbors
import time
import torch.utils.tensorboard as tensorboard
from logger import *
from scheduler import *
import argparse
from utils import *

class AverageMeter(object):
    def __init__(self, name, fmt = ':f'):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n = 1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum/self.count

    def __str__(self):
        fmtstr = "{name} {val" + self.fmt + "} ({avg" + self.fmt + "})"
        return fmtstr.format(**self.__dict__)

    def get_avg(self):
        return self.avg

def save_model(model, CFG, epoch, model_name):
    #if best_model:
    #    checkpoint_path = os.path.join(CFG.checkpoint_dir, f'{CFG.model_name}_best.pt')
    #else:
    checkpoint_path = os.path.join(CFG.checkpoint_dir, f'{CFG.model_name}_{epoch}_{model_name}.pt')

    torch.save(model.state_dict(), checkpoint_path)


def val_step(model, val_loader, CFG, epoch, tb_writer = None, loss_meter_val = None):
    model.eval()

    running_loss = 0.0
    total_samples = 0.0

    pbar = tqdm(enumerate(val_loader), total = len(val_loader), desc = f'Val Epoch: {epoch + 1}')
    log_interval = 100
    
    with torch.no_grad():
        for i, batch in pbar:
            batch = {k: v.cuda() for k, v in batch.items() if k in [
                'image_features', 'reduced_expression', 
                'image_neighbors', 'gene_neighbors', 'image_mask', 'gene_mask']}
            batch_size = len(batch)
            total_samples += batch_size

            loss = model(batch)
            running_loss += loss.item()

            loss_meter_val.update(loss.item(), batch['image_features'].size(0))

            if tb_writer and i % log_interval == 0:
                step = epoch * len(val_loader) + i
                tb_writer.add_scalar('val/loss', loss.item(), step)

    logging.info(f"Val Epoch {epoch + 1}, {len(val_loader)} completed, "
                            f"Val Loss: {loss_meter_val.avg:.4f}")
    
    return loss_meter_val.avg


def train_step(
        model, 
        train_loader, 
        optimizer, 
        CFG, epoch, 
        scaler = None, 
        tb_writer = None, 
        scheduler = None, 
        loss_meter = None):
    
    model.train()
    total_loss = 0.0
    total_samples = 0.0

    pbar = tqdm(enumerate(train_loader), total = len(train_loader), desc = f'Train Epoch: {epoch + 1}')
    log_interval = CFG.log_interval

    for i, batch in pbar:
        batch = {k: v.cuda() for k, v in batch.items() if k in [
            'image_features', 
            'reduced_expression', 
            'image_neighbors', 'gene_neighbors', 'image_mask', 'gene_mask']}

        batch_size = len(batch)
        total_samples += batch_size

        loss = model(batch)
        optimizer.zero_grad()
        
        if CFG.use_scaler:
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
        else:
            loss.backward()
            optimizer.step()

        loss_meter.update(loss.item(), batch['image_features'].size(0))
        if tb_writer and i % log_interval == 0:
            step = epoch * len(train_loader) + i
            tb_writer.add_scalar('train/loss', loss.item(), step)
        
        if (i + 1) % log_interval == 0:
            logging.info(f"Epoch {epoch + 1}, Iteration {i + 1}/{len(train_loader)}, "
                         f"Loss: {loss_meter.avg:.4f}, ")

    if scheduler is not None:
        scheduler.step()

    logging.info(f"Epoch {epoch + 1}, {len(train_loader)} completed, "
                 f"Loss: {loss_meter.avg:.4f}, LR: {scheduler.get_lr()}")
    
    return model, loss_meter.avg, scaler, optimizer

        
def train_model(model, train_loader, val_loader, CFG):
    optimizer = torch.optim.AdamW(model.parameters(), lr = CFG.lr, weight_decay = CFG.weight_decay)
    scaler = None
    scheduler = None

    loss_meter = AverageMeter('train')
    loss_meter_val = AverageMeter('val')
    
    if CFG.use_scheduler:
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, len(train_loader))

    if CFG.use_scaler:
        scaler = torch.cuda.amp.GradScaler(init_scale = 512.0, growth_factor = 1.5, backoff_factor = 0.5, growth_interval = 100)

    if CFG.use_tensorboard:
        tensorboard_path = os.path.join(CFG.output_dir, CFG.model_name, 'tensorboard')
        writer = tensorboard.SummaryWriter(tensorboard_path)

    for epoch in range(CFG.num_epochs):
        model, train_loss, scaler, optimizer = train_step(
            model, 
            train_loader, 
            optimizer, 
            CFG,
            epoch,
            scaler,
            writer,
            scheduler, 
            loss_meter)
        val_loss = val_step(model, val_loader, CFG, epoch, writer, loss_meter_val)
        save_model(model, CFG, epoch, model_name = f'train_{train_loss}_val_{val_loss}')


def main(CFG):
    folds = np.load(CFG.folds_path, allow_pickle=True)
    df = pd.read_csv(CFG.coords_data, index_col = 0)

    train_df = df[df['id'].isin(list(folds[0][0]))]
    val_df = df[df['id'].isin(list(folds[0][1]))]

    missing_id = ['NCBI855', 'NCBI854']
    val_df = val_df[~val_df['id'].isin(missing_id)]
    train_df = train_df[~train_df['id'].isin(missing_id)]

    # set up datasets
    train_dataset = STDataset_Neighbors(train_df, CFG)
    val_dataset = STDataset_Neighbors(val_df, CFG)

    train_loader = DataLoader(train_dataset, batch_size=CFG.batch_size, shuffle=True, num_workers=CFG.num_workers)
    val_loader = DataLoader(val_dataset, batch_size = CFG.batch_size, shuffle = False, num_workers = CFG.num_workers)

    os.makedirs(os.path.join(CFG.output_dir, CFG.model_name), exist_ok=True)
    CFG.log_path = None
    log_base_path = os.path.join(CFG.output_dir, CFG.model_name, 'logs')
    os.makedirs(log_base_path, exist_ok=True)

    log_filename = 'out.log'
    CFG.log_path = os.path.join(log_base_path, log_filename)
    CFG.log_level = logging.DEBUG if CFG.debug else logging.INFO
    setup_logging(CFG.log_path, CFG.log_level)

    if CFG.model_type == 'v1':
        model = CLIPModelAttn(
            CFG.image_embedding,
            CFG.spot_embedding,
            projection_dim = CFG.projection_dim,
            args = CFG)
    if CFG.model_type == 'v2':
        model = CLIPModel_Neighbor(
            CFG.temperature,
            CFG.image_embedding,
            CFG.spot_embedding,
            projection_dim = CFG.projection_dim,
            args = CFG)


    model = model.cuda()

    CFG.checkpoint_dir = os.path.join(CFG.output_dir, CFG.model_name, 'checkpoints')
    os.makedirs(CFG.checkpoint_dir, exist_ok=True)

    random.seed(CFG.seed)
    np.random.seed(CFG.seed)
    torch.manual_seed(CFG.seed)

    logging.info('Starting model training...')
    train_model(model, train_loader, val_loader, CFG)


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--hvg_matrix_path', type = str, default = "/raid/HEST-1K/071524_hvg_matrix/"
    )
    parser.add_argument(
        '--image_features_path', type = str, default =  "/raid/mpleasure/data_deepstorage/st_projects/visium/uni_20x_hest_patching_normalized/"
    )
    parser.add_argument(
        '--folds_path', type = str, default = '/raid/eredekop/071024_ST/data/90_10_split.npy'
    )
    parser.add_argument(
        '--coords_data', type = str, default = '/raid/mpleasure/data_deepstorage/st_projects/visium/coords_with_neighbors.csv'
    )
    parser.add_argument(
        '--output_dir', type = str, default = '/raid/mpleasure/data_deepstorage/st_projects/visium/bleepattn'
    )
    parser.add_argument(
        '--lr', type = float, default = 1e-4
    )
    parser.add_argument(
        '--weight_decay', type = float, default = 1e-2
    )
    parser.add_argument(
        '--batch_size', type = int, default = 16
    )
    parser.add_argument(
        '--num_workers', type = int, default = 8
    )
    parser.add_argument(
        '--num_epochs', type = int, default = 300
    )
    parser.add_argument(
        '--patience', type = int, default = 2
    )
    parser.add_argument(
        '--factor', type = float, default = 0.5
    )
    parser.add_argument(
        '--image_embedding', type = int, default = 1024
    )
    parser.add_argument(
        '--spot_embedding', type = int, default = 7968
    )
    parser.add_argument(
        '--temperature', type = float, default = 1.0
    )
    parser.add_argument(
        '--size', type = int, default = 224
    )
    parser.add_argument(
        '--num_projection_layers', type = int, default = 1
    )
    parser.add_argument(
        '--projection_dim', type = int, default = 512
    )
    parser.add_argument(
        '--dropout', type = float, default = 0.1
    )
    parser.add_argument(
        '--log_interval', type = int, default = 100
    )
    parser.add_argument(
        '--use_scheduler', type = bool, default = True
    )
    parser.add_argument(
        '--use_scaler', type = bool, default = False
    )
    parser.add_argument(
        '--use_tensorboard', type = bool, default = True
    )
    parser.add_argument(
        '--seed', type = int, default = 0
    )
    parser.add_argument(
        '--use_soft_labels', type = bool, default = True
    )
    parser.add_argument(
        '--debug', type = bool, default = False
    )
    parser.add_argument(
        '--lambda_entropy', type = float, default = 0.1
    )
    parser.add_argument(
        '--use_entropy_reg', type = bool, default = True
    )
    parser.add_argument(
        '--use_fuzzy_attention', type = bool, default = False
    )
    parser.add_argument(
        '--use_similarity', type = bool, default = True
    )
    parser.add_argument(
        '--num_heads', type = int, default = 8
    )
    parser.add_argument(
        '--add_center', type = bool, default = False
    )
    parser.add_argument(
        '--model_type', type = str, default = 'v2'
    )
    CFG = parser.parse_args()


    CFG.model_name = f'ST_bleep_neighbor_attention_lr_{CFG.lr}_neighbor_model_{CFG.model_type}_image_attn_only_embedding_dim_{CFG.projection_dim}'

    main(CFG)