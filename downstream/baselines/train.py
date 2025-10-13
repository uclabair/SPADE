import os
import shutil
import tempfile
import pickle
import time
import ast
import pandas as pd
import argparse, os, sys, datetime, glob, importlib, csv
from torch.optim.lr_scheduler import ReduceLROnPlateau, CosineAnnealingLR
from sklearn.metrics import roc_auc_score, accuracy_score
from torch import nn
from torch.nn import L1Loss, CrossEntropyLoss
from tqdm import tqdm
import scipy
from sklearn.metrics import recall_score
import torch
torch.manual_seed(0)
import random
random.seed(0)
import numpy as np
import os
from torch.utils.data import Dataset
from dataset import (
    PatchDataset_PLCO, 
    PatchDataset_Camelyon, 
    PatchDataset_CPTAC, 
    PatchDataset_PANDA, 
    PatchDataset_PLCOLung, 
    PatchDataset_Ovarian,
    PatchDataset_TCGA,
    PatchDataset_PLCOBreast,
    PatchDataset_BRCAGene,
    PatchDataset_CRCGene
    )
from models import MIL_Attention_fc, TransMIL
from torch.utils.data import DataLoader, Dataset
from sklearn.metrics import roc_auc_score, accuracy_score
np.random.seed(0)
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="1"
import torch.nn.functional as F
import argparse

def read_pickle(filename):
    with open(filename, 'rb') as file:
        data = pickle.load(file)
    return data

def get_sampler(labels_train):
    class_idx, class_sample_count = np.unique(labels_train, return_counts=True)
    class_sample_count = class_sample_count[class_idx]
    class_weights = 1 / torch.Tensor(class_sample_count)
    sample_weights = class_weights[labels_train]
    sampler = torch.utils.data.WeightedRandomSampler(
        sample_weights, num_samples=len(labels_train), replacement=True
    )
    return sampler

def train(train_loader, val_loader, model, criterion, args):
    optimizer = torch.optim.AdamW(params = model.parameters(), lr = args.lr)
    scheduler = CosineAnnealingLR(optimizer, T_max = 20, eta_min = 0)

    softmax = nn.Softmax(dim = 1)

    total_start = time.time()
    val_loss_prev = 1000000
    n_epochs = args.epochs

    auc_best = 0

    for epoch in range(n_epochs):

        model.train()
        epoch_loss = 0
        Preds = []
        Y_true = []
        epoch_ce_loss_list = []

        progress_bar = tqdm(enumerate(train_loader), total = len(train_loader), ncols = 110)
        progress_bar.set_description(f'Epoch: {epoch}')

        for step, batch in progress_bar:
            feature = batch['feature'].type('torch.FloatTensor').cuda()
            label = batch['label'].type('torch.FloatTensor').cuda()

            optimizer.zero_grad()
            if args.model_type == 'transmil':
                logits = model(data=feature)
            else:
                logits, Y_prob, Y_hat, A_raw, results_dict = model(feature)

            if args.task_type == 'binary':
                logits = logits[:, 0]
                loss = criterion(logits, label)

            elif args.task_type == 'multi':
                label = batch['label'].type('torch.LongTensor').cuda()
                loss = criterion(logits, label)

            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

            Y_true.extend(label.cpu().detach().numpy())
            progress_bar.set_postfix({"loss": epoch_loss / (step + 1)})

            if args.task_type == 'binary':
                Preds.extend(torch.sigmoid(logits).cpu().detach().numpy())
            elif args.task_type == 'multi':
                Preds.extend(torch.softmax(logits, dim = 1).cpu().detach().numpy())

        epoch_ce_loss_list.append(epoch_loss / (step + 1))
        print(f'Train loss: {epoch_loss / (step + 1)}')

        if args.task_type == 'binary':
            auc = roc_auc_score(np.array(Y_true), np.array(Preds))
        elif args.task_type == 'multi':
            auc = roc_auc_score(np.array(Y_true), np.array(Preds), multi_class='ovr', average = 'macro')

        print(f'Train auc: {auc}')


        Preds = []
        Y_true = []
        val_ce_epoch_loss_list = []
        model.train(False)
        epoch_loss = 0

        progress_bar = tqdm(enumerate(val_loader), total = len(val_loader), ncols = 110)

        with torch.no_grad():
            for val_step, batch in progress_bar:
                feature = batch['feature'].type('torch.FloatTensor').cuda()
                label = batch['label'].type('torch.FloatTensor').cuda()

                if args.model_type == 'transmil':
                    logits = model(data=feature)
                else:
                    logits, Y_prob, Y_hat, A_raw, results_dict = model(feature)

                if args.task_type == 'binary':
                    logits = logits[:, 0]
                    loss = criterion(logits, label)

                elif args.task_type == 'multi':
                    label = batch['label'].type('torch.LongTensor').cuda()
                    loss = criterion(logits, label)


                epoch_loss += loss.item()

                progress_bar.set_postfix({"loss": epoch_loss / (val_step + 1)})

                Y_true.extend(label.cpu().detach().numpy())

                if args.task_type == 'binary':
                    Preds.extend(torch.sigmoid(logits).cpu().detach().numpy())
                elif args.task_type == 'multi':
                    Preds.extend(torch.softmax(logits, dim = 1).cpu().detach().numpy())

        val_loss = epoch_loss / (val_step + 1)
        val_ce_epoch_loss_list.append(val_loss)

        if args.task_type == 'binary':
            auc = roc_auc_score(np.array(Y_true), np.array(Preds))
        elif args.task_type == 'multi':
            auc = roc_auc_score(np.array(Y_true), np.array(Preds), multi_class='ovr', average = 'macro')

        scheduler.step()
        print(f'Val auc: {auc}')


        print('Saving model...')
        torch.save({
            'model': model.state_dict(),
        }, f'{args.save_folder}/{args.exp_name}_{args.task}_{epoch}_{auc}.pt')


case_id_mapping = {
    'camelyon16' : ('image', 'binary'),
    'cptac_lung': ('Case_ID', 'label'),
    'panda': ('image_id', 'isup_grade'),
    'ovarian': ('image_id',  'numeric_label'),
    'tcga_brca': ('Case ID', 'label'),
    'tcga_prad': ('Case ID', 'label'),
    'plco_breast': ('plco_id', 'label_mapped'),
    'brca_gene': ('Case ID', 'label'),
    'crc_gene': ('Case ID', 'label'),



}

def main(args):
    args.save_folder = os.path.join(args.save_root, 'checkpoints')
    os.makedirs(args.save_folder, exist_ok=True)

    train_df = pd.read_csv(args.train_df)
    val_df = pd.read_csv(args.val_df)

    if args.task == 'plco_lung':
        train_dataset = PatchDataset_PLCOLung(
            train_df, mode = 'train', feat_path=args.feat_path
        )
        val_dataset = PatchDataset_PLCOLung(
            val_df, mode = 'val', feat_path= args.feat_path
        )
        train_labels = list(train_df['label'])
    
    else:
        try:
            id_col, label_col = case_id_mapping[args.task]
            train_labels = list(train_df[label_col])
        except:
            print('Pick valid task!')
            return 0
    

    if args.task != 'plco_lung':
        train_dataset = PatchDataset(
            train_df, 
            feat_path = args.feat_path, 
            id_col = id_col, label_col = label_col)
        val_dataset = PatchDataset(
            val_df, 
            feat_path = args.feat_path,
            id_col = id_col, label_col = label_col)


    train_loader = DataLoader(train_dataset, batch_size = 1, sampler = get_sampler(train_labels), shuffle = False, num_workers = 2, persistent_workers = True)
    val_loader = DataLoader(val_dataset, batch_size = 1, shuffle = False, num_workers = 2, persistent_workers = True)

    if args.task == 'cptac_lung':
        criterion = nn.BCEWithLogitsLoss()
        if args.model_type == 'abmil':
            model = MIL_Attention_fc(
                size_arg = args.feature_type, 
                n_classes=1).cuda()
        if args.model_type == 'transmil':
            print(args.feature_type)
            model = TransMIL(
                size_arg = args.feature_type, 
                n_classes=1).cuda()

    elif args.task == 'camelyon16':
        criterion = nn.CrossEntropyLoss()
        if args.model_type == 'abmil':
            model = MIL_Attention_fc(
                size_arg = args.feature_type, 
                n_classes = 3).cuda()
        if args.model_type == 'transmil':
            model = TransMIL(
                size_arg = args.feature_type, 
                n_classes=3).cuda()

    elif args.task == 'panda':
        criterion = nn.CrossEntropyLoss()
        if args.model_type == 'abmil':
            model = MIL_Attention_fc(
                size_arg = args.feature_type, 
                n_classes = 6).cuda()
        if args.model_type == 'transmil':
            model = TransMIL(
                size_arg = args.feature_type,
                n_classes=6).cuda()

    elif args.task == 'plco_lung':
        criterion = nn.BCEWithLogitsLoss()
        if args.model_type == 'abmil':
            model = MIL_Attention_fc(
                size_arg = args.feature_type, 
                n_classes = 1).cuda()
        if args.model_type == 'transmil':
            model = TransMIL(
                size_arg = args.feature_type, 
                n_classes=1).cuda()

    elif args.task == 'ovarian':
        criterion = nn.CrossEntropyLoss()
        if args.model_type == 'abmil':
            model = MIL_Attention_fc(
                size_arg = args.feature_type, 
                n_classes = 5).cuda()
        if args.model_type == 'transmil':
            model = TransMIL(
                size_arg = args.feature_type, 
                n_classes=5).cuda()

    elif args.task == 'tcga_brca':
        criterion = nn.BCEWithLogitsLoss()
        if args.model_type == 'abmil':
            model = MIL_Attention_fc(
                size_arg = args.feature_type, 
                n_classes = 1).cuda()
        if args.model_type == 'transmil':
            model = TransMIL(
                size_arg = args.feature_type, 
                n_classes=1).cuda()

    elif args.task == 'tcga_prad':
        criterion = nn.CrossEntropyLoss()
        if args.model_type == 'abmil':
            model = MIL_Attention_fc(
                size_arg = args.feature_type, 
                n_classes = 4).cuda()
        if args.model_type == 'transmil':
            model = TransMIL(
                size_arg = args.feature_type, 
                n_classes=4).cuda()

    elif args.task == 'plco_breast':
        criterion = nn.CrossEntropyLoss()
        if args.model_type == 'abmil':
            model = MIL_Attention_fc(
                size_arg = args.feature_type, 
                n_classes = 3).cuda()
        if args.model_type == 'transmil':
            model = TransMIL(
                size_arg = args.feature_type, 
                n_classes=3).cuda()

    elif args.task == 'brca_gene':
        criterion = nn.BCEWithLogitsLoss()
        if args.model_type == 'abmil':
            model = MIL_Attention_fc(
                size_arg = args.feature_type, 
                n_classes = 1).cuda()

        if args.model_type == 'transmil':
            model = TransMIL(
                size_arg = args.feature_type, 
                n_classes=1).cuda()
    
    elif args.task == 'crc_gene':
        criterion = nn.BCEWithLogitsLoss()
        if args.model_type == 'abmil':
            model = MIL_Attention_fc(
                size_arg = args.feature_type, 
                n_classes = 1).cuda()

        if args.model_type == 'transmil':
            model = TransMIL(
                size_arg = args.feature_type, 
                n_classes=1).cuda()



    train(train_loader, val_loader, model, criterion, args)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--feature_type', type = str, default = 'uni'
    )
    parser.add_argument(
        '--train_df', type = str, default = '/raid/Camelyon/splits/train_3c_df.csv'
    )
    parser.add_argument(
        '--val_df', type = str, default = '/raid/Camelyon/splits/val_3c_df.csv'
    )
    parser.add_argument(
        '--test_df', type = str, default = '/raid/Camelyon/splits/test_3c_df.csv'
    )
    parser.add_argument(
        '--feat_path', type = str, default = '/raid/Camelyon/Camelyon16/protoattn_feats/all_feats'
    )
    parser.add_argument(
        '--save_root', type = str, default = '/raid/Camelyon/Camelyon16/protoattn_feats'
    )
    parser.add_argument(
        '--exp_name', type = str, default = 'protoattn'
    )
    parser.add_argument(
        '--task', type = str, default = 'camleyon16'
    )
    parser.add_argument(
        '--task_type', type = str, default = 'multi'
    )
    parser.add_argument(
        '--lr', type = float, default = 1e-5
    )
    parser.add_argument(
        '--epochs', type = int, default = 20
    )
    parser.add_argument(
        '--model_type', default = 'abmil'
    )
    args = parser.parse_args()

    main(args)