import os
import shutil
import tempfile
import pickle
import time
import ast
import pandas as pd
import argparse, os, sys, datetime, glob, importlib, csv
from torch.optim.lr_scheduler import ReduceLROnPlateau, CosineAnnealingLR
from sklearn.metrics import (
    roc_auc_score, 
    accuracy_score, 
    precision_score, 
    recall_score, 
    f1_score,  
    auc, 
    roc_curve, 
    confusion_matrix, 
    precision_recall_curve)
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
    PatchDatasetCV
    )
from models import MIL_Attention_fc, TransMIL
from torch.utils.data import DataLoader, Dataset
from sklearn.metrics import roc_auc_score, accuracy_score
np.random.seed(0)
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="1"
import torch.nn.functional as F
import argparse

class_count_mapping = {
    'cptac_lung': 1,
    'camelyon16': 3,
    'panda': 6,
    'plco_lung': 1,
    'ovarian': 5,
    'tcga_brca': 1,
    'tcga_prad': 4,
    'plco_breast': 3,
    'brca_gene': 1,
    'crc_gene': 1
}

def calculate_metrics(preds, y_true, task_type = 'binary'):
    if task_type == 'binary:'
        results_arr = np.array(preds > best_threshold).astype(int)
        auc_ = roc_auc_score(np.array(y_true), np.array(preds))
        precision = precision_score(y_true, results_arr)
        recall = recall_score(y_true, results_arr)
        f1 = f1_score(y_true, results_arr)
        tn, fp, fn, tp = confusion_matrix(y_true, results_arr).ravel()

        sensitivity = tp / (tp + fn)  
        specificity = tn / (tn + fp)  

        precision, recall, thresholds = precision_recall_curve(np.array(y_true), preds)
        auprc = auc(recall, precision)
        out_metrics = {
            'auc': auc,
            'precision': precision,
            'recall': recall,
            'f1': f1
            'auprc': auprc,
            'sensitivity': sensitivity,
            'specificity': specificity
        }
        return out_metrics
    elif task_type == 'multi':
        results = np.argmax(preds, 1)
        auc_ = roc_auc_score(np.array(y_true), np.array(preds), multi_class='ovr', average = 'macro')
        precision = precision_score(y_true, results, average = 'macro')
        recall = recall_score(y_true, results, average = 'macro')
        f1 = f1_score(y_true, results, average = 'macro')

        out_metrics = {
            'auc': auc,
            'precision_macro': precision,
            'recall_macro': recall,
            'f1_macro': f1
        }
        return out_metrics
    else:
        raise ValueError('Select valid task type!')
        return 0

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

def get_model_loss(args):
    if args.task in ['cptac_lung', 'plco_lung', 'tcga_brca', 'brca_gene', 'crc_gene']:
        criterion = nn.BCEWithLogitsLoss()

    else:
        criterion = nn.CrossEntropyLoss()

    if args.model_type == 'abmil':
        model = MIL_Attention_fc(
            size_arg = args.feature_type, 
            n_classes=class_count_mapping[args.task]).cuda()
    else:
        model = TransMIL(
            size_arg = args.feature_type, 
            n_classes=class_count_mapping[args.task]).cuda()

    return model, criterion

def train(train_loader, val_loader, model, criterion, args, save_folder, fold = None):
    optimizer = torch.optim.AdamW(params = model.parameters(), lr = args.lr)
    scheduler = CosineAnnealingLR(optimizer, T_max = 20, eta_min = 0)

    softmax = nn.Softmax(dim = 1)

    best_epoch = None

    total_start = time.time()
    val_loss_prev = 1000000
    n_epochs = args.epochs

    auc_best = 0

    fold_metrics = {}

    for epoch in range(n_epochs):

        model.train()
        epoch_loss = 0
        Preds = []
        Y_true = []
        epoch_ce_loss_list = []

        fold_metrics[epoch] = {}

        progress_bar = tqdm(enumerate(train_loader), total = len(train_loader), ncols = 110)
        progress_bar.set_description(f'Epoch: [{epoch}/{n_epochs}], Fold: {fold}')

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
        train_metrics = calculate_metrics(Preds, Y_true, task_type = args.task_type)
        fold_metrics[epoch]['train'] = train_metrics

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
        val_metrics = calculate_metrics(Preds, Y_true, task_type = args.task_type)
        fold_metrics[epoch]['val'] = val_metrics
        print(f'Val AUC: {val_metrics['auc']}')
        if val_metrics['auc'] > auc_best:
            auc_best = val_metrics['auc']
            best_epoch = epoch

        scheduler.step()

        print('Saving model...')
        torch.save({
            'model': model.state_dict(),
        }, f'{save_folder}/{args.exp_name}_{args.task}_{epoch}_{auc}.pt')

    fold_metrics['best_epoch'] = best_epoch

def main(args):

    fold_count = args.fold_count
    with open(args.cv_split_file, 'rb') as f:
        cv_folds = pkl.load(f)

    cross_fold_metrics = {}

    for i in range(fold_count):
        print(f'Starting fold: {i}')
        save_folder = os.path.join(args.save_root, f'checkpoints_fold_{i}')
        os.makedirs(save_folder, exist_ok=True)

        train_names = cv_folds[i]['train']['train_names']
        train_labels = cv_folds[i]['train']['train_labels']
        test_names = cv_folds[i]['test']['test_names']
        test_labels = cv_folds[i]['test']['test_labels']

        train_dataset = PatchDatasetCV(
            train_names, train_labels,
            feat_path = args.feat_path, h5_file = args.h5_file)
        test_dataset = PatchDatasetCV(
            test_names, test_labels,
            feat_path = args.feat_path, h5_file = args.h5_file)

        train_loader = DataLoader(train_dataset, batch_size = 1, sampler = get_sampler(train_labels), shuffle = False, num_workers = 2, persistent_workers = True)
        val_loader = DataLoader(val_dataset, batch_size = 1, shuffle = False, num_workers = 2, persistent_workers = True)

        model, criterion = get_model_loss(args)
        fold_metrics = train(train_loader, val_loader, model, criterion, args, save_folder, fold = i)
        cross_fold_metrics[i] = fold_metrics
    
    with open(os.path.join(args.save_root, f'cross_val_results_{args.task}.pkl'), 'wb') as f:
        pkl.dump(cross_fold_metrics, f)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--feature_type', type = str, default = 'uni'
    )
    parser.add_argument(
        '--cv_split_file', type = str, default = None
    )
    parser.add_argument(
        '--fold_count', type = int, default = 5
    )
    parser.add_argument(
        '--h5_file', type = bool, default = False
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