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
from models import MIL_Attention_fc, TransMIL, LinearProbingModel
from sklearn.linear_model import LogisticRegression
from torch.utils.data import DataLoader, Dataset
from sklearn.metrics import roc_auc_score, accuracy_score
np.random.seed(0)
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="1"
import torch.nn.functional as F
import argparse
import pickle as pkl

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
    preds = np.array(preds)
    best_threshold = 0.5
    if task_type == 'binary':
        preds = preds[:, 1]
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
            'auc': auc_,
            'precision': precision,
            'recall': recall,
            'f1': f1,
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
            'auc': auc_,
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

def load_data_lr(args, train_names, test_names, train_labels, test_labels):
    
    train_features = []
    test_features = []
    train_labels_ = []
    test_labels_ = []

    for name, label in zip(train_names, train_labels):
        try:
            if args.h5_file:
                feature_file = os.path.join(args.feat_path, f'{name}.h5')
                features = load_h5_features(feature_file)
            else:
                feature_bag = os.path.join(args.feat_path, f'{name}.npy')
                features = np.array(np.load(feature_bag, allow_pickle = True))
        except:
            print(f'Warning missing {name}')
            continue
    
        train_features.append(features)
        train_labels_.append(label)


    for name, label in zip(test_names, test_labels):
        try:
            if args.h5_file:
                feature_file = os.path.join(args.feat_path, f'{name}.h5')
                features = load_h5_features(feature_file)
            else:
                feature_bag = os.path.join(args.feat_path, f'{name}.npy')
                features = np.array(np.load(feature_bag, allow_pickle = True))
        except:
            print(f'Warning missing {name}')
            continue
    
        test_features.append(features)
        test_labels_.append(label)

    return np.array(train_features), np.array(test_features), np.array(train_labels_), np.array(test_labels_)
    

def main(args):

    fold_count = args.fold_count
    with open(args.cv_split_file, 'rb') as f:
        cv_folds = pkl.load(f)

    cross_fold_metrics = {}

    for i in range(fold_count):
        print(f'Starting fold: {i}')
        save_folder = os.path.join(args.save_root, f'checkpoints_fold_{i}')
        os.makedirs(save_folder, exist_ok=True)

        train_names = cv_folds[i]['train']['train_ids']
        train_labels = cv_folds[i]['train']['train_labels']
        test_names = cv_folds[i]['test']['test_ids']
        test_labels = cv_folds[i]['test']['test_labels']

        train_features, test_features, train_labels, test_labels = load_data_lr(
            args,
            train_names,
            test_names,
            train_labels, 
            test_labels
        )

        NUM_C = 2
        COST = (train_features.shape[1] * NUM_C)/100
        clf = LogisticRegression(C = COST, max_iter = 10000, verbose = 0, random_state = 0)
        clf.fit(X = train_features, y = train_labels)
        pred_labels = clf.predict(X = test_features)
        pred_scores = clf.predict_proba(X = test_features)

        print(pred_scores.shape)
        print(test_labels.shape)
        print(np.unique(test_labels))

        test_metrics = calculate_metrics(pred_scores, test_labels, task_type = args.task_type)
        with open(os.path.join(save_folder, f'fold_{i}_metrics.pkl'), 'wb') as f:
            pkl.dump(test_metrics, f)

        cross_fold_metrics[i] = test_metrics
    
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
        '--batch_size', default = 1, type = int
    )
    args = parser.parse_args()

    main(args)