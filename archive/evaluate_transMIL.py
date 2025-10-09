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
from torch.utils.data import Dataset
from dataset import PatchDataset_PLCO
from models import MIL_Attention_fc, TransMIL
from torch.utils.data import DataLoader, Dataset
from sklearn.metrics import roc_auc_score, accuracy_score, precision_score, recall_score, f1_score,  auc, roc_curve, confusion_matrix, precision_recall_curve
np.random.seed(0)
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="1"
import torch.nn.functional as F

def read_pickle(filename):
    with open(filename, 'rb') as file:
        data = pickle.load(file)
    return data

def get_sampler(labels_train):
    # print('sampler')
    # print(labels_train)
    class_idx, class_sample_count = np.unique(labels_train, return_counts=True)
    class_sample_count = class_sample_count[class_idx]
    class_weights = 1 / torch.Tensor(class_sample_count)
    sample_weights = class_weights[labels_train]
    sampler = torch.utils.data.WeightedRandomSampler(
        sample_weights, num_samples=len(labels_train), replacement=True
    )
    return sampler


dataset = 'bleep'

dataset_type = 'PLCO'
if dataset_type =='PLCO':
    train_df = pd.read_csv('/raid/mpleasure/PLCO/parsed_data/lung/splits/train_df.csv')
    val_df = pd.read_csv('/raid/mpleasure/PLCO/parsed_data/lung/splits/val_df.csv')
    test_df = pd.read_csv('/raid/mpleasure/PLCO/parsed_data/lung/splits/test_df.csv')

    train_ds = PatchDataset_PLCO(train_df, mode ='train')
    val_ds = PatchDataset_PLCO(val_df, mode ='val')
    test_ds = PatchDataset_PLCO(test_df, mode ='test')

    train_labels = list(train_df['label'])

elif dataset_type =='CAM16':
    train_df = pd.read_csv('/raid/Camelyon/splits/train_df.csv')
    val_df = pd.read_csv('/raid/Camelyon/splits/val_df.csv')
    test_df = pd.read_csv('/raid/Camelyon/splits/test_df.csv')

    train_ds = PatchDataset_CAM16(train_df, mode ='train')
    val_ds = PatchDataset_CAM16(val_df, mode ='val')
    test_ds = PatchDataset_CAM16(test_df, mode ='test')

    train_labels = list(train_df['binary'])




val_loader = DataLoader(val_ds, batch_size=1, shuffle=False, num_workers=2, persistent_workers=True)
test_loader = DataLoader(test_ds, batch_size=1, shuffle=False, num_workers=2, persistent_workers=True)

bce_loss = nn.BCEWithLogitsLoss() 

model = TransMIL(n_classes=1,size_arg=dataset).cuda()

state_dict = torch.load('/raid/eredekop/071024_ST/baselines/checkpoints/092124_TRANSMIL_{0}_norm_PLCO'.format(dataset))

model.load_state_dict(state_dict['model'])


total_start = time.time()
val_loss_prev = 100000
auc_best = 0

# Preds = []
# Y_true = []
# model.train(False)
# val_loss = 0
# progress_bar = tqdm(enumerate(val_loader), total=len(val_loader), ncols=110)
# with torch.no_grad():
#     for val_step, batch in progress_bar:
#         feature = batch["feature"].type('torch.FloatTensor').cuda()
#         label = batch["label"].type('torch.FloatTensor').cuda()
    
#         logits,  Y_prob, Y_hat, A_raw, results_dict = model(feature)
#         logits = logits[:, 0]
#         loss = bce_loss(logits, label)

#         # epoch_loss += loss.item()

#         # progress_bar.set_postfix({"loss": epoch_loss / (val_step + 1)})

#         Y_true.extend(label.cpu().detach().numpy().astype(np.uint8))
#         Preds.extend(torch.sigmoid(logits).cpu().detach().numpy())
# fpr, tpr, thresholds = roc_curve(Y_true, Preds)
# roc_auc = auc(fpr, tpr)
# j_scores = tpr - fpr
# best_index = np.argmax(j_scores)
# best_threshold = thresholds[best_index]
# print('best_threshold', best_threshold)

best_threshold = 0.5
Preds = []
Y_true = []
model.train(False)
val_loss = 0
progress_bar = tqdm(enumerate(test_loader), total=len(test_loader), ncols=110)
with torch.no_grad():
    for val_step, batch in progress_bar:
        feature = batch["feature"].type('torch.FloatTensor').cuda()
        label = batch["label"].type('torch.FloatTensor').cuda()
        logits = model(data=feature)
        logits = logits[:, 0]
        loss = bce_loss(logits, label)

        # epoch_loss += loss.item()

        # progress_bar.set_postfix({"loss": epoch_loss / (val_step + 1)})

        Y_true.extend(label.cpu().detach().numpy().astype(np.uint8))
        Preds.extend(torch.sigmoid(logits).cpu().detach().numpy())

# val_loss /= val_step
# val_ce_epoch_loss_list.append(val_loss)
Preds = np.array(Preds)
print(np.min(Preds), np.max(Preds))
auc_ = roc_auc_score(np.array(Y_true).astype(np.uint8), np.array(Preds)) 
precision = precision_score(Y_true, np.array(Preds>best_threshold).astype(np.uint8) )
recall = recall_score(Y_true, np.array(Preds>best_threshold).astype(np.uint8) )
f1 = f1_score(Y_true, np.array(Preds>best_threshold).astype(np.uint8) )

print('test auc', auc_)
print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"F1-score: {f1:.4f}")
tn, fp, fn, tp = confusion_matrix(Y_true, np.array(Preds>best_threshold).astype(np.uint8) ).ravel()

sensitivity = tp / (tp + fn)  
specificity = tn / (tn + fp)  
print(f"Sensitivity (Recall): {sensitivity:.4f}")
print(f"Specificity: {specificity:.4f}")

precision, recall, thresholds = precision_recall_curve(np.array(Y_true).astype(np.uint8), Preds)
auprc = auc(recall, precision)

print(f"AUPRC: {auprc:.4f}")
np.save('./cache/preds_trnas_{0}.npy'.format(dataset), np.array(Preds))
np.save('./cache/y_true_trans_{0}.npy'.format(dataset), np.array(Y_true))