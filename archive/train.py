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
from dataset import PatchDataset_PLCO, PatchDataset_CAM16
from models import MIL_Attention_fc
from torch.utils.data import DataLoader, Dataset
from sklearn.metrics import roc_auc_score, accuracy_score
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



feature_type = 'bleep'
dataset_type = 'PLCO' #'CAM16'

if dataset_type =='PLCO':
    train_df = pd.read_csv('/raid/mpleasure/PLCO/parsed_data/lung/splits/train_df.csv')
    val_df = pd.read_csv('/raid/mpleasure/PLCO/parsed_data/lung/splits/val_df.csv')
    test_df = pd.read_csv('/raid/mpleasure/PLCO/parsed_data/lung/splits/test_df.csv')

    train_ds = PatchDataset_PLCO(train_df, mode ='train')
    val_ds = PatchDataset_PLCO(val_df, mode ='val')
    test_ds = PatchDataset_PLCO(test_df, mode ='test')

    train_labels = list(train_df['label'])

elif dataset_type =='CAM16':
    train_df = pd.read_csv('/raid/Camelyon/splits/train_3c_df.csv')
    val_df = pd.read_csv('/raid/Camelyon/splits/val_3c_df.csv')
    test_df = pd.read_csv('/raid/Camelyon/splits/test_3c_df.csv')

    train_ds = PatchDataset_CAM16(train_df, mode ='train')
    val_ds = PatchDataset_CAM16(val_df, mode ='val')
    test_ds = PatchDataset_CAM16(test_df, mode ='test')

    train_labels = list(train_df['binary'])


train_loader = DataLoader(train_ds, batch_size=1, sampler=get_sampler(train_labels), shuffle=False, num_workers=2, persistent_workers=True)
val_loader = DataLoader(val_ds, batch_size=1, shuffle=False, num_workers=2, persistent_workers=True)
# test_loader = DataLoader(test_ds, batch_size=1, shuffle=False, num_workers=2, persistent_workers=True)

bce_loss = nn.BCEWithLogitsLoss()  #nn.CrossEntropyLoss() #

model = MIL_Attention_fc(size_arg=feature_type, n_classes=1).cuda()

optimizer = torch.optim.AdamW(params=model.parameters(), lr=1e-4)
scheduler = CosineAnnealingLR(optimizer, T_max=20, eta_min=0)

softmax = nn.Softmax(dim=1)

total_start = time.time()
val_loss_prev = 100000
n_epochs=20
auc_best = 0

for epoch in range(n_epochs):
    model.train()
    epoch_loss = 0
    Preds = []
    Y_true = []
    epoch_ce_loss_list = []
    progress_bar = tqdm(enumerate(train_loader), total=len(train_loader), ncols=110)
    progress_bar.set_description(f"Epoch {epoch}")
    for step, batch in progress_bar:
        feature = batch["feature"].type('torch.FloatTensor').cuda()
        # label = batch["label"].type('torch.LongTensor').cuda() #
        label = batch["label"].type('torch.FloatTensor').cuda()

        optimizer.zero_grad()
        
        logits,  Y_prob, Y_hat, A_raw, results_dict = model(feature)
        logits = logits[:, 0]
        loss = bce_loss(logits, label)


        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()

        Y_true.extend(label.cpu().detach().numpy().astype(np.uint8))
        # Preds.extend(torch.sigmoid(logits).cpu().detach().numpy())

        progress_bar.set_postfix({"loss": epoch_loss / (step + 1)})
        # Y_true.extend(label.cpu().detach().numpy().astype(np.uint8))
        Preds.extend(torch.sigmoid(logits).cpu().detach().numpy())

        # Preds.extend(softmax(logits).cpu().detach().numpy())


    epoch_ce_loss_list.append(epoch_loss / (step + 1))
    print('train loss', epoch_loss / (step + 1))
    auc = roc_auc_score(np.array(Y_true).astype(np.uint8), np.array(Preds))#, multi_class='ovr') 
    # acc = accuracy_score(np.array(Y_true).astype(np.uint8), np.array(Preds) >=0.5) 
    # accs.append(auc)
    # self.scheduler.step(auc)
    print('Train auc={0}'.format(auc))

    Preds = []
    Y_true = []
    val_ce_epoch_loss_list = []
    model.train(False)
    epoch_loss = 0
    progress_bar = tqdm(enumerate(val_loader), total=len(val_loader), ncols=110)
    # progress_bar.set_description(f"Epoch {epoch}")
    
    with torch.no_grad():
        for val_step, batch in progress_bar:
        # for val_step, batch in enumerate(val_loader, start=1):
            feature = batch["feature"].type('torch.FloatTensor').cuda()
            # label = batch["label"].type('torch.LongTensor').cuda() #
            label = batch["label"].type('torch.FloatTensor').cuda()
        
            logits,  Y_prob, Y_hat, A_raw, results_dict = model(feature)
            logits = logits[:, 0]
            loss = bce_loss(logits, label)
    
            epoch_loss += loss.item()

            progress_bar.set_postfix({"loss": epoch_loss / (val_step + 1)})

            Y_true.extend(label.cpu().detach().numpy().astype(np.uint8))
            Preds.extend(torch.sigmoid(logits).cpu().detach().numpy())
            # Preds.extend(softmax(logits).cpu().detach().numpy())

    val_loss = epoch_loss / (val_step + 1)
    val_ce_epoch_loss_list.append(val_loss)
    # print('Val loss={0}'.format(val_loss))
    auc = roc_auc_score(np.array(Y_true).astype(np.uint8), np.array(Preds))#, multi_class='ovr') 
    # acc = accuracy_score(np.array(Y_true).astype(np.uint8), np.array(Preds) >=0.5) 
    # accs.append(auc)
    scheduler.step()
    print('Val auc={0}'.format(auc))


    if auc > auc_best:
        print('SAVED')
        auc_best = auc
        torch.save({
        'model': model.state_dict()
        },'/raid/eredekop/071024_ST/baselines/checkpoints/092124_ABMIL_{0}_norm_{1}'.format(feature_type, dataset_type))


    # Preds = []
    # Y_true = []
    # model.train(False)
    # val_loss = 0
    # progress_bar = tqdm(enumerate(test_loader), total=len(test_loader), ncols=110)
    # # progress_bar.set_description(f"Epoch {epoch}")
    # with torch.no_grad():
    #     for val_step, batch in progress_bar:
    #         feature = batch["feature"].type('torch.FloatTensor').cuda()
    #         # label = batch["label"].type('torch.LongTensor').cuda() #
    #         label = batch["label"].type('torch.FloatTensor').cuda()
        
    #         logits,  Y_prob, Y_hat, A_raw, results_dict = model(feature)
    #         logits = logits[:, 0]
    #         loss = bce_loss(logits, label)
    
    #         epoch_loss += loss.item()

    #         progress_bar.set_postfix({"loss": epoch_loss / (val_step + 1)})

    #         Y_true.extend(label.cpu().detach().numpy().astype(np.uint8))
    #         Preds.extend(torch.sigmoid(logits).cpu().detach().numpy())
    #         # Preds.extend(softmax(logits).cpu().detach().numpy())

    # val_loss /= val_step
    # val_ce_epoch_loss_list.append(val_loss)
    
    # auc = roc_auc_score(np.array(Y_true).astype(np.uint8), np.array(Preds))#, multi_class='ovr') 
    # print('test auc', auc)
