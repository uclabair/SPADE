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
    PatchDataset_PLCO, 
    PatchDataset
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

def evaluate(val_loader, model, criterion, args):
    softmax = nn.Softmax(dim = 1)

    total_start = time.time()
    val_loss_prev = 1000000
    n_epochs = args.epochs

    auc_best = 0


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

            #logits, Y_prob, Y_hat, A_raw, results_dict = model(feature)

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
    best_threshold = 0.5

    Preds = np.array(Preds)

    if args.task_type == 'binary':
        results_arr = np.array(Preds > best_threshold).astype(int)
        auc_ = roc_auc_score(np.array(Y_true), np.array(Preds))
        precision = precision_score(Y_true, results_arr)
        recall = recall_score(Y_true, results_arr)
        f1 = f1_score(Y_true, results_arr)

        print('test auc', auc_)
        print(f"Precision: {precision:.4f}")
        print(f"Recall: {recall:.4f}")
        print(f"F1-score: {f1:.4f}")
        tn, fp, fn, tp = confusion_matrix(Y_true, results_arr).ravel()

        sensitivity = tp / (tp + fn)  
        specificity = tn / (tn + fp)  
        print(f"Sensitivity (Recall): {sensitivity:.4f}")
        print(f"Specificity: {specificity:.4f}")

        precision, recall, thresholds = precision_recall_curve(np.array(Y_true), Preds)
        auprc = auc(recall, precision)

        print(f"AUPRC: {auprc:.4f}")


    elif args.task_type == 'multi':
        print(Preds.shape)
        results = np.argmax(Preds, 1)
        print(results.shape)
        auc_ = roc_auc_score(np.array(Y_true), np.array(Preds), multi_class='ovr', average = 'macro')
        precision = precision_score(Y_true, results, average = 'macro')
        recall = recall_score(Y_true, results, average = 'macro')
        f1 = f1_score(Y_true, results, average = 'macro')

        print('test auc', auc_)
        print(f"Macro Precision: {precision:.4f}")
        print(f"Macro Recall: {recall:.4f}")
        print(f"Macro F1-score: {f1:.4f}")
        #tn, fp, fn, tp = confusion_matrix(Y_true, np.array(Preds>best_threshold)).ravel()

        #sensitivity = tp / (tp + fn)  
        #specificity = tn / (tn + fp)  
        #print(f"Sensitivity (Recall): {sensitivity:.4f}")
        #print(f"Specificity: {specificity:.4f}")

        #precision, recall, thresholds = precision_recall_curve(np.array(Y_true), Preds)
        #auprc = auc(recall, precision)

        #print(f"AUPRC: {auprc:.4f}")


    
def main(args):
    args.save_folder = os.path.join(args.save_root, 'checkpoints')
    os.makedirs(args.save_folder, exist_ok=True)

    test_df = pd.read_csv(args.test_df)

    if args.task == 'camelyon16':
        test_dataset = PatchDataset_Camelyon(test_df, mode = 'test', feat_path = args.feat_path)

    elif args.task == 'cptac_lung':
        test_dataset = PatchDataset_CPTAC(test_df, mode = 'test', feat_path = args.feat_path)
    
    elif args.task == 'panda':
        test_dataset = PatchDataset_PANDA(test_df, mode = 'test', feat_path = args.feat_path)
    
    elif args.task == 'plco_lung':
        test_dataset = PatchDataset_PLCOLung(
            test_df, mode = 'test', feat_path=args.feat_path
        )
    elif args.task == 'ovarian':
        test_dataset = PatchDataset_Ovarian(
            test_df, mode = 'test', feat_path=args.feat_path
        )
    elif args.task == 'tcga_brca':
        test_dataset = PatchDataset_TCGA(
            test_df, mode = 'test', feat_path= args.feat_path
        )

    elif args.task == 'tcga_prad':
        test_dataset = PatchDataset_TCGA(
            test_df, mode = 'test', feat_path= args.feat_path
        )

    elif args.task == 'plco_breast':
        test_dataset = PatchDataset_PLCOBreast(
            test_df, mode = 'test', feat_path= args.feat_path
        )
    elif args.task == 'brca_gene':
        test_dataset = PatchDataset_BRCAGene(
            test_df, mode = 'test', feat_path = args.feat_path
        )

    elif args.task == 'crc_gene':
        test_dataset = PatchDataset_CRCGene(
            test_df, mode = 'test', feat_path = args.feat_path
        )

    else:
        print('Pick valid argument!')
        return 0


    test_loader = DataLoader(test_dataset, batch_size = 1, shuffle = False, num_workers = 2, persistent_workers = True)
    
    if args.task == 'cptac_lung':
        criterion = nn.BCEWithLogitsLoss()
        if args.model_type == 'abmil':
            model = MIL_Attention_fc(size_arg = args.feature_type, n_classes=1).cuda()
        elif args.model_type == 'transmil':
            model = TransMIL(size_arg = args.feature_type, n_classes=1).cuda()



    elif args.task == 'camelyon16':
        criterion = nn.CrossEntropyLoss()
        if args.model_type == 'abmil':
            model = MIL_Attention_fc(size_arg = args.feature_type, n_classes = 3).cuda()
        elif args.model_type == 'transmil':
            model = TransMIL(size_arg = args.feature_type, n_classes=3).cuda()



    elif args.task == 'panda':
        criterion = nn.CrossEntropyLoss()
        if args.model_type == 'abmil':
            model = MIL_Attention_fc(size_arg = args.feature_type, n_classes = 6).cuda()
        elif args.model_type == 'transmil':
            model = TransMIL(size_arg = args.feature_type, n_classes=6).cuda()

    elif args.task == 'plco_lung':
        criterion = nn.BCEWithLogitsLoss()
        if args.model_type == 'abmil':
            model = MIL_Attention_fc(size_arg = args.feature_type, n_classes = 1).cuda()
        elif args.model_type == 'transmil':
            model = TransMIL(size_arg = args.feature_type, n_classes=1).cuda()

    elif args.task == 'ovarian':
        criterion = nn.CrossEntropyLoss()
        if args.model_type == 'abmil':
            model = MIL_Attention_fc(size_arg = args.feature_type, n_classes = 5).cuda()
        elif args.model_type == 'transmil':
            model = TransMIL(size_arg = args.feature_type, n_classes=5).cuda()
    
    elif args.task == 'tcga_brca':
        criterion = nn.BCEWithLogitsLoss()
        if args.model_type == 'abmil':
            model = MIL_Attention_fc(size_arg = args.feature_type, n_classes = 1).cuda()
        if args.model_type == 'transmil':
            model = TransMIL(size_arg = args.feature_type, n_classes=1).cuda()

    elif args.task == 'tcga_prad':
        criterion = nn.CrossEntropyLoss()
        if args.model_type == 'abmil':
            model = MIL_Attention_fc(size_arg = args.feature_type, n_classes = 4).cuda()
        if args.model_type == 'transmil':
            model = TransMIL(size_arg = args.feature_type, n_classes=4).cuda()

    elif args.task == 'plco_breast':
        criterion = nn.CrossEntropyLoss()
        if args.model_type == 'abmil':
            model = MIL_Attention_fc(size_arg = args.feature_type, n_classes = 3).cuda()
        if args.model_type == 'transmil':
            model = TransMIL(size_arg = args.feature_type, n_classes=3).cuda()


    elif args.task == 'brca_gene':
        criterion = nn.BCEWithLogitsLoss()
        if args.model_type == 'abmil':
            model = MIL_Attention_fc(size_arg = args.feature_type, n_classes = 1).cuda()

        if args.model_type == 'transmil':
            model = TransMIL(size_arg = args.feature_type, n_classes=1).cuda()
    
    elif args.task == 'crc_gene':
        criterion = nn.BCEWithLogitsLoss()
        if args.model_type == 'abmil':
            model = MIL_Attention_fc(size_arg = args.feature_type, n_classes = 1).cuda()

        if args.model_type == 'transmil':
            model = TransMIL(size_arg = args.feature_type, n_classes=1).cuda()



    state_dict = torch.load(args.checkpoint_to_test)
    model.load_state_dict(state_dict['model'])
    evaluate(test_loader, model, criterion, args)


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
        '--checkpoint_to_test', type = str, default = ''
    )
    parser.add_argument(
        '--model_type', type = str, default = 'abmil'
    )
    args = parser.parse_args()

    main(args)