import os
import shutil
import tempfile
import pickle
import time
import ast
import pandas as pd
import argparse, os, sys, datetime, glob, importlib, csv
from torch.optim.lr_scheduler import ReduceLROnPlateau, CosineAnnealingLR
from sklearn.metrics import roc_auc_score, accuracy_score, precision_score, recall_score, f1_score,  auc, roc_curve, confusion_matrix, precision_recall_curve

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

    attentions_all = []
    model.train(False)
    epoch_loss = 0

    progress_bar = tqdm(enumerate(val_loader), total = len(val_loader), ncols = 110)
    preds = []

    with torch.no_grad():
        for val_step, batch in progress_bar:
            feature = batch['feature'].type('torch.FloatTensor').cuda()
            label = batch['label'].type('torch.FloatTensor').cuda()

            if args.model_type == 'transmil':
                logits = model(data=feature)
            else:
                logits, Y_prob, Y_hat, attns, results_dict = model(feature)

            #logits, Y_prob, Y_hat, A_raw, results_dict = model(feature)
            
            preds.append(torch.sigmoid(logits).cpu().detach().numpy())
            attentions_all.append(attns.detach().cpu().numpy().squeeze())

            progress_bar.set_postfix({"steps": (val_step + 1)})

    attentions_all = np.array(attentions_all, dtype = object)

    np.save(os.path.join(f'/raid/mpleasure/data_deepstorage/st_projects/attention_visuals', f'{args.task}_{args.exp_name}.npy'), attentions_all)
    np.save(os.path.join(f'/raid/mpleasure/data_deepstorage/st_projects/attention_visuals', f'{args.task}_{args.exp_name}_preds.npy'), preds)


    
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