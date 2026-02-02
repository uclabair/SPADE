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

DEFAULT_LR_SEARCH = [5e-6, 1e-5, 5e-5, 1e-4, 1e-3]
DEFAULT_HIDDEN_DIM_SEARCH = [128, 256, 384, 512]
DEFAULT_WD_SEARCH = [0.01]

def calculate_metrics(preds, y_true, task_type = 'binary'):
    preds = np.array(preds)
    best_threshold = 0.5
    if task_type == 'binary':
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
            'f1': f1
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

def get_model_loss(args, hidden_dim = None):

    if args.task in ['cptac_lung', 'plco_lung', 'tcga_brca', 'brca_gene', 'crc_gene']:
        criterion = nn.BCEWithLogitsLoss()
    else:
        criterion = nn.CrossEntropyLoss()

    if args.model_type == 'abmil':
        model = MIL_Attention_fc(
            size_arg = args.feature_type, 
            n_classes=class_count_mapping[args.task],
            hidden_dim = hidden_dim).cuda()
    elif args.model_type == 'transmil':
        model = TransMIL(
            size_arg = args.feature_type, 
            n_classes=class_count_mapping[args.task],
            hidden_dim = hidden_dim).cuda()
            
    return model, criterion


def train(train_loader, val_loader, model, criterion, args, save_folder, fold = None, lr = None, wd = None):
    lr_to_use = lr if lr is not None else args.lr
    optimizer = torch.optim.AdamW(params = model.parameters(), lr = lr_to_use, weight_decay = wd)
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
            elif args.model_type == 'abmil':
                logits, Y_prob, Y_hat, A_raw, results_dict = model(feature)
            else:
                logits = model(feature)

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
                elif args.model_type == 'abmil':
                    logits, Y_prob, Y_hat, A_raw, results_dict = model(feature)
                else:
                    logits = model(feature)

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
        val_auc = val_metrics['auc']
        fold_metrics[epoch]['val'] = val_metrics
        print(f"Val AUC: {val_metrics['auc']}")
        if val_metrics['auc'] > auc_best:
            auc_best = val_metrics['auc']
            best_epoch = epoch

        scheduler.step()

        print('Saving model...')
        torch.save({
            'model': model.state_dict(),
        }, f'{save_folder}/{args.exp_name}_{args.task}_{epoch}_{val_auc}.pt')

    fold_metrics['best_epoch'] = best_epoch

    with open(os.path.join(save_folder, f'fold_{fold}_metrics.pkl'), 'wb') as f:
        pkl.dump(fold_metrics, f)

    return fold_metrics

def run_hyperparam_search(train_loader, val_loader, args, save_folder, fold, lr_search, hidden_dim_search):
    best_val_auc = 0
    best_lr = args.lr
    best_hidden_dim = None
    hyperparam_results = []
    best_wd = None


    for lr in lr_search:
        for hidden_dim in hidden_dim_search:
            for wd in DEFAULT_WD_SEARCH:
                print(f'  Trying lr={lr}, hidden_dim={hidden_dim}, weight_decay = {wd}')
                model, criterion = get_model_loss(args, hidden_dim=hidden_dim)

                search_epochs = min(args.epochs, 10)
                optimizer = torch.optim.AdamW(params=model.parameters(), lr=lr, weight_decay = wd)
                scheduler = CosineAnnealingLR(optimizer, T_max=search_epochs, eta_min=0)

                softmax = nn.Softmax(dim=1)
                best_epoch_auc = 0
                best_epoch_f1 = 0

                for epoch in tqdm(range(search_epochs)):
                    model.train()
                    for step, batch in tqdm(enumerate(train_loader), total = len(train_loader)):
                        feature = batch['feature'].type('torch.FloatTensor').cuda()
                        label = batch['label'].type('torch.FloatTensor').cuda()
                        optimizer.zero_grad()
                        if args.model_type == 'transmil':
                            logits = model(data=feature)
                        elif args.model_type == 'abmil':
                            logits, Y_prob, Y_hat, A_raw, results_dict = model(feature)
                        else:
                            logits = model(feature)

                        if args.task_type == 'binary':
                            logits = logits[:, 0]
                            loss = criterion(logits, label)
                        elif args.task_type == 'multi':
                            label = batch['label'].type('torch.LongTensor').cuda()
                            loss = criterion(logits, label)

                        loss.backward()
                        optimizer.step()

                    model.eval()
                    Preds = []
                    Y_true = []
                    with torch.no_grad():
                        for batch in val_loader:
                            feature = batch['feature'].type('torch.FloatTensor').cuda()
                            label = batch['label'].type('torch.FloatTensor').cuda()
                            if args.model_type == 'transmil':
                                logits = model(data=feature)
                            elif args.model_type == 'abmil':
                                logits, Y_prob, Y_hat, A_raw, results_dict = model(feature)
                            else:
                                logits = model(feature)

                            Y_true.extend(label.cpu().detach().numpy())
                            if args.task_type == 'binary':
                                logits = logits[:, 0]
                                Preds.extend(torch.sigmoid(logits).cpu().detach().numpy())
                            elif args.task_type == 'multi':
                                Preds.extend(torch.softmax(logits, dim=1).cpu().detach().numpy())

                    val_metrics = calculate_metrics(Preds, Y_true, task_type=args.task_type)
                    if val_metrics['auc'] > best_epoch_auc:
                        best_epoch_auc = val_metrics['auc']
                        best_epoch_f1 = val_metrics['f1']

                    scheduler.step()

            print(f'Best val AUC: {best_epoch_auc:.4f}')
            hyperparam_results.append({
                'lr': lr,
                'hidden_dim': hidden_dim,
                'wd': wd,
                'val_auc': best_epoch_auc,
                'val_f1': best_epoch_f1 
            })

            if best_epoch_auc > best_val_auc:
                best_val_auc = best_epoch_auc
                best_lr = lr
                best_hidden_dim = hidden_dim
                best_wd = wd

            del model, optimizer, scheduler
            torch.cuda.empty_cache()

    return best_lr, best_hidden_dim, best_wd, best_val_auc, hyperparam_results

def evaluate_model(model, loader, args):
    """Evaluate model and return predictions and labels."""
    model.eval()
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for batch in loader:
            feature = batch['feature'].type('torch.FloatTensor').cuda()
            label = batch['label'].cpu().numpy()

            if args.model_type == 'transmil':
                logits = model(data=feature)
            elif args.model_type == 'abmil':
                logits, _, _, _, _ = model(feature)
            else:
                logits = model(feature)

            if args.task_type == 'binary':
                logits = logits[:, 0]
                preds = torch.sigmoid(logits).cpu().numpy()
            else:
                preds = torch.softmax(logits, dim=1).cpu().numpy()

            all_preds.extend(preds)
            all_labels.extend(label)

    return np.array(all_preds), np.array(all_labels)

def ensemble_predictions(all_fold_preds, task_type='binary'):
    stacked = np.stack(all_fold_preds, axis=0)
    return np.mean(stacked, axis=0)


def save_results_to_csv(results_list, save_path):
    df = pd.DataFrame(results_list)
    df.to_csv(save_path, index=False)
    print(f'Results saved to: {save_path}')
    return df


def main(args):
    timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')

    fold_count = args.fold_count
    with open(args.cv_split_file, 'rb') as f:
        cv_folds = pkl.load(f)

    if 'fold_0' in cv_folds:
        fold_keys = [f'fold_{i}' for i in range(args.fold_count)]
        
    else:
        fold_keys = list(range(args.fold_count))
    
    has_holdout = 'holdout_test' in cv_folds
    print(f'HAS HOLDOUT: {has_holdout}')

    all_results = []
    cross_fold_metrics = {}
    hyperparam_summary = {}
    hyperparam_results_all = []

    if args.hyperparam_search:
        lr_search = DEFAULT_LR_SEARCH
        hidden_dim_search = DEFAULT_HIDDEN_DIM_SEARCH
        
        print(f'Hyperparameter search enabled:')
        print(f'  Learning rates: {lr_search}')
        print(f'  Hidden dimensions: {hidden_dim_search}')
    else:
        lr_search = [args.lr]
        hidden_dim_search = [None]


    print('PHASE 1: Hyperparameter Search')
    hyperparam_scores = {}

    for i, fold_key in enumerate(fold_keys):
        print(f'Starting fold: {i}')
        save_folder = os.path.join(args.save_root, f'checkpoints_fold_{i}')
        os.makedirs(save_folder, exist_ok=True)

        fold_data = cv_folds[fold_key]
        train_names = fold_data['train']['train_ids']
        train_labels = fold_data['train']['train_labels']
        val_names = fold_data['val']['val_ids']
        val_labels = fold_data['val']['val_labels']

        train_dataset = PatchDatasetCV(
            train_names, train_labels,
            feat_path = args.feat_path, h5_file = args.h5_file)
        val_dataset = PatchDatasetCV(
            val_names, val_labels,
            feat_path = args.feat_path, h5_file = args.h5_file)

        train_loader = DataLoader(
            train_dataset, 
            batch_size = args.batch_size, 
            sampler = get_sampler(train_labels), 
            shuffle = False, num_workers = 2, persistent_workers = True)
        val_loader = DataLoader(
            val_dataset, 
            batch_size = args.batch_size, 
            shuffle = False, num_workers = 2, persistent_workers = True)

        
        if args.hyperparam_search:
            print(f'Running hyperparameter search for fold {i}...')
            best_lr, best_hidden_dim, best_wd, best_search_auc, hp_results = run_hyperparam_search(
                train_loader, val_loader, args, save_folder, i, lr_search, hidden_dim_search
            )
            print(f'Best hyperparameters: lr={best_lr}, hidden_dim={best_hidden_dim}, wd={best_wd}, val_auc={best_search_auc:.4f}')
            for hp_res in hp_results:
                key = (hp_res['lr'], hp_res['hidden_dim'], hp_res['wd'])
                if key not in hyperparam_scores:
                    hyperparam_scores[key] = []
                hyperparam_scores[key].append(hp_res['val_auc'])

                # Store for CSV
                hyperparam_results_all.append({
                    'task': args.task,
                    'model_type': args.model_type,
                    'feature_type': args.feature_type,
                    'phase': 'hyperparam_search',
                    'fold': i,
                    'lr': hp_res['lr'],
                    'hidden_dim': hp_res['hidden_dim'],
                    'wd': hp_res['wd'],
                    'val_auc': hp_res['val_auc'],
                    'val_f1': hp_res['val_f1'],
                    'timestamp': timestamp
                })
        else:
            best_lr = args.lr
            best_hidden_dim = None
            best_wd = 1e-5

        #print(f'Training with lr={best_lr}, hidden_dim={best_hidden_dim}')
        #model, criterion = get_model_loss(args, hidden_dim=best_hidden_dim)
        #fold_metrics = train(train_loader, test_loader, model, criterion, args, save_folder, fold=i, lr=best_lr)
        #fold_metrics['best_lr'] = best_lr
        #fold_metrics['best_hidden_dim'] = best_hidden_dim
        #cross_fold_metrics[i] = fold_metrics
    if args.hyperparam_search and hyperparam_scores:
        best_hp = max(hyperparam_scores.keys(), key=lambda k: np.mean(hyperparam_scores[k]))
        best_lr, best_hidden_dim = best_hp
        best_mean_auc = np.mean(hyperparam_scores[best_hp])
        print(f'\nBest hyperparameters: lr={best_lr}, hidden_dim={best_hidden_dim}, wd={best_wd}')
        print(f'Mean CV AUC: {best_mean_auc:.4f} +/- {np.std(hyperparam_scores[best_hp]):.4f}')
    else:
        best_lr = args.lr
        best_hidden_dim = None

    print('PHASE 2: Training Final Models')

    fold_models = [] 
    fold_val_aucs = []
    fold_val_f1s = []

    for fold_idx, fold_key in enumerate(fold_keys):
        print(f'\n--- Training Fold {fold_idx} ---')

        save_folder = os.path.join(args.save_root, f'checkpoints_fold_{fold_idx}')
        os.makedirs(save_folder, exist_ok=True)

        fold_data = cv_folds[fold_key]
        train_names = fold_data['train']['train_ids']
        train_labels = fold_data['train']['train_labels']
        val_names = fold_data['val']['val_ids']
        val_labels = fold_data['val']['val_labels']

        train_dataset = PatchDatasetCV(train_names, train_labels, feat_path=args.feat_path, h5_file=args.h5_file)
        val_dataset = PatchDatasetCV(val_names, val_labels, feat_path=args.feat_path, h5_file=args.h5_file)

        train_loader = DataLoader(train_dataset, batch_size=args.batch_size, sampler=get_sampler(train_labels),
                                  shuffle=False, num_workers=2, persistent_workers=True)
        val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False,
                                num_workers=2, persistent_workers=True)

        # Train model
        model, criterion = get_model_loss(args, hidden_dim=best_hidden_dim)
        fold_metrics = train(train_loader, val_loader, model, criterion, args, save_folder, fold=fold_idx, lr=best_lr, wd = best_wd)

        # Get best epoch's validation AUC
        best_epoch = fold_metrics.get('best_epoch', 0)
        if best_epoch is not None and best_epoch in fold_metrics:
            val_auc = fold_metrics[best_epoch]['val']['auc']
        else:
            val_auc = 0.0

        fold_val_aucs.append(val_auc)
        fold_val_f1s.append(fold_metrics[best_epoch]['val']['f1'])
        fold_models.append(model)

        cross_fold_metrics[fold_idx] = {
            'best_epoch': best_epoch,
            'val_auc': val_auc,
            'val_f1': fold_metrics[best_epoch]['val']['f1'],
            'best_lr': best_lr,
            'best_hidden_dim': best_hidden_dim
        }

        all_results.append({
            'task': args.task,
            'model_type': args.model_type,
            'feature_type': args.feature_type,
            'phase': 'cv_validation',
            'fold': fold_idx,
            'lr': best_lr,
            'hidden_dim': best_hidden_dim,
            'best_epoch': best_epoch,
            'val_auc': val_auc,
            'timestamp': timestamp
        })

    print(f'\nCV Validation AUCs: {fold_val_aucs}')
    print(f'Mean CV AUC: {np.mean(fold_val_aucs):.4f} +/- {np.std(fold_val_aucs):.4f}')

    print(f'\nCV Validation F1ss: {fold_val_f1s}')
    print(f'Mean CV F1: {np.mean(fold_val_f1s):.4f} +/- {np.std(fold_val_f1s):.4f}')


    print('PHASE 3: Holdout Test Evaluation (Ensemble)')

    holdout_data = cv_folds['holdout_test']
    test_names = holdout_data['test_ids']
    test_labels = holdout_data['test_labels']

    test_dataset = PatchDatasetCV(test_names, test_labels, feat_path=args.feat_path, h5_file=args.h5_file)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False,
                                num_workers=2, persistent_workers=True)

    all_fold_preds = []
    for fold_idx, model in enumerate(fold_models):
        preds, labels = evaluate_model(model, test_loader, args)
        all_fold_preds.append(preds)

        individual_metrics = calculate_metrics(preds, labels, task_type=args.task_type)
        print(f'Fold {fold_idx} model - Test AUC: {individual_metrics["auc"]:.4f}')

        all_results.append({
            'task': args.task,
            'model_type': args.model_type,
            'feature_type': args.feature_type,
            'phase': 'holdout_individual',
            'fold': fold_idx,
            'lr': best_lr,
            'hidden_dim': best_hidden_dim,
            'test_auc': individual_metrics['auc'],
            'test_f1': individual_metrics['f1'],
            'timestamp': timestamp
        })

    ensemble_preds = ensemble_predictions(all_fold_preds, args.task_type)
    ensemble_metrics = calculate_metrics(ensemble_preds, labels, task_type=args.task_type)

    print(f'\nEnsemble Test AUC: {ensemble_metrics["auc"]:.4f}')
    print(f'\nEnsemble Test F1: {ensemble_metrics["f1"]:.4f}')

    holdout_results = {
        'ensemble_auc': ensemble_metrics['auc'],
        'ensemble_f1': ensemble_metrics['f1'],
        'ensemble_metrics': ensemble_metrics,
        'individual_aucs': [calculate_metrics(p, labels, args.task_type)['auc'] for p in all_fold_preds],
        'individual_f1s': [calculate_metrics(p, labels, args.task_type)['f1'] for p in all_fold_preds]
    }

    all_results.append({
        'task': args.task,
        'model_type': args.model_type,
        'feature_type': args.feature_type,
        'phase': 'holdout_ensemble',
        'fold': 'ensemble',
        'lr': best_lr,
        'hidden_dim': best_hidden_dim,
        'test_holdout_auc': ensemble_metrics['auc'],
        'test_holdout_f1': ensemble_metrics['f1'],
        'timestamp': timestamp,
    })
    os.makedirs(args.save_root, exist_ok=True)
    results_dict = {
        'cross_fold_metrics': cross_fold_metrics,
        'best_lr': best_lr,
        'best_hidden_dim': best_hidden_dim,
        'cv_mean_auc': np.mean(fold_val_aucs),
        'cv_std_auc': np.std(fold_val_aucs),
        'cv_mean_f1s': np.mean(fold_val_f1s),
        'cv_std_f1s': np.std(fold_val_f1s),
        'holdout_results': holdout_results,
        'hyperparam_search_results': hyperparam_results_all,
        'args': vars(args)
    }
    with open(os.path.join(args.save_root, f'results_{args.task}.pkl'), 'wb') as f:
        pkl.dump(results_dict, f)

    csv_path = os.path.join(args.save_root, f'results_{args.task}.csv')
    save_results_to_csv(all_results, csv_path)

    if args.hyperparam_search:
        hp_csv_path = os.path.join(args.save_root, f'hyperparam_search_{args.task}.csv')
        save_results_to_csv(hyperparam_results_all, hp_csv_path)

    print(f'{"="*70}\n')
    print('FINAL SUMMARY')
    print(f'Task: {args.task}')
    print(f'Model: {args.model_type}')
    print(f'Feature type: {args.feature_type}')
    print(f'Best hyperparameters: lr={best_lr}, hidden_dim={best_hidden_dim}')
    print(f'CV Mean AUC: {np.mean(fold_val_aucs):.4f} +/- {np.std(fold_val_aucs):.4f}')
    if holdout_results:
        print(f'Holdout Ensemble AUC: {holdout_results["ensemble_auc"]:.4f}')
    print(f'Results saved to: {args.save_root}')


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
    parser.add_argument(
        '--batch_size', default = 1, type = int
    )
    parser.add_argument(
        '--hyperparam_search', action='store_true',
        help='Enable hyperparameter search for lr and hidden dimensions'
    )
    parser.add_argument(
        '--lr_search', type=str, default=None,
        help='Comma-separated list of learning rates to search (e.g., "1e-5,5e-5,1e-4")'
    )
    parser.add_argument(
        '--hidden_dim_search', type=str, default=None,
        help='Comma-separated list of hidden dimensions to search (e.g., "256,384,512")'
    )
    args = parser.parse_args()

    main(args)