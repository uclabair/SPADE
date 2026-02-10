import os
import os
import glob
import numpy as np
import pandas as pd
import pickle as pkl
import matplotlib.pyplot as plts
import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.utils.tensorboard import SummaryWriter

from utils import *
from losses import *
from models import *
from dataset import SurvivalDataset
import argparse
from train_survival import train_cycle
from tqdm import tqdm
from sksurv.metrics import concordance_index_censored


DEFAULT_LR_SEARCH = [1e-5, 5e-5, 1e-4, 5e-4]
DEFAULT_HIDDEN_DIM_SEARCH = [256, 384, 512]

def set_up_path_df(folder):
    all_files = sorted(glob.glob(os.path.join(folder, '*.npy')))
    file_list = []
    for file in all_files:
        name = file.split('/')[-1].split('.')[0]
        file_list.append([file, name])

    file_df = pd.DataFrame(file_list, columns = ['file', 'plco_id'])
    return file_df

def run_hyperparam_search_survival(train_loader, val_loader, args, save_folder, fold, lr_search, hidden_dim_search):

    best_val_cindex = 0
    best_lr = args.lr
    best_hidden_dim = None
    hyperparam_results = []

    if args.loss_fn == 'nll':
        loss_fn = NLLSurvLoss(alpha=args.nll_alpha)
    elif args.loss_fn == 'cox':
        loss_fn = CoxLoss()

    for lr in tqdm(lr_search):
        for hidden_dim in hidden_dim_search:
            print(f'  Trying lr={lr}, hidden_dim={hidden_dim}')

            if args.model_type == 'abmil':
                model = MIL_Attention_fc(input_size=args.input_dim, n_classes=args.n_label_bins, hidden_dim=hidden_dim)
            elif args.model_type == 'transmil':
                model = TransMIL(size_arg=args.feature_type, n_classes=args.n_label_bins, hidden_dim=hidden_dim)
            model = model.cuda()

            search_epochs = min(args.max_epochs, 10)
            optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=0.01)

            best_epoch_cindex = 0

            for epoch in range(search_epochs):
                model.train()
                for batch in tqdm(train_loader):
                    data = batch['img'].cuda()
                    if len(data.shape) > 3:
                        data = data.squeeze(0)
                    label = batch['label'].cuda()
                    censorship = batch['censorship'].cuda()

                    optimizer.zero_grad()
                    if args.model_type == 'abmil':
                        logits, _, _, _, _ = model(data.float())
                    else:
                        logits = model(data=data.float())

                    if isinstance(loss_fn, NLLSurvLoss):
                        surv_loss_dict = loss_fn(logits=logits, times=label, censorships=censorship)
                    elif isinstance(loss_fn, CoxLoss):
                        surv_loss_dict = loss_fn(logits=logits, times=label, censorships=censorship)

                    loss = surv_loss_dict['loss']
                    if loss is not None:
                        loss.backward()
                        optimizer.step()

                model.eval()
                all_risk_scores, all_censorships, all_event_times = [], [], []
                with torch.no_grad():
                    for batch in val_loader:
                        data = batch['img'].cuda()
                        if len(data.shape) > 3:
                            data = data.squeeze(0)
                        label = batch['label'].cuda()
                        event_time = batch['survival_time'].cuda()
                        censorship = batch['censorship'].cuda()

                        if args.model_type == 'abmil':
                            logits, _, _, _, _ = model(data.float())
                        else:
                            logits = model(data=data.float())

                        if isinstance(loss_fn, NLLSurvLoss):
                            hazards = torch.sigmoid(logits)
                            survival = torch.cumprod(1 - hazards, dim=1)
                            risk = -torch.sum(survival, dim=1).unsqueeze(dim=1)
                        else:
                            risk = torch.exp(logits)

                        all_risk_scores.append(risk.detach().cpu().numpy())
                        all_censorships.append(censorship.cpu().numpy())
                        all_event_times.append(event_time.cpu().numpy())

                all_risk_scores = np.concatenate(all_risk_scores).squeeze()
                all_censorships = np.concatenate(all_censorships).squeeze()
                all_event_times = np.concatenate(all_event_times).squeeze()

                if len(all_risk_scores.shape) > 1:
                    all_risk_scores = all_risk_scores[:, 0]

                c_index = concordance_index_censored(
                    (1 - all_censorships).astype(bool), all_event_times, all_risk_scores, tied_tol=1e-08)[0]

                if c_index > best_epoch_cindex:
                    best_epoch_cindex = c_index

            print(f'Best val C-index: {best_epoch_cindex:.4f}')
            hyperparam_results.append({
                'lr': lr,
                'hidden_dim': hidden_dim,
                'val_cindex': best_epoch_cindex
            })

            if best_epoch_cindex > best_val_cindex:
                best_val_cindex = best_epoch_cindex
                best_lr = lr
                best_hidden_dim = hidden_dim

            del model, optimizer
            torch.cuda.empty_cache()

    return best_lr, best_hidden_dim, best_val_cindex, hyperparam_results

def evaluate_survival_model(model, loader, loss_fn, args):
    model.eval()
    all_risk_scores = []
    all_censorships = []
    all_event_times = []

    with torch.no_grad():
        for batch in loader:
            data = batch['img'].cuda()
            if len(data.shape) > 3:
                data = data.squeeze(0)
            event_time = batch['survival_time'].cuda()
            censorship = batch['censorship'].cuda()

            if args.model_type == 'abmil':
                logits, _, _, _, _ = model(data.float())
            else:
                logits = model(data=data.float())

            if isinstance(loss_fn, NLLSurvLoss):
                hazards = torch.sigmoid(logits)
                survival = torch.cumprod(1 - hazards, dim=1)
                risk = -torch.sum(survival, dim=1).unsqueeze(dim=1)
            else:
                risk = torch.exp(logits)

            all_risk_scores.append(risk.detach().cpu().numpy())
            all_censorships.append(censorship.cpu().numpy())
            all_event_times.append(event_time.cpu().numpy())

    all_risk_scores = np.concatenate(all_risk_scores).squeeze()
    all_censorships = np.concatenate(all_censorships).squeeze()
    all_event_times = np.concatenate(all_event_times).squeeze()

    if len(all_risk_scores.shape) > 1:
        all_risk_scores = all_risk_scores[:, 0]

    return all_risk_scores, all_censorships, all_event_times

def ensemble_survival_predictions(all_fold_risks):
    """Average risk predictions from multiple folds."""
    stacked = np.stack(all_fold_risks, axis=0)
    return np.mean(stacked, axis=0)


def save_results_to_csv(results_list, save_path):
    df = pd.DataFrame(results_list)
    df.to_csv(save_path, index=False)
    print(f'Results saved to: {save_path}')
    return df


def main(args):
    import datetime
    timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')

    fold_count = args.fold_count
    with open(args.splits, 'rb') as f:
        cv_folds = pkl.load(f)

    if 'fold_0' in cv_folds:
        fold_keys = [f'fold_{i}' for i in range(args.fold_count)]
        has_holdout = 'holdout_test' in cv_folds
    else:
        fold_keys = list(range(args.fold_count))
        has_holdout = False

    print(fold_keys)

    all_results = []
    hyperparam_results_all = []
    cross_fold_metrics = {}
    cross_fold_save_folder = os.path.join(
        args.save_root, 
        args.exp_name
        )

    df = pd.read_csv(args.labels, index_col=0)

    if args.loss_fn == 'nll':
        loss_fn = NLLSurvLoss(alpha=args.nll_alpha)
    elif args.loss_fn == 'cox':
        loss_fn = CoxLoss()

    if args.hyperparam_search:
        lr_search = [float(x) for x in args.lr_search.split(',')] if args.lr_search else DEFAULT_LR_SEARCH
        hidden_dim_search = [int(x) for x in args.hidden_dim_search.split(',')] if args.hidden_dim_search else DEFAULT_HIDDEN_DIM_SEARCH
        print(f'Hyperparameter search enabled:')
        print(f'Learning rates: {lr_search}')
        print(f'Hidden dimensions: {hidden_dim_search}')
    else:
        lr_search = [args.lr]
        hidden_dim_search = [None]

    print('PHASE 1: Hyperparameter Search')
    hyperparam_scores = {}

    for i, fold_key in enumerate(fold_keys):
        print(f'Starting fold: {i}')
        save_folder = os.path.join(
            args.save_root, 
            args.exp_name, 
            f'fold_{i}'
            )
        os.makedirs(save_folder, exist_ok=True)

        args.results_dir = os.path.join(save_folder, 'results')
        args.writer_dir = os.path.join(save_folder, 'writer')

        os.makedirs(args.results_dir, exist_ok=True)
        os.makedirs(args.writer_dir, exist_ok=True)

        fold_data = cv_folds[fold_key]
        train_names = fold_data['train']['train_ids']
        val_names = fold_data['val']['val_ids']
    

        train_dataset = SurvivalDataset(
            args.bag_root, 
            df, 
            train_names, 
            survival_time_col=args.survival_time_col, 
            censorship_col=args.censorship_col, 
            pid_col = args.pid_col, 
            task = args.task,
            h5_file = args.h5_file
        )
        val_dataset = SurvivalDataset(
            args.bag_root, 
            df, 
            val_names, 
            survival_time_col=args.survival_time_col, 
            censorship_col=args.censorship_col, 
            pid_col = args.pid_col, 
            task = args.task,
            h5_file = args.h5_file
        )
        
        train_loader = torch.utils.data.DataLoader(
            train_dataset, 
            batch_size = args.batch_size, 
            num_workers = args.num_workers, 
            shuffle = True)
        val_loader = torch.utils.data.DataLoader(
            val_dataset, 
            batch_size = args.batch_size, 
            num_workers = args.num_workers, 
            shuffle = False)

        if args.hyperparam_search:
            _, _, _, hp_results = run_hyperparam_search_survival(
                train_loader, val_loader, args, args.save_root, i, lr_search, hidden_dim_search
            )

            for hp_res in hp_results:
                key = (hp_res['lr'], hp_res['hidden_dim'])
                if key not in hyperparam_scores:
                    hyperparam_scores[key] = []
                hyperparam_scores[key].append(hp_res['val_cindex'])

                hyperparam_results_all.append({
                    'task': args.task,
                    'model_type': args.model_type,
                    'feature_type': args.feature_type,
                    'phase': 'hyperparam_search',
                    'fold': i,
                    'lr': hp_res['lr'],
                    'hidden_dim': hp_res['hidden_dim'],
                    'val_cindex': hp_res['val_cindex'],
                    'timestamp': timestamp
                })

        #print(f'Training with lr={best_lr}, hidden_dim={best_hidden_dim}')
        #fold_metrics = train_cycle(train_loader, test_loader, args, fold=i, lr=best_lr, hidden_dim=best_hidden_dim)
        #fold_metrics['best_lr'] = best_lr
        #fold_metrics['best_hidden_dim'] = best_hidden_dim

        #with open(os.path.join(args.results_dir, f'fold_{i}_{args.task}.pkl'), 'wb') as f:
        #    pkl.dump(fold_metrics, f)

        #cross_fold_metrics[i] = fold_metrics
    if args.hyperparam_search and hyperparam_scores:
        best_hp = max(hyperparam_scores.keys(), key=lambda k: np.mean(hyperparam_scores[k]))
        best_lr, best_hidden_dim = best_hp
        print(f'\nBest hyperparameters: lr={best_lr}, hidden_dim={best_hidden_dim}')
        print(f'Mean CV C-index: {np.mean(hyperparam_scores[best_hp]):.4f}')
    else:
        best_lr = args.lr
        best_hidden_dim = None
    
    print('PHASE 2: Training Final Models')
    fold_models = []
    fold_val_cindices = []

    for fold_idx, fold_key in enumerate(fold_keys):
        print(f'\n--- Training Fold {fold_idx} ---')

        save_folder = os.path.join(args.save_root, args.exp_name, f'fold_{fold_idx}')
        os.makedirs(save_folder, exist_ok=True)

        args.results_dir = os.path.join(save_folder, 'results')
        args.writer_dir = os.path.join(save_folder, 'writer')
        os.makedirs(args.results_dir, exist_ok=True)
        os.makedirs(args.writer_dir, exist_ok=True)

        fold_data = cv_folds[fold_key]
        train_names = fold_data['train']['train_ids']
        val_names = fold_data['val']['val_ids']

        train_dataset = SurvivalDataset(
            args.bag_root, df, train_names,
            survival_time_col=args.survival_time_col,
            censorship_col=args.censorship_col,
            pid_col=args.pid_col, task=args.task, h5_file=args.h5_file
        )
        val_dataset = SurvivalDataset(
            args.bag_root, df, val_names,
            survival_time_col=args.survival_time_col,
            censorship_col=args.censorship_col,
            pid_col=args.pid_col, task=args.task, h5_file=args.h5_file
        )

        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size,
                                                    num_workers=args.num_workers, shuffle=True)
        val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=args.batch_size,
                                                  num_workers=args.num_workers, shuffle=False)

        fold_metrics = train_cycle(train_loader, val_loader, args, fold=fold_idx, lr=best_lr, hidden_dim=best_hidden_dim)

        best_epoch = max(fold_metrics.keys(), key=lambda k: fold_metrics[k]['val']['c_index'] if isinstance(k, int) and 'val' in fold_metrics[k] else 0)
        if isinstance(best_epoch, int):
            val_cindex = fold_metrics[best_epoch]['val']['c_index']
        else:
            val_cindex = 0.0

        fold_val_cindices.append(val_cindex)

        if args.model_type == 'abmil':
            model = MIL_Attention_fc(input_size=args.input_dim, n_classes=args.n_label_bins, hidden_dim=best_hidden_dim)
        elif args.model_type == 'transmil':
            model = TransMIL(size_arg=args.feature_type, n_classes=args.n_label_bins, hidden_dim=best_hidden_dim)
        model = model.cuda()

        checkpoints = glob.glob(os.path.join(save_folder, '*.pt'))
        if checkpoints:
            latest_ckpt = max(checkpoints, key=os.path.getctime)
            state_dict = torch.load(latest_ckpt)
            model.load_state_dict(state_dict['model_state_dict'])

        fold_models.append(model)

        cross_fold_metrics[fold_idx] = {
            'val_cindex': val_cindex,
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
            'val_cindex': val_cindex,
            'timestamp': timestamp
        })

    print(f'\nCV Validation C-indices: {fold_val_cindices}')
    print(f'Mean CV C-index: {np.mean(fold_val_cindices):.4f} +/- {np.std(fold_val_cindices):.4f}')

    holdout_results = None
    print('PHASE 3: Holdout Test Evaluation (Ensemble)')
    holdout_data = cv_folds['holdout_test']
    test_names = holdout_data['test_ids']

    test_dataset = SurvivalDataset(
        args.bag_root, df, test_names,
        survival_time_col=args.survival_time_col,
        censorship_col=args.censorship_col,
        pid_col=args.pid_col, task=args.task, h5_file=args.h5_file
    )
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=args.batch_size,
                                                num_workers=args.num_workers, shuffle=False)

    all_fold_risks = []
    individual_cindices = []

    for fold_idx, model in enumerate(fold_models):
        risks, censorships, event_times = evaluate_survival_model(model, test_loader, loss_fn, args)
        all_fold_risks.append(risks)

        c_index = concordance_index_censored(
            (1 - censorships).astype(bool), event_times, risks, tied_tol=1e-08)[0]
        individual_cindices.append(c_index)
        print(f'Fold {fold_idx} model - Test C-index: {c_index:.4f}')

        all_results.append({
            'task': args.task,
            'model_type': args.model_type,
            'feature_type': args.feature_type,
            'phase': 'holdout_individual',
            'fold': fold_idx,
            'lr': best_lr,
            'hidden_dim': best_hidden_dim,
            'test_cindex': c_index,
            'timestamp': timestamp
        })

    ensemble_risks = ensemble_survival_predictions(all_fold_risks)
    ensemble_cindex = concordance_index_censored(
        (1 - censorships).astype(bool), event_times, ensemble_risks, tied_tol=1e-08)[0]

    print(f'\nEnsemble Test C-index: {ensemble_cindex:.4f}')

    holdout_results = {
        'ensemble_cindex': ensemble_cindex,
        'individual_cindices': individual_cindices
    }

    all_results.append({
        'task': args.task,
        'model_type': args.model_type,
        'feature_type': args.feature_type,
        'phase': 'holdout_ensemble',
        'fold': 'ensemble',
        'lr': best_lr,
        'hidden_dim': best_hidden_dim,
        'test_cindex': ensemble_cindex,
        'timestamp': timestamp
    })

    #cross_fold_metrics['hyperparam_summary'] = hyperparam_summary
    os.makedirs(args.save_root, exist_ok=True)

    results_dict = {
        'cross_fold_metrics': cross_fold_metrics,
        'best_lr': best_lr,
        'best_hidden_dim': best_hidden_dim,
        'cv_mean_cindex': np.mean(fold_val_cindices),
        'cv_std_cindex': np.std(fold_val_cindices),
        'holdout_results': holdout_results,
        'hyperparam_search_results': hyperparam_results_all,
        'args': vars(args)
    }

    with open(os.path.join(args.save_root, f'results_{args.task}_{timestamp}.pkl'), 'wb') as f:
        pkl.dump(results_dict, f)

    csv_path = os.path.join(args.save_root, f'results_{args.task}_{timestamp}.csv')
    save_results_to_csv(all_results, csv_path)

    if args.hyperparam_search:
        hp_csv_path = os.path.join(args.save_root, f'hyperparam_search_{args.task}_{timestamp}.csv')
        save_results_to_csv(hyperparam_results_all, hp_csv_path)

    print('FINAL SUMMARY')
    print(f'Task: {args.task}')
    print(f'Model: {args.model_type}')
    print(f'Feature type: {args.feature_type}')
    print(f'Best hyperparameters: lr={best_lr}, hidden_dim={best_hidden_dim}')
    print(f'CV Mean C-index: {np.mean(fold_val_cindices):.4f} +/- {np.std(fold_val_cindices):.4f}')
    print(f'Holdout Ensemble C-index: {holdout_results["ensemble_cindex"]:.4f}')
    print(f'Results saved to: {args.save_root}')

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        '--save_root', type = str, default = '/raid/mpleasure/data_deepstorage/st_projects/bcr_downstream/survival_models'
    )
    parser.add_argument(
        '--exp_name', type = str, default = 'bcr_nll_survival'
    )
    parser.add_argument(
        '--splits', type = str, default = '/raid/mpleasure/data_deepstorage/st_projects/bcr_downstream/splits_10_03.pkl'
    )
    parser.add_argument(
        '--labels', type = str, default = '/raid/mpleasure/data_deepstorage/st_projects/bcr_downstream/bcr_survival_data_labels.csv'
    )
    parser.add_argument(
        '--bag_root', type = str, default = '/raid/mpleasure/data_deepstorage/st_projects/bcr_downstream/presaved_bags_'
    )
    parser.add_argument(
        '--task', type = str, default = 'bcr', choices = ['bcr', 'plco_lung', 'plco_breast', 'tcga_prad', 'tcga_ucec', 'tcga_brca', 'plco_lung_conch']
    )
    parser.add_argument(
        '--fold_count', type = int, default = 5
    )
    parser.add_argument(
        '--h5_file', type = bool, default = False
    )
    parser.add_argument('--loss_fn', type=str, default='nll', choices=['nll', 'cox', 'sumo', 'ipcwls', 'rank'],
                    help='which loss function to use')
    parser.add_argument('--nll_alpha', type=float, default=0,
                    help='Balance between censored / uncensored loss')
    parser.add_argument(
        '--accum_steps', type = int, default = 1
    )
    parser.add_argument(
        '--lr', type = float, default = 1e-4
    )
    parser.add_argument(
        '--weight_decay', type = float, default = 1e-5
    )
    parser.add_argument(
        '--max_epochs', type = int, default = 20
    )
    parser.add_argument('--in_dropout', default=0.1, type=float,
                    help='Probability of dropping out input features.')
    parser.add_argument(
        '--input_dim', type = int, default = 512
    )
    parser.add_argument(
        '--alpha', type = float, default = 0.5
    )
    parser.add_argument('--warmup_steps', type=int,
                    default=-1, help='warmup iterations')
    parser.add_argument('--warmup_epochs', type=int,
                    default=-1, help='warmup epochs')
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--print_every', default=100,
                    type=int, help='how often to print')
    parser.add_argument('--seed', type=int, default=1,
                        help='random seed for reproducible experiment (default: 1)')
    parser.add_argument('--num_workers', type=int, default=2)
    
    parser.add_argument(
        '--survival_time_col', type = str, default = 'days_to_event'
    )
    parser.add_argument(
        '--censorship_col', type = str, default = 'bcr'
    )
    parser.add_argument(
        '--pid_col', type = str, default = 'slide_name'
    )
    parser.add_argument('--n_label_bins', type=int, default=4,
                    help='number of bins for event time discretization')
    parser.add_argument(
        '--label_bins', default = None
    )
    parser.add_argument(
        '--recompute_loss_at_end', type = bool, default = True
    )
    parser.add_argument(
        '--model_type', default = 'abmil'
    )
    parser.add_argument(
        '--feature_type', default = 'bleep'
    )
    parser.add_argument(
        '--hyperparam_search', action='store_true',
    )
    parser.add_argument(
        '--lr_search', type=str, default=None,
    )
    parser.add_argument(
        '--hidden_dim_search', type=str, default=None,
    )

    args = parser.parse_args()
    main(args)
