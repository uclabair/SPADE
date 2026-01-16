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
from dataset import SurvivalDatasetTangle
import argparse
from train_survival_tangle import train_cycle

def set_up_path_df(folder):
    all_files = sorted(glob.glob(os.path.join(folder, '*.npy')))
    file_list = []
    for file in all_files:
        name = file.split('/')[-1].split('.')[0]
        file_list.append([file, name])

    file_df = pd.DataFrame(file_list, columns = ['file', 'plco_id'])
    return file_df

def main(args):
    fold_count = args.fold_count
    with open(args.splits, 'rb') as f:
        cv_folds = pkl.load(f)

    cross_fold_metrics = {}
    cross_fold_save_folder = os.path.join(
        args.save_root, 
        args.exp_name
        )

    for i in range(fold_count):
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

        train_names = cv_folds[i]['train']['train_ids']
        test_names = cv_folds[i]['test']['test_ids']
    
        df = pd.read_csv(args.labels, index_col = 0)

        train_dataset = SurvivalDatasetTangle(
            args.bag_root, 
            df, 
            train_names, 
            survival_time_col=args.survival_time_col, 
            censorship_col=args.censorship_col, 
            pid_col = args.pid_col, 
            task = args.task,
            h5_file = args.h5_file
        )
        test_dataset = SurvivalDatasetTangle(
            args.bag_root, 
            df, 
            test_names, 
            survival_time_col=args.survival_time_col, 
            censorship_col=args.censorship_col, 
            pid_col = args.pid_col, 
            task = args.task,
            h5_file = args.h5_file
        )
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size = args.batch_size, num_workers = args.num_workers, shuffle = True)
        test_loader = torch.utils.data.DataLoader(test_dataset, batch_size = args.batch_size, num_workers = args.num_workers, shuffle = False)

        fold_metrics = train_cycle(train_loader, test_loader, args, fold = i)
        with open(os.path.join(args.results_dir, f'fold_{i}_{args.task}.pkl'), 'wb') as f:
            pkl.dump(fold_metrics, f)

        cross_fold_metrics[i] = fold_metrics

    with open(os.path.join(args.save_root, f'cross_val_results_{args.task}.pkl'), 'wb') as f:
        pkl.dump(cross_fold_metrics, f)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    ## data, save paths, and other folders
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
        '--bag_root', type = str, default = ''
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


    ## model and training params
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
    
    ## dataset info
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

    args = parser.parse_args()
    main(args)
