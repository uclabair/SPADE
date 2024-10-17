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
from dataset import SurvivalDataset
import argparse
from train_survival import train_cycle

def set_up_path_df(folder):
    all_files = sorted(glob.glob(os.path.join(folder, '*.npy')))
    file_list = []
    for file in all_files:
        name = file.split('/')[-1].split('.')[0]
        file_list.append([file, name])

    file_df = pd.DataFrame(file_list, columns = ['file', 'plco_id'])
    return file_df

def main(args):

    ## set up save dirs
    save_folder = os.path.join(args.save_root, args.exp_name)
    os.makedirs(save_folder, exist_ok=True)

    args.results_dir = os.path.join(save_folder, 'results')
    args.writer_dir = os.path.join(save_folder, 'writer')

    os.makedirs(args.results_dir, exist_ok=True)
    os.makedirs(args.writer_dir, exist_ok=True)

    ## set up dataloaders
    if args.task == 'bcr':
        with open(args.splits, 'rb') as f:
            splits = pkl.load(f)
        labels = pd.read_csv(args.labels, index_col = 0)

        df_train = labels[labels[args.pid_col].isin(splits['train'])]
        df_val = labels[labels[args.pid_col].isin(splits['val'])]

        train_dataset = SurvivalDataset(
            args.bag_root, df_train, survival_time_col=args.survival_time_col, censorship_col=args.censorship_col, pid_col = args.pid_col, task = args.task
        )
        val_dataset = SurvivalDataset(
            args.bag_root, df_val, survival_time_col=args.survival_time_col, censorship_col=args.censorship_col, pid_col = args.pid_col, task = args.task
        )
        
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size = args.batch_size, num_workers = args.num_workers, shuffle = True)
        val_loader = torch.utils.data.DataLoader(val_dataset, batch_size = args.batch_size, num_workers = args.num_workers, shuffle = False)
    elif args.task == 'plco_lung':
        df_train = set_up_path_df(args.train_bag_root)
        df_val = set_up_path_df(args.val_bag_root)
        labels = pd.read_csv(args.labels, index_col = 0)

        df_train = df_train.merge(labels, on = 'plco_id', how = 'left')
        df_val = df_val.merge(labels, on = 'plco_id', how = 'left')

        train_dataset = SurvivalDataset(
            args.train_bag_root, df_train, survival_time_col=args.survival_time_col, censorship_col=args.censorship_col, pid_col = args.pid_col, task = args.task
        )
        val_dataset = SurvivalDataset(
            args.val_bag_root, df_val, survival_time_col=args.survival_time_col, censorship_col=args.censorship_col, pid_col = args.pid_col, task = args.task
        )

        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size = args.batch_size, num_workers = args.num_workers, shuffle = True)
        val_loader = torch.utils.data.DataLoader(val_dataset, batch_size = args.batch_size, num_workers = args.num_workers, shuffle = False)


    ## train model
    train_cycle(train_loader, val_loader, args)

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
        '--bag_root', type = str, default = '/raid/mpleasure/data_deepstorage/st_projects/bcr_downstream/presaved_bags_'
    )
    parser.add_argument(
        '--task', type = str, default = 'bcr', choices = ['bcr', 'plco_lung', 'plco_breast']
    )
    parser.add_argument(
        '--train_bag_root', type = str, default = '/raid/mpleasure/PLCO/parsed_data/lung/splits/train_uni_features'
    )
    parser.add_argument(
        '--val_bag_root', type = str, default = '/raid/mpleasure/PLCO/parsed_data/lung/splits/val_uni_features'
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
        '--input_dim', type = int, default = 1024
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
    


    args = parser.parse_args()
    main(args)
