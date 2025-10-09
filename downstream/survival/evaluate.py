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
from train_survival import run_test

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
    args.results_dir = os.path.join(save_folder, 'test_results')

    os.makedirs(args.results_dir, exist_ok=True)


    ## set up dataloaders
    if args.task in ['tcga_prad','tcga_ucec', 'tcga_brca', 'plco_breast']:
        
        df_test = pd.read_csv(args.test_csv, index_col = 0)

        test_dataset = SurvivalDataset(
            args.bag_root, df_test, survival_time_col=args.survival_time_col, censorship_col=args.censorship_col, pid_col = args.pid_col, task = args.task
        )
        test_loader = torch.utils.data.DataLoader(test_dataset, batch_size = args.batch_size, num_workers = args.num_workers, shuffle = False)


    elif args.task == 'bcr':
        print('Testing bcr!')
        with open(args.splits, 'rb') as f:
            splits = pkl.load(f)
        labels = pd.read_csv(args.labels, index_col = 0)

        df_test = labels[labels[args.pid_col].isin(splits['test'])]

        test_dataset = SurvivalDataset(
            args.bag_root, df_test, survival_time_col=args.survival_time_col, censorship_col=args.censorship_col, pid_col = args.pid_col, task = args.task
        )
        test_loader = torch.utils.data.DataLoader(test_dataset, batch_size = args.batch_size, num_workers = args.num_workers, shuffle = False)
    elif args.task == 'plco_lung':
        df_test = set_up_path_df(args.test_bag_root)
        labels = pd.read_csv(args.labels, index_col = 0)

        df_test = df_test.merge(labels, on = 'plco_id', how = 'left')

        test_dataset = SurvivalDataset(
            args.test_bag_root, df_test, survival_time_col=args.survival_time_col, censorship_col=args.censorship_col, pid_col = args.pid_col, task = args.task
        )
        test_loader = torch.utils.data.DataLoader(test_dataset, batch_size = args.batch_size, num_workers = args.num_workers, shuffle = False)


    ## test model
    test_results, test_dumps = run_test(test_loader, args)
    with open(os.path.join(args.results_dir, f'test_results.pkl'), 'wb') as f:
        pkl.dump(test_results, f)

    with open(os.path.join(args.results_dir, f'test_dump.pkl'), 'wb') as f:
        pkl.dump(test_dumps, f)

    
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
        '--task', type = str, default = 'bcr', choices = ['bcr', 'plco_lung', 'plco_breast', 'tcga_prad', 'tcga_ucec', 'tcga_brca']
    )
    parser.add_argument(
        '--train_bag_root', type = str, default = '/raid/mpleasure/PLCO/parsed_data/lung/splits/train_uni_features'
    )
    parser.add_argument(
        '--val_bag_root', type = str, default = '/raid/mpleasure/PLCO/parsed_data/lung/splits/val_uni_features'
    )
    parser.add_argument(
        '--test_bag_root', type = str, default = '/raid/mpleasure/PLCO/parsed_data/lung/splits/test_uni_features'
    )
    parser.add_argument(
        '--model_checkpoint', type = str, default = ''
    )


    ## model and training params
    parser.add_argument('--loss_fn', type=str, default='nll', choices=['nll', 'cox', 'sumo', 'ipcwls', 'rank'],
                    help='which loss function to use')
    parser.add_argument('--nll_alpha', type=float, default=0,
                    help='Balance between censored / uncensored loss')
    parser.add_argument(
        '--input_dim', type = int, default = 512
    )
    parser.add_argument(
        '--alpha', type = float, default = 0.5
    )
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
    
    parser.add_argument(
        '--test_csv', type = str, default = ''
    )


    args = parser.parse_args()
    main(args)
