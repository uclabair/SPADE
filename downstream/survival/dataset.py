from torch.utils.data import DataLoader, Dataset
import torch
import numpy as np
import pandas as pd
from glob import glob
from tqdm import tqdm
import pickle
import os
import h5py

def load_h5_features(h5_path: str):
    with h5py.File(h5_path, 'r') as f:
        return np.array(f[features])


class SurvivalDataset():
    def __init__(
            self, 
            embeds_root, 
            df, 
            fold_names,
            survival_time_col, 
            censorship_col, 
            pid_col, 
            n_label_bins = 4, 
            label_bins = None, 
            from_bag = True, 
            task = 'bcr',
            h5_file = False):
        self.embeds_root = embeds_root
        self.df = df
        self.from_bag = from_bag
        self.task = task
        self.fold_names = fold_names
        self.h5_file = h5_file
        
        self.survival_time_col = survival_time_col
        self.censorship_col = censorship_col
        self.n_label_bins = n_label_bins
        self.label_bins = label_bins
        
        if self.n_label_bins > 0:
            disc_labels, label_bins = compute_discretization(
                df = self.df, 
                survival_time_col = self.survival_time_col,
                censorship_col = self.censorship_col,
                n_label_bins = self.n_label_bins,
                label_bins = self.label_bins,
                pid_col=pid_col)
            self.df = self.df.join(disc_labels)
            self.label_bins = label_bins
            self.target_col = disc_labels.name
        
        self.pid_col = pid_col
        self.df = self.df.set_index(pid_col, drop=False)
        self.disc_labels = torch.tensor(disc_labels.values)
        self.survival_time_labels = torch.tensor(self.df[self.survival_time_col].values)
        self.censorship_labels = torch.tensor(self.df[censorship_col].values)
        
    def get_pids(self):
        return self.fold_names
    
    def __len__(self):
        return len(self.fold_names)
    
    def get_label_bins(self):
        return self.label_bins
    
    def __getitem__(self, idx):
        slide_name = self.fold_names[idx]
        curr_row = self.df.loc[slide_name]
        embed_root = os.path.join(self.embeds_root, slide_name)
        censorship = curr_row[self.censorship_col]
        time = curr_row[self.survival_time_col]
        target = curr_row[self.target_col]
        
        if self.from_bag:
            if self.h5_file:
                bag = load_h5_features(os.path.join(self.embeds_root, f'{slide_name}.h5'))
            else:
                bag = np.load(os.path.join(self.embeds_root, f'{slide_name}.npy'))
           
        else:
            all_embeds = sorted(glob.glob(os.path.join(embed_root, '*.npy')))
        
            bag = []
            for embed_file in all_embeds:
                embed = np.load(embed_file)
                if embed.shape[0] != 1024:
                    continue
                elif len(embed.shape) > 1:
                    continue
                else:
                    bag.append(embed)

            bag = np.vstack(bag)
            
        out = {
            'img': bag,
            'survival_time': torch.tensor([time]),
            'censorship': torch.tensor([censorship]),
            'label': torch.tensor([target])
        }
        
        return out
    

def compute_discretization(df, survival_time_col, censorship_col, n_label_bins = 4, label_bins = None, pid_col = None):
    df = df[~df[pid_col].duplicated()]
    
    if label_bins is not None:
        assert len(label_bins) == n_label_bins + 1
        q_bins = label_bins
    else:
        uncensored_df = df[df[censorship_col] == 0]
        disc_labels, q_bins = pd.qcut(uncensored_df[survival_time_col], q = n_label_bins, retbins = True, labels = False)
        q_bins[-1] = 1e6 # set rightmost to be infinite
        q_bins[0] = -1e-6 # set leftmost to 0
        
    disc_labels, q_bins = pd.cut(
        df[survival_time_col], bins = q_bins,
        retbins = True, labels = False,
        include_lowest = True
    )
    
    assert isinstance(disc_labels, pd.Series) and (disc_labels.index.name == df.index.name)
    disc_labels.name = 'disc_label'
    return disc_labels, q_bins


class SurvivalDatasetTangle():
    def __init__(
            self, 
            embeds_root, 
            df, 
            fold_names,
            survival_time_col, 
            censorship_col, 
            pid_col, 
            n_label_bins = 4, 
            label_bins = None, 
            from_bag = True, 
            task = 'bcr'):
        self.embeds_root = embeds_root
        self.df = df
        self.from_bag = from_bag
        self.task = task
        self.fold_names = fold_names
        
        self.survival_time_col = survival_time_col
        self.censorship_col = censorship_col
        self.n_label_bins = n_label_bins
        self.label_bins = label_bins
        
        if self.n_label_bins > 0:
            disc_labels, label_bins = compute_discretization(
                df = self.df, 
                survival_time_col = self.survival_time_col,
                censorship_col = self.censorship_col,
                n_label_bins = self.n_label_bins,
                label_bins = self.label_bins,
                pid_col=pid_col)
            self.df = self.df.join(disc_labels)
            self.label_bins = label_bins
            self.target_col = disc_labels.name
        
        self.pid_col = pid_col
        self.df = self.df.set_index(pid_col, drop=False)
        self.disc_labels = torch.tensor(disc_labels.values)
        self.survival_time_labels = torch.tensor(self.df[self.survival_time_col].values)
        self.censorship_labels = torch.tensor(self.df[censorship_col].values)
        
    def get_pids(self):
        return self.df[self.pid_col].values
    
    def __len__(self):
        return len(self.df)
    
    def get_label_bins(self):
        return self.label_bins
    
    def __getitem__(self, idx):
        slide_name = self.fold_names[idx]
        curr_row = self.df.loc[slide_name]
        embed_root = os.path.join(self.embeds_root, slide_name)
        censorship = curr_row[self.censorship_col]
        time = curr_row[self.survival_time_col]
        target = curr_row[self.target_col]
        
        if self.task == 'plco_breast':
            file_path = '/raid/mpleasure/PLCO/parsed_data_breast/breast/plco_id_to_image_name_map_cleaned.pkl'
            with open(file_path, 'rb') as file:
                breast_image_list = pickle.load(file)
            curr_id = curr_row['plco_id']

            embeds = []
            for item in breast_image_list[curr_id]:
                try:
                    embeds.append(list(np.load(os.path.join(self.embeds_root, f'{item}.npy'), allow_pickle=True)))
                except:
                    continue
        else:
            embeds = np.load(os.path.join(self.embeds_root, f'{slide_name}.npy'))
            #print(embeds.shape)
       
        out = {
            'img': embeds,
            'survival_time': torch.tensor([time]),
            'censorship': torch.tensor([censorship]),
            'label': torch.tensor([target])
        }
            
        
        return out