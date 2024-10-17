from torch.utils.data import DataLoader, Dataset
import numpy as np
from glob import glob
from tqdm import tqdm
import pickle

import torch
import torch.nn as nn
import torch.nn.functional as F

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from pycox.models import CoxPH
from pycox.evaluation import EvalSurv

import pandas as pd
import pickle as pkl

import os
import glob
from tqdm import tqdm

from torchsurv.loss import cox
from torchsurv.metrics.cindex import ConcordanceIndex

class BCR_Dataset():
    def __init__(self, embeds_root, df, from_bag = False):
        self.embeds_root = embeds_root
        self.df = df
        self.from_bag = from_bag
        
    def __len__(self):
        return len(self.df)
        
    def __getitem__(self, idx):
        curr_row = self.df.iloc[idx]
        slide_name = curr_row['slide_name']
        embed_root = os.path.join(self.embeds_root, slide_name)
        event = curr_row['bcr']
        time = curr_row['days_to_event']
        
        if self.from_bag:
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
        return bag, slide_name, event, time
    
def presave_bags():

    with open('/raid/mpleasure/data_deepstorage/st_projects/bcr_downstream/splits_10_03.pkl', 'rb') as f:
        splits = pkl.load(f)

    embeds_root = '/raid/mpleasure/data_deepstorage/st_projects/bcr_downstream/uni_embeds'
    labels = pd.read_csv('/raid/mpleasure/data_deepstorage/st_projects/bcr_downstream/bcr_survival_data_labels.csv', index_col = 0)
    df_train = labels[labels['slide_name'].isin(splits['train'])]

    bag_save_root = '/raid/mpleasure/data_deepstorage/st_projects/bcr_downstream/presaved_bags'

    train_dataset = BCR_Dataset(
        bag_save_root, labels)
    
    dataloader = torch.utils.data.DataLoader(
        train_dataset, batch_size = 1, shuffle = False, num_workers = 2)

    for batch in tqdm(dataloader):
        bag, name, _, _ = batch
        
        np.save(os.path.join(bag_save_root, f'{name}.npy'), bag)
        

    



if __name__ == "__main__":
    presave_bags()