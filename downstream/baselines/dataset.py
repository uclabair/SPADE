from torch.utils.data import DataLoader, Dataset
import numpy as np
import torch
from glob import glob
from tqdm import tqdm
import pickle
import os

class PatchDataset_PLCO(Dataset):
    def __init__(self, df, mode, dataset='bleep'):
        self.df = df
        self.mode = mode
        self.path = '/raid/mpleasure/PLCO/parsed_data/lung/splits/neighbor_attn_feats_v1_512_img_only/all_feats'

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        try:
            id_ = self.df.iloc[index]['plco_id']
            label = self.df.iloc[index]['label']
            if self.mode =='train':
                features = np.load(f'{self.path}/train/{id_}.npy', allow_pickle=True)
            elif self.mode =='val':
                features = np.load(f'{self.path}/val/{id_}.npy', allow_pickle=True)
            elif self.mode =='test':
                features = np.load(f'{self.path}/test/{id_}.npy', allow_pickle=True)
            features = np.array(features)
        except:
            id_ = self.df.iloc[1]['plco_id']
            label = self.df.iloc[0]['label']
            if self.mode =='train':
                features = np.load(f'{self.path}/train/{id_}.npy', allow_pickle=True)
            elif self.mode =='val':
                features = np.load(f'{self.path}/val/{id_}.npy', allow_pickle=True)
            elif self.mode =='test':
                features = np.load(f'{self.path}/test/{id_}.npy', allow_pickle=True)
            features = np.array(features)
    
        return {'feature': torch.tensor(features.astype(np.float32)), 'label': torch.tensor(label)}


class PatchDataset(Dataset):
    def __init__(self, df, feat_path = None, id_col = 'Case ID', label_col = 'label'):
        '''
        Default Dataset Across Downstream Subtyping and Gene Alteration Datasets

        df: pandas dataframe for train, val or test set
        feat_path: path to feature folder
        id_col: col in dataframe specifying the name of patient or case
        label_col: the col in dataframe with the classifciation label

        ID COL - LABEL
        CRCGene: 'Case ID', 'label'
        BRCAGene: 'Case ID', 'label'
        PLCOBreast: 'plco_id', 'label_mapped'
        TCGA: 'Case ID', 'label'
        Ovarian: 'image_id', 'numeric_label'
        PANDA: 'image_id', 'isup_grade'
        CPTAC: 'Case_ID', 'label'
        Camelyon: 'image', 'binary'
        PLCOLung: 'plco_id', 'label'
        '''

        self.df = df
        self.feat_path = feat_path
        self.id_col = id_col
        self.label_col = label_col

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
       
        try:
            id_ = self.df.iloc[index][self.id_col]
            label = self.df.iloc[index][self.label_col]
            feature_bag = os.path.join(self.feat_path, f'{id_}.npy')
            features = np.array(np.load(feature_bag, allow_pickle = True))
        
        except:
            id_ = self.df.iloc[0][self.id_col]
            label = self.df.iloc[0][self.label_col]
            feature_bag = os.path.join(self.feat_path, f'{id_}.npy')
            features = np.array(np.load(feature_bag, allow_pickle = True))
    
        return {'feature': torch.tensor(features.astype(np.float32)), 'label': torch.tensor(label)}

