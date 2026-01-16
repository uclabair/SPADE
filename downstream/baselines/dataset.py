from torch.utils.data import DataLoader, Dataset
import numpy as np
import torch
from glob import glob
from tqdm import tqdm
import pickle
import os
import h5py

def load_h5_features(h5_path: str):
    with h5py.File(h5_path, 'r') as f:
        return np.array(f[features])


class PatchDataset(Dataset):
    def __init__(self, df, feat_path = None, id_col = 'Case ID', label_col = 'label', h5_file = False):
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
        self.h5_file = h5_file

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):

        try:
            id_ = self.df.iloc[index][self.id_col]
            label = self.df.iloc[index][self.label_col]
            if self.h5_file:
                feature_file = os.path.join(self.feat_path, f'{id_}.h5')
                features = load_h5_features(feature_file)
            else:
                feature_bag = os.path.join(self.feat_path, f'{id_}.npy')
                features = np.array(np.load(feature_bag, allow_pickle = True))
        
        except:
            id_ = self.df.iloc[0][self.id_col]
            label = self.df.iloc[0][self.label_col]
            if self.h5_file:
                feature_file = os.path.join(self.feat_path, f'{id_}.h5')
                features = load_h5_features(feature_file)
            else:
                feature_bag = os.path.join(self.feat_path, f'{id_}.npy')
                features = np.array(np.load(feature_bag, allow_pickle = True))
        
        return {'feature': torch.tensor(features.astype(np.float32)), 'label': torch.tensor(label)}

class PatchDatasetCV(Dataset):
    def __init__(self, names, labels, feat_path = None, h5_file = None):
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

        self.names = names
        self.labels = labels
        self.feat_path = feat_path
        self.h5_file = h5_file

    def __len__(self):
        return len(self.names)

    def __getitem__(self, index):

        try:
            id_ = self.names[index]
            label = self.labels[index]
            if self.h5_file:
                feature_file = os.path.join(self.feat_path, f'{id_}.h5')
                features = load_h5_features(feature_file)
            else:
                feature_bag = os.path.join(self.feat_path, f'{id_}.npy')
                features = np.array(np.load(feature_bag, allow_pickle = True))

            if len(features) == 0:
                id_ = self.names[0]
                label = self.labels[0]
                if self.h5_file:
                    feature_file = os.path.join(self.feat_path, f'{id_}.h5')
                    features = load_h5_features(feature_file)
                else:
                    feature_bag = os.path.join(self.feat_path, f'{id_}.npy')
                    features = np.array(np.load(feature_bag, allow_pickle = True))
            

        
        except:
            id_ = self.names[0]
            label = self.labels[0]
            if self.h5_file:
                feature_file = os.path.join(self.feat_path, f'{id_}.h5')
                features = load_h5_features(feature_file)
            else:
                feature_bag = os.path.join(self.feat_path, f'{id_}.npy')
                features = np.array(np.load(feature_bag, allow_pickle = True))
        
        return {'feature': torch.tensor(features.astype(np.float32)), 'label': torch.tensor(label)}

