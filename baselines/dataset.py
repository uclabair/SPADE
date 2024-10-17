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
        ##self.dataset = dataset

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        # print(index)
        
        path = '/raid/mpleasure/PLCO/parsed_data/lung/splits/neighbor_attn_feats_v1_512_img_only/all_feats'
        #path = '/raid/mpleasure/PLCO/parsed_data/lung/splits/neighbor_attn_feats_v3_cross_modal_512_8_heads/all_feats'
        #path = '/raid/mpleasure/PLCO/parsed_data/lung/splits/neighbor_attn_feats_512_v2_image_only/all_feats'
        #path = '/raid/mpleasure/PLCO/parsed_data/lung/splits/proto_attn_feats_image_proto_only/all_feats'
        #train_feat_folder = '/raid/mpleasure/PLCO/parsed_data/lung/splits/train_uni_features'
        #val_feat_folder = '/raid/mpleasure/PLCO/parsed_data/lung/splits/val_uni_features'
        #test_feat_folder = '/raid/mpleasure/PLCO/parsed_data/lung/splits/test_uni_features'
        try:

            id_ = self.df.iloc[index]['plco_id']
            label = self.df.iloc[index]['label']
            if self.mode =='train':
                #features = np.load(f'{train_feat_folder}/{id_}.npy', allow_pickle=True)
                features = np.load(f'{path}/train/{id_}.npy', allow_pickle=True)
            elif self.mode =='val':
                #features = np.load(f'{val_feat_folder}/{id_}.npy', allow_pickle=True)
                features = np.load(f'{path}/val/{id_}.npy', allow_pickle=True)
            elif self.mode =='test':
                #features = np.load(f'{test_feat_folder}/{id_}.npy', allow_pickle=True)
                features = np.load(f'{path}/test/{id_}.npy', allow_pickle=True)
            features = np.array(features)
        except:
            id_ = self.df.iloc[0]['plco_id']
            label = self.df.iloc[0]['label']
            if self.mode =='train':
                features = np.load(f'{path}/train/{id_}.npy', allow_pickle=True)
                #features = np.load(f'{train_feat_folder}/{id_}.npy', allow_pickle=True)
            elif self.mode =='val':
                features = np.load(f'{path}/val/{id_}.npy', allow_pickle=True)
                #features = np.load(f'{val_feat_folder}/{id_}.npy', allow_pickle=True)
            elif self.mode =='test':
                features = np.load(f'{path}/test/{id_}.npy', allow_pickle=True)
                #features = np.load(f'{test_feat_folder}/{id_}.npy', allow_pickle=True)
            features = np.array(features)
    
        return {'feature': torch.tensor(features.astype(np.float32)), 'label': torch.tensor(label)}
    

class PatchDataset_Camelyon(Dataset):
    def __init__(self, df, mode, dataset='bleep', feat_path = None):
        self.df = df
        self.mode = mode
        self.feat_path = feat_path

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        # print(index)
        
        
        try:
            id_ = self.df.iloc[index]['image']
            label = self.df.iloc[index]['binary']
            name = id_.split('.')[0]
            feature_bag = os.path.join(self.feat_path, f'{name}.npy')
            features = np.load(feature_bag, allow_pickle = True)
            features = np.array(features)
        except:
            id_ = self.df.iloc[0]['image']
            label = self.df.iloc[0]['binary']
            name = id_.split('.')[0]
            feature_bag = os.path.join(self.feat_path, f'{name}.npy')
            features = np.load(feature_bag, allow_pickle = True)
            features = np.array(features)
    
        return {'feature': torch.tensor(features.astype(np.float32)), 'label': torch.tensor(label)}


class PatchDataset_CPTAC(Dataset):
    def __init__(self, df, mode, dataset='bleep', feat_path = None):
        self.df = df
        self.mode = mode
        self.feat_path = feat_path

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        # print(index)
        
        
        try:
            id_ = self.df.iloc[index]['Case_ID']
            label = self.df.iloc[index]['label']
            feature_bag = os.path.join(self.feat_path, f'{id_}.npy')
            features = np.load(feature_bag, allow_pickle = True)
            features = np.array(features)
        except:
            id_ = self.df.iloc[0]['Case_ID']
            label = self.df.iloc[0]['label']
            feature_bag = os.path.join(self.feat_path, f'{id_}.npy')
            features = np.load(feature_bag, allow_pickle = True)
            features = np.array(features)
    
        return {'feature': torch.tensor(features.astype(np.float32)), 'label': torch.tensor(label)}
    
class PatchDataset_PANDA(Dataset):
    def __init__(self, df, mode, dataset='panda', feat_path = None):
        self.df = df
        self.mode = mode
        self.feat_path = feat_path

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        # print(index)
        
        
        try:
            id_ = self.df.iloc[index]['image_id']
            label = self.df.iloc[index]['isup_grade']
            feature_bag = os.path.join(self.feat_path, f'{id_}.npy')
            features = np.load(feature_bag, allow_pickle = True)
            features = np.array(features)
        except:
            id_ = self.df.iloc[0]['image_id']
            label = self.df.iloc[0]['isup_grade']
            feature_bag = os.path.join(self.feat_path, f'{id_}.npy')
            features = np.load(feature_bag, allow_pickle = True)
            features = np.array(features)
    
        return {'feature': torch.tensor(features.astype(np.float32)), 'label': torch.tensor(label)}
    