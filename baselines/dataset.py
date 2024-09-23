from torch.utils.data import DataLoader, Dataset
import numpy as np
import torch
from glob import glob
from tqdm import tqdm
import pickle

from torch.utils.data import DataLoader, Dataset
import numpy as np
import torch
from glob import glob
from tqdm import tqdm
import pickle

class PatchDataset_CAM16(Dataset):
    def __init__(self, df, mode, dataset='bleep'):
        self.df = df
        self.mode = mode
        self.dataset = dataset

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        # print(index)
        path = '/raid/Camelyon/Camelyon16/bleep_norm_features'
        # try:
        id_ = self.df.iloc[index]['image']
        label = self.df.iloc[index]['binary']
        features = np.load('{0}/{1}'.format(path, id_.replace('.tif', '.npy')), allow_pickle=True)
        features = np.array(features)
        # print(index, features.shape)
        # if features.shape[1] == 0:
        id_ = self.df.iloc[0]['image']
        label = self.df.iloc[0]['binary']
        features = np.load('{0}/{1}'.format(path, id_.replace('.tif', '.npy')), allow_pickle=True)
        features = np.array(features)
        return {'feature': torch.tensor(features.astype(np.float32)), 'label': torch.tensor(label)}
        # except:
        #     id_ = self.df.iloc[0]['image']
        #     label = self.df.iloc[0]['binary']
        #     features = np.load('{0}/{1}'.format(path, id_.replace('.tif', '.npy')), allow_pickle=True)
        #     features = np.array(features)
        #     # print(0, features.shape)
       
        #     return {'feature': torch.tensor(features.astype(np.float32)), 'label': torch.tensor(label)}

class PatchDataset_PLCO(Dataset):
    def __init__(self, df, mode, dataset='bleep'):
        self.df = df
        self.mode = mode
        self.dataset = dataset

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        # print(index)
        path = '/raid/mpleasure/PLCO/parsed_data/lung/splits/prototypes_sharp/train_{0}_features_norm'.format(self.dataset)
        # try:
        id_ = self.df.iloc[index]['plco_id']
        label = self.df.iloc[index]['label']
        if self.mode =='train':
            features = np.load('{0}/{1}.npy'.format(path, id_), allow_pickle=True)
        elif self.mode =='val':
            features = np.load('{0}/{1}.npy'.format(path.replace('train', 'val'), id_).format(id_), allow_pickle=True)
        elif self.mode =='test':
            features = np.load('{0}/{1}.npy'.format(path.replace('train', 'test'), id_).format(id_), allow_pickle=True)
        features = np.array(features)
    
        return {'feature': torch.tensor(features.astype(np.float32)), 'label': torch.tensor(label)}
        # except:
        #     id_ = self.df.iloc[0]['plco_id']
        #     label = self.df.iloc[0]['label']

        #     if self.mode =='train':
        #         features = np.load('{0}/{1}.npy'.format(path, id_), allow_pickle=True)
        #     elif self.mode =='val':
        #         features = np.load('{0}/{1}.npy'.format(path.replace('train', 'val'), id_).format(id_), allow_pickle=True)
        #     elif self.mode =='test':
        #         features = np.load('{0}/{1}.npy'.format(path.replace('train', 'test'), id_).format(id_), allow_pickle=True)
        #     features = np.array(features)
        #     return {'feature': torch.tensor(features.astype(np.float32)), 'label': torch.tensor(label)}