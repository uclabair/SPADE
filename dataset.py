import os
import cv2
import pandas as pd
import torch
import numpy as np
import random
from PIL import Image
import pickle
import ast

class STDataset_PreLoad(torch.utils.data.Dataset):
    def __init__(self, df, args):
        self.df = df
        self.args = args
        self.preload_dataset()
    def preload_dataset(self):
        self.reduced_matrices = {}
        for id_ in self.df['id'].unique():
            self.reduced_matrices[id_] = np.load(os.path.join(self.args.hvg_matrix_path, f"hvg_matrix_{id_}.npy")).T

    
    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        idx_list = idx
        row = self.df.iloc[idx]
        id_ = row['id']
        patch_x = row['patch_x_obj']
        patch_y = row['patch_y_obj']
        index = row['bead']
        image_features = np.load(os.path.join(self.args.image_features_path, f'{id_}/{patch_x}_{patch_y}.npy'))
        reduced_matrix = self.reduced_matrices[id_]

        item = {
            'image_features': torch.tensor(image_features).float(),
            'reduced_expression': torch.tensor(reduced_matrix[index]).float(),
            'spatial_coords': [patch_x, patch_y]
        }

        return item