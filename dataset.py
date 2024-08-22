import os
import cv2
import pandas as pd
import torch
from sklearn.decomposition import TruncatedSVD
# from scipy.sparse import csr_matrix
import numpy as np
import torchvision.transforms.functional as TF
import random
from PIL import Image

class CLIPDataset(torch.utils.data.Dataset):
    def __init__(self, df, CFG):
        self.df = df
        self.CFG = CFG

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        id_ = row['id']
        patch_x = row['patch_x']
        patch_y = row['patch_y']

        index = row['bead']
        
        reduced_matrix = np.load(os.path.join(self.CFG.hvg_matrix_path, f"hvg_matrix_{id_}.npy")).T
        image_features = np.load(os.path.join(self.CFG.image_features_path, f"{id_}/{patch_x}_{patch_y}.npy"))

        item = {
            'image_features': torch.tensor(image_features).float(),
            'reduced_expression': torch.tensor(reduced_matrix[index]).float(),
            'spatial_coords': [patch_x, patch_y]
        }

        return item

    def __len__(self):
        return self.df.shape[0]