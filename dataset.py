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
import ast

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

class STDataset_PreLoad(torch.utils.data.Dataset):
    def __init__(self, df, CFG):
        self.df = df
        self.CFG = CFG
        self.preload_dataset()

    def preload_dataset(self):
        self.reduced_matrices = {}
        for id_ in self.df['id'].unique():
            self.reduced_matrices[id_] = np.load(os.path.join(self.CFG.hvg_matrix_path, f'hvg_matrix_{id_}.npy'), mmap_mode = 'r').T
        
    def __getitem__(self, idx):
        idx_list = idx
        row = self.df.iloc[idx]
        id_ = row['id']
        patch_x = row['patch_x_obj']
        patch_y = row['patch_y_obj']

        index = row['bead']

        image_features = np.load(os.path.join(self.CFG.image_features_path, f'{id_}/{patch_x}_{patch_y}.npy'))
        reduced_matrix = self.reduced_matrices[id_]

        item = {
            'image_features': torch.tensor(image_features).float(),
            'reduced_expression': torch.tensor(reduced_matrix[index]).float(),
            'spatial_coords': [patch_x, patch_y]
        }

        return item

    def __len__(self):
        return self.df.shape[0]
    

class STDataset_Neighbors(torch.utils.data.Dataset):
    def __init__(self, df, CFG):
        self.df = df
        self.CFG = CFG
        self.max_neighbors = None
        self.preload_dataset()

    def preload_dataset(self):
        self.reduced_matrices = {}
        for id_ in self.df['id'].unique():
            self.reduced_matrices[id_] = np.load(os.path.join(self.CFG.hvg_matrix_path, f'hvg_matrix_{id_}.npy'), mmap_mode = 'r').T
    
    def pad_neighbors(self, neighbors_tensor, type = 'image'):
        if self.max_neighbors is None:
            self.max_neighbors = self.df['bead_neighbors'].apply(lambda x: len(ast.literal_eval(x))).max()

        if type == 'image':
            padded_neighbors = torch.zeros((self.max_neighbors, self.CFG.image_embedding))
        elif type == 'gene':
            padded_neighbors = torch.zeros((self.max_neighbors, self.CFG.spot_embedding))
        
        length = min(neighbors_tensor.size(0), self.max_neighbors)
        padded_neighbors[:length] = neighbors_tensor[:length]

        return padded_neighbors, length


    def __getitem__(self, idx):
        idx_list = idx
        row = self.df.iloc[idx]
        id_ = row['id']
        patch_x = row['patch_x_obj']
        patch_y = row['patch_y_obj']

        index = row['bead']

        image_features = np.load(os.path.join(self.CFG.image_features_path, f'{id_}/{patch_x}_{patch_y}.npy'))
        reduced_matrix = self.reduced_matrices[id_]

        reduced_expression = torch.tensor(reduced_matrix[index]).float()

        ## pull neighbor embeddings and features
        subset = self.df[self.df['id'] == id_]
        neighbors = subset[subset['bead'].isin(ast.literal_eval(row.bead_neighbors))]
        neighbor_spot_embeds = torch.tensor(reduced_matrix[ast.literal_eval(row.bead_neighbors)])
        neighbor_image_embeddings = []
        for ind, n_row in neighbors.iterrows():
            image_features = np.load(os.path.join(self.CFG.image_features_path, f'{id_}/{n_row.patch_x_obj}_{n_row.patch_y_obj}.npy'))
            neighbor_image_embeddings.append(image_features)
        neighbor_image_embeddings = torch.tensor(np.vstack(neighbor_image_embeddings))

        # pad to make all same size
        padded_neighbor_image_embeddings, img_length = self.pad_neighbors(neighbor_image_embeddings, type = 'image')
        padded_neighbor_gene_embeddings, gene_length = self.pad_neighbors(neighbor_spot_embeds, type = 'gene')

        # save masks
        image_mask = torch.zeros(self.max_neighbors)
        image_mask[:img_length] = 1

        gene_mask = torch.zeros(self.max_neighbors)
        gene_mask[:gene_length] = 1

        item = {
            'image_features': torch.tensor(image_features).float(),
            'reduced_expression': reduced_expression,
            'spatial_coords': [patch_x, patch_y],
            'image_neighbors': padded_neighbor_image_embeddings.float(),
            'gene_neighbors': padded_neighbor_gene_embeddings.float(),
            'image_mask': image_mask,
            'gene_mask': gene_mask
        }

        return item

    def __len__(self):
        return self.df.shape[0]