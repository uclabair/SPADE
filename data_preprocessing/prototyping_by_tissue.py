import pandas  as pd
import numpy as np
import glob
import os
import pyarrow.parquet as pq
from sklearn.metrics.pairwise import cosine_similarity
from scipy.spatial.distance import pdist, squareform
import torch
import dask.array as da
from dask.diagnostics import ProgressBar
import faiss
import h5py
from tqdm import tqdm
import matplotlib.pyplot as plt
import dask.dataframe as dd
import leidenalg
import gc
import pickle as pkl

import igraph as ig
import h5py
import numpy as np
from collections import defaultdict
import scanpy as sc
import anndata
import argparse
import pickle

import networkx as nx
from multiprocessing import Pool

from collections import Counter
import argparse

class STDataset_PreLoad(torch.utils.data.Dataset):
    def __init__(self, df, CFG):
        self.df = df
        self.CFG = CFG
        self.preload_dataset()

    def preload_dataset(self):
        self.reduced_matrices = {}
        for id_ in self.df['id'].unique():
            if id_ == 'TENX28':
                print('Still here!')
                continue
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
            'x': patch_x,
            'y': patch_y,
            'bead': index,
            'id': id_
        }

        return item

    def __len__(self):
        return self.df.shape[0]
    

def run_clustering_leiden(embeddings, metadata, pca = True, gpu = False, resolutions = [], save_dir = None, use_parallel_graph_construct = False):
    n_samples, n_features = embeddings.shape
    n_components = n_features - 1

    res = faiss.StandardGpuResources()

    if pca == True:

        print('Running PCA...')
        pca_matrix = faiss.PCAMatrix(n_features, n_components)
        pca_matrix.train(embeddings)
        transformed_embeddings = pca_matrix.apply_py(embeddings)

        del embeddings
        gc.collect()

        b = faiss.vector_to_array(pca_matrix.b)
        A = faiss.vector_to_array(pca_matrix.A).reshape(pca_matrix.d_out, pca_matrix.d_in)
        np.save(os.path.join(save_dir, f'pca_b.npy'), b)
        np.save(os.path.join(save_dir, f'pca_a.npy'), A)
        embeddings_for_nn = transformed_embeddings
    else:
        embeddings_for_nn = embeddings
        A = None
        b = None
    print(f'Embeddings shape: {embeddings_for_nn.shape}')
    
    print('Starting nearest neighbors...')
    n_neighbors = int(n_samples/1000)
    index = faiss.IndexFlatL2(embeddings_for_nn.shape[1])
    index.add(embeddings_for_nn)
    distances, indices = index.search(embeddings_for_nn, k = n_neighbors)
    np.save(os.path.join(save_dir, 'nn_indices.npy'), indices)



    print('Starting graph construction...')
    #if use_parallel_graph_construct:
    #    ig_graph = construct_graph_parallel(indices, num_workers = 8, batch_size = 10000)
    #else:
    #    edges = [(i, neighbor) for i, neighbors in tqdm(enumerate(indices)) for neighbor in neighbors if i != neighbor]
    #    ig_graph = ig.Graph(edges)
    num_points = indices.shape[0]
    num_neighbors = indices.shape[1]

    adj_matrix = lil_matrix((num_points, num_points), dtype = 'int')
    print(adj_matrix.shape)
    for i in tqdm(range(num_points)):
        index = indices[i]
        for neighbor_index in index:
            adj_matrix[i, neighbor_index] = 1

    adj_matrix = csr_matrix(adj_matrix, shape = (num_points, num_points))

    ig_graph = ig.Graph.Adjacency(adj_matrix)

    gc.collect()
    
    print('Starting leiden partition...')
    cluster_memberships = {}
    cluster_memberships_df = metadata.copy()
    for resolution in tqdm(resolutions):
        partition = leidenalg.find_partition(ig_graph, 
                                leidenalg.RBConfigurationVertexPartition,
                                resolution_parameter = resolution,
                                seed = 24)
        clusters = partition.membership
        cluster_memberships[resolution] = clusters

        np.save(f'{save_dir}/clusters_resolution_{resolution}.npy', clusters)
        cluster_memberships_df[f'clusters_resolution_{resolution}'] = clusters

    with open(f'{save_dir}/all_cluster_memberships.pkl', 'wb') as f:
        pickle.dump(cluster_memberships, f)

    cluster_memberships_df.to_csv(os.path.join(save_dir, 'cluster_memberships_train_set.csv'), index=False)
    
    del cluster_memberships_df
    del ig_graph
    gc.collect()

    return cluster_memberships, indices, A, b, embeddings_for_nn




def cluster_kmeans(features, n_proto, n_iter = 50, n_init = 5, feature_dim = 1024, n_proto_patches = 50000, use_cuda = False):
    #number_gpus = torch.cuda.device_count()
    kmeans = faiss.Kmeans(
        features.shape[1],
        n_proto,
        niter = n_iter,
        nredo = n_init,
        verbose = False,
        max_points_per_centroid = n_proto_patches
    )

    kmeans.train(features)
    centroids = kmeans.centroids[np.newaxis, ...]
    _, labels = kmeans.index.search(features, 1)

    return centroids, labels


def run_tissue_type(args, organ_type, subset_coords):

    dataset = STDataset_PreLoad(subset_coords, args)
    
    all_image_embedding = []

    for i, batch in enumerate(tqdm(dataset)):
        image_features = batch['image_features'].numpy()
        all_image_embedding.append(image_features)

    if len(all_image_embedding) > 0:
        all_image_embedding = np.vstack(all_image_embedding)
    else:
        print(f'Issue with: {organ_type}')
        return 0
    
    np.save(os.path.join(args.out_dir, f'{organ_type}_image_embeds_raw.npy'), all_image_embedding)

    subset_coords.to_csv(os.path.join(args.out_dir, f'{organ_type}_metadata.csv'))

    centroids_image, labels_image = cluster_kmeans(all_image_embedding, n_proto = args.k)

    subset_coords['image_prototype'] = labels_image

    save_folder = os.path.join(args.out_dir, f'k_{args.k}', organ_type)
    os.makedirs(save_folder, exist_ok=True)

    subset_coords.to_csv(os.path.join(save_folder, f'{organ_type}_coords_with_clusters.csv'))
    np.save(os.path.join(save_folder, f'{organ_type}_centroids_image.npy'), centroids_image)



def main(args):
    os.makedirs(args.out_dir, exist_ok = True)
    hest_1k = pd.read_csv('/raid/HEST-1K/HEST_v1_0_0.csv', index_col = 0)
    hest_1k_info = hest_1k[['id','species', 'organ', 'disease_state', 'patient']].reset_index()
    
    folds = np.load(args.folds_path, allow_pickle=True)
    df = pd.read_csv(args.coords_data, index_col = 0)

    train_df = df[df['id'].isin(list(folds[0][0]))]
    val_df = df[df['id'].isin(list(folds[0][1]))]

    missing_id = ['NCBI855', 'NCBI854', 'TENX28']
    val_df = val_df[~val_df['id'].isin(missing_id)]
    
    train_df = train_df[~train_df['id'].isin(missing_id)]
    

    organ_types = hest_1k_info['organ'].unique()

    for organ_type in tqdm(organ_types):
        pids = hest_1k_info[hest_1k_info['organ'] == organ_type].id.unique()
        subset_coords_df = train_df[train_df['id'].isin(pids)].reset_index()
        subset_coords_df.drop(columns = ['Unnamed: 0', 'index'], inplace = True)
        run_tissue_type(args, organ_type, subset_coords_df)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--hest_info_csv', type = str, default = '/raid/HEST-1K/HEST_v1_0_0.csv'
    )
    parser.add_argument(
        '--out_dir', type = str, default = '/raid/mpleasure/data_deepstorage/st_projects/visium/clustering_results/kmeans_organ_specific_clustering_90_10_split_VIRCHOW2'
    )
    parser.add_argument(
        '--folds_path', type = str, default = '/raid/eredekop/071024_ST/data/90_10_split.npy'
    )
    parser.add_argument(
        '--image_features_path', type = str, default = "/raid/mpleasure/data_deepstorage/st_projects/visium/virchow2_20x_hest_patching_normalized"
    )
    parser.add_argument(
        '--hvg_matrix_path', type = str, default = "/raid/HEST-1K/071524_hvg_matrix/"
    )
    parser.add_argument(
        '--coords_data', type = str, default = '/raid/mpleasure/data_deepstorage/st_projects/visium/updated_coords_hest_patching_no_mouse.csv'
    )
    parser.add_argument(
        '--k', type = int, default = 16
    )
    parser.add_argument(
        '--nredo', type = int, default = 50
    )
    parser.add_argument(
        '--niter', type = int, default = 5
    )

    args = parser.parse_args()
    main(args)