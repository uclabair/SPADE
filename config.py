hvg_matrix_path = "/raid/HEST-1K/071524_hvg_matrix/"
image_features_path = "/raid/mpleasure/data_deepstorage/st_projects/visium/uni_embeds_224_20x_normalized_no_thresh_08_14/"
folds_path = '/raid/eredekop/071024_ST/data/folds.npy'
coords_data_path_v1 = '/raid/mpleasure/data_deepstorage/st_projects/visium/all_coords_visium_data.csv'
coords_data_path_v2 = '/raid/mpleasure/data_deepstorage/st_projects/visium/all_coords_visium_data_v3.csv'
checkpoint_dir = "/raid/eredekop/071024_ST/checkpoints/"

# Model and Training Configurations
lr = 1e-4
weight_decay = 1e-2
batch_size = 512
num_workers = 8
num_epochs = 100
patience = 2
factor = 0.5


model_name = 'uni'
image_embedding = 1024
spot_embedding = 7968 #number of shared hvgs (change for each dataset)

pretrained = True
trainable = True 
temperature = 1.0

size = 224

# for projection head; used for both image and text encoders
num_projection_layers = 1
projection_dim = 256 
dropout = 0.1
