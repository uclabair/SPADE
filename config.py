# ---------------------------------------------------------------------
# Data Paths (update these locally)
# ---------------------------------------------------------------------

# Path to NumPy HVG matrices (each file: hvg_matrix_<slide_id>.npy)
hvg_matrix_path = "data/hvg_matrices/"

# Path to precomputed image features (each slide folder with <x>_<y>.npy)
image_features_path = "data/image_features/"

# Optional: precomputed fold split (NumPy array or pickle)
folds_path = "data/folds.npy"

# Coordinates CSV describing per-spot patch positions and slide IDs
coords_data_path_v1 = "data/coords.csv"

# (Optional) secondary coordinates file
coords_data_path_v2 = "data/all_coords.csv"

# Directory for model checkpoints
checkpoint_dir = "outputs/checkpoints/"

# ---------------------------------------------------------------------
# Model & Training Configurations
# ---------------------------------------------------------------------
lr = 1e-4
weight_decay = 1e-2
batch_size = 128
num_workers = 2
num_epochs = 100
patience = 2          # for optional LR scheduler w/ ReduceLROnPlateau
factor = 0.5          # LR decay factor when patience exhausted

# ---------------------------------------------------------------------
# Model Architecture
# ---------------------------------------------------------------------
image_embedding = 768           # dim of precomputed image features
spot_embedding = 7986            # dim of HVG feature vectors
projection_dim = 256             # latent projection dimension
num_projection_layers = 1        # MLP depth
dropout = 0.1                    # dropout prob.
temperature = 1.0                # contrastive temperature


seed = 42
device = "cuda"     # or "cpu"
