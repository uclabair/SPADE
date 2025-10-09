## Repository for: 

SPADE: Spatial Transcriptomics and Pathology Alignment Using a Mixture of Data Experts for an Expressive Latent Space

## Authors: 

Ekaterina Redekop*, Mara Pleasure*, Zichen Wang, Kimberly Flores, Anthony Sisk, William Speier, Corey W. Arnold  (*Indicates shared co-first authors)

### Link to paper: https://arxiv.org/abs/2506.21857v1


## Training:

```bash 
python main.py \
--protos 0,1,2 \
--exp_name uni_k16
```


## Extracting SPADE embeddings with distance weighting

We compute per-tile embeddings by routing path features through a Mixture-of-Data-Experts head. 

```bash
python experts_routing.py \
  --input_bags_dir path/to/bags \
  --output_dir path/to/out \
  --checkpoint_dir path/to/checkpoints \
  --checkpoint_regex 'ep([0-9]+)_trloss([0-9]+\.[0-9]+)_valloss[0-9]+\.[0-9]+_proto([0-9]+)\.pt' \
  --num_experts 16 \
  --centroids_path path/to/centroids.npy \
  --reassigned_clusters_path path/to/reassigned_clusters.npy \
  --feat_dim 1024 \
  --proj_dim 256 \
  --temperature 0.1 \
  --half --device cuda
```
