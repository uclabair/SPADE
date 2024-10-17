python3 evaluate_clean.py \
    --feature_type 'uni' \
    --test_df '/raid/CPTAC/splits/test_df.csv' \
    --feat_path '/raid/CPTAC/protoattn_feats/all_feats' \
    --save_root '/raid/CPTAC/protoattn_feats' \
    --exp_name 'protoattn' \
    --task 'cptac_lung' \
    --task_type 'binary' \
    --checkpoint_to_test '/raid/CPTAC/protoattn_feats/checkpoints/protoattn_1e-3_cptac_lung_0_1.0.pt'