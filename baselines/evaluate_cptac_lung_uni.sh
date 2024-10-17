python3 evaluate_clean.py \
    --feature_type 'uni' \
    --test_df '/raid/CPTAC/splits/test_df.csv' \
    --feat_path '/raid/CPTAC/uni_bag_features' \
    --save_root '/raid/CPTAC/uni_cptac_lung' \
    --exp_name 'uni' \
    --task 'cptac_lung' \
    --task_type 'binary' \
    --checkpoint_to_test '/raid/CPTAC/uni_cptac_lung/checkpoints/uni_cptac_lung_19_1.0.pt'