python3 train_clean.py \
    --feature_type 'uni' \
    --train_df '/raid/CPTAC/splits/train_df.csv' \
    --val_df '/raid/CPTAC/splits/val_df.csv' \
    --test_df '/raid/CPTAC/splits/test_df.csv' \
    --feat_path '/raid/CPTAC/uni_bag_features' \
    --save_root '/raid/CPTAC/uni_cptac_lung' \
    --exp_name 'uni' \
    --task 'cptac_lung' \
    --task_type 'binary'