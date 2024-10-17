python3 train_clean.py \
    --feature_type 'uni' \
    --train_df '/raid/PANDA_challenge/splits/train_df_isup_center.csv' \
    --val_df '/raid/PANDA_challenge/splits/val_df_isup_center.csv' \
    --test_df '/raid/PANDA_challenge/splits/test_df_isup_center.csv' \
    --feat_path '/raid/PANDA_challenge/uni_bags' \
    --save_root '/raid/PANDA_challenge/uni_panda' \
    --exp_name 'uni' \
    --task 'panda' \
    --task_type 'multi'