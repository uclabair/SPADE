python3 evaluate_clean.py \
    --feature_type 'uni' \
    --test_df '/raid/PANDA_challenge/splits/test_df_isup_center.csv' \
    --feat_path '/raid/PANDA_challenge/uni_bags' \
    --save_root '/raid/PANDA_challenge/uni_panda' \
    --exp_name 'uni' \
    --task 'panda' \
    --task_type 'multi' \
    --checkpoint_to_test '/raid/PANDA_challenge/uni_panda/checkpoints/uni_panda_6_0.9232872389750946.pt'