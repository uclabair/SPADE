python3 evaluate_clean.py \
    --feature_type 'uni' \
    --test_df '/raid/PANDA_challenge/splits/test_df_isup_center.csv' \
    --feat_path '/raid/PANDA_challenge/protoattn_feats/all_feats' \
    --save_root '/raid/PANDA_challenge/protoattn_feats' \
    --exp_name 'uni' \
    --task 'panda' \
    --task_type 'multi' \
    --checkpoint_to_test '/raid/PANDA_challenge/protoattn_feats/checkpoints/protoattn_1e-3_panda_14_0.9437540524924847.pt'