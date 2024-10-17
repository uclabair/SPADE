python3 evaluate.py --exp_name 'protoattn_plco_lung' \
    --model_checkpoint '/raid/mpleasure/data_deepstorage/st_projects/bcr_downstream/survival_models/protoattn_plco_lung_10_14/results/checkpoint_epoch_9_val_0.4762724756642624_c_ind_0.75.pth' \
    --task 'plco_lung' \
    --labels '/raid/mpleasure/PLCO/parsed_data/lung/lung_survival_task.csv' \
    --test_bag_root '/raid/mpleasure/PLCO/parsed_data/lung/splits/proto_attn_feats_image_proto_only/all_feats/test' \
    --survival_time_col 'days' \
    --censorship_col 'label' \
    --pid_col 'plco_id' 