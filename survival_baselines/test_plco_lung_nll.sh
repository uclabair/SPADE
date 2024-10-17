python3 evaluate.py --exp_name 'protoattn_plco_lung' \
    --model_checkpoint '/raid/mpleasure/data_deepstorage/st_projects/bcr_downstream/survival_models/plco_lung_nll_survival/results/checkpoint_epoch_1_val_0.49550261355384634_c_ind_0.6.pth' \
    --task 'plco_lung' \
    --labels '/raid/mpleasure/PLCO/parsed_data/lung/lung_survival_task.csv' \
    --test_bag_root '/raid/mpleasure/PLCO/parsed_data/lung/splits/test_uni_features' \
    --survival_time_col 'days' \
    --censorship_col 'label' \
    --pid_col 'plco_id' 