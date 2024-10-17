python3 main.py --exp_name 'plco_lung_nll_survival' \
    --lr 1e-4 --max_epochs 20 \
    --labels '/raid/mpleasure/PLCO/parsed_data/lung/lung_survival_task.csv' \
    --train_bag_root '/raid/mpleasure/PLCO/parsed_data/lung/splits/proto_attn_feats_image_proto_only/all_feats/train' \
    --val_bag_root '/raid/mpleasure/PLCO/parsed_data/lung/splits/proto_attn_feats_image_proto_only/all_feats/val' \
    --task 'plco_lung' \
    --survival_time_col 'days' \
    --censorship_col 'label' \
    --pid_col 'plco_id' \
    --exp_name 'protoattn_plco_lung'