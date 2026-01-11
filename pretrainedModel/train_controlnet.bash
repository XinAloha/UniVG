#!/bin/bash


# Activate conda environment
source activate xxx

# Train ControlNet with pretrained weights
python3 train_controlnet.py \
    --config ./models/cldm_v15.yaml \
    --pretrained /path/to/pretrained/ldm_checkpoint.ckpt \
    --data_path /path/to/training/data \
    --batch_size 2 \
    --learning_rate 1e-5 \
    --max_epochs 2000 \
    --checkpoint_dir ./control_checkpoints/ \
    --checkpoint_prefix controlnet \
    --save_every_n_epochs 100 \
    --gpus 1 \
    --precision 16
