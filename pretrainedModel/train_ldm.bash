#!/bin/bash

# Activate conda environment
source activate xxx

# Train Latent Diffusion Model
python3 train_ldm.py \
    --config ./models/ldm_v15.yaml \
    --resume_path /path/to/sd-v1-5-pruned.ckpt \
    --data_path /path/to/training/images \
    --batch_size 2 \
    --learning_rate 1e-4 \
    --max_epochs 100 \
    --checkpoint_dir ./ldm_checkpoints/ \
    --checkpoint_prefix ldm \
    --save_every_n_epochs 10 \
    --log_freq 10 \
    --gpus 1 \
    --precision 16
