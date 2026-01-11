#!/bin/bash

# Activate conda environment
source activate xxx

# Run ControlNet inference
python3 inference_controlnet.py \
    --config ./models/cldm_v15.yaml \
    --checkpoint /path/to/trained/checkpoint.ckpt \
    --input_path /path/to/input/masks \
    --output_path /path/to/output/images \
    --prompt "" \
    --a_prompt "" \
    --n_prompt "" \
    --num_samples 1 \
    --image_resolution 512 \
    --ddim_steps 20 \
    --strength 1.0 \
    --scale 9.0 \
    --seed -1 \
    --eta 0.0 \
    --device cuda
