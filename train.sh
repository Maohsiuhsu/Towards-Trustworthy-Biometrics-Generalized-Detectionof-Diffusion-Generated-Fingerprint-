#!/usr/bin/env bash

# Example training command for open-source release.
# Update arguments based on your dataset and experiment design.
CUDA_VISIBLE_DEVICES=0 python train_mix.py \
  --name experiments/default_train \
  --fake_data_name 'DDIM,inpaint_with_FK,guided' \
  --real_data_name real \
  --data_root ./data \
  --flip \
  --dataset_mode normal_v2_2 \
  --batch_size 128 \
  --lr 0.0001
