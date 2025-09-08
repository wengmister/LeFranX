#!/bin/bash

python -m lerobot.scripts.train \
  --policy.type diffusion \
  --dataset.repo_id [YOUR_DATASET_REPO_PATH] \
  --dataset.root [YOUR_DATASET_ROOT_PATH] \
  --steps 50000 \
  --batch_size 32 \
  --optimizer.lr 1e-4 \
  --save_freq 5000 \
  --eval_freq 5000 \
  --log_freq 100 \
  --policy.horizon 16 \
  --policy.n_obs_steps 2 \
  --policy.n_action_steps 8 \
  --policy.use_amp false \
  --policy.push_to_hub false \
  --wandb.enable true \
  --wandb.project [YOUR_WANDB_PROJECT] \
  --output_dir [YOUR_OUTPUT_DIR]

echo "Diffusion Policy training completed!"