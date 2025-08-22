#!/bin/bash

python -m lerobot.scripts.train \
  --policy.type diffusion \
  --dataset.repo_id local_dataset_franka_wave \
  --dataset.root ./datasets/franka_wave \
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
  --wandb.project lerobot_franka_wave \
  --wandb.notes "Diffusion Policy training on Franka wave dataset" \
  --output_dir ./outputs/diffusion_policy_franka_wave