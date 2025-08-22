#!/bin/bash

echo "Training ACT policy on franka_wave dataset..."

python -m lerobot.scripts.train \
    --policy.type=act \
    --dataset.repo_id=local_dataset_franka_wave \
    --dataset.root=./datasets/franka_wave \
    --batch_size=32 \
    --steps=100000 \
    --optimizer.lr=1e-4 \
    --scheduler.type=cosine_decay_with_warmup \
    --scheduler.num_warmup_steps=1000 \
    --scheduler.num_decay_steps=99000 \
    --scheduler.peak_lr=1e-4 \
    --scheduler.decay_lr=1e-6 \
    --policy.n_action_steps=8 \
    --policy.n_obs_steps=1 \
    --policy.chunk_size=8 \
    --policy.push_to_hub=false \
    --wandb.enable=true \
    --wandb.project=lerobot_act_franka \
    --policy.device=cuda \
    --eval_freq=10000 \
    --save_freq=10000 \
    --output_dir=./outputs/act_policy_franka_wave

echo "ACT training completed!"