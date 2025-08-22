#!/bin/bash

python -m lerobot.record \
  --robot.type=franka_fer \
  --robot.server_ip=192.168.18.1 \
  --robot.server_port=5000 \
  --policy.path=./outputs/diffusion_policy_franka_wave/checkpoints/050000/pretrained_model \
  --dataset.fps=30 \
  --dataset.episode_time_s=30 \
  --dataset.num_episodes=1 \
  --dataset.root=./datasets/policy_rollout \
  --dataset.repo_id=local/eval_policy_rollout \
  --dataset.single_task="Policy deployment test" \
  --dataset.push_to_hub=false