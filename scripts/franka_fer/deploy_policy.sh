#!/bin/bash

python -m lerobot.record \
  --robot.type=franka_fer \
  --robot.server_ip=[YOUR_ROBOT_IP] \
  --robot.server_port=5000 \
  --policy.path=[YOUR_POLICY_PATH] \
  --dataset.fps=30 \
  --dataset.episode_time_s=30 \
  --dataset.num_episodes=1 \
  --dataset.root=[YOUR_DATASET_ROOT] \
  --dataset.repo_id=[YOUR_REPO_ID] \
  --dataset.single_task="Policy deployment" \
  --dataset.push_to_hub=false