#!/usr/bin/env python3
"""
Training script that bypasses draccus for robot config parsing.
This avoids the circular import issue while still using the official lerobot training pipeline.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import torch
from lerobot.configs.train import TrainPipelineConfig
from lerobot.configs.default import DatasetConfig, WandBConfig
from lerobot.policies.act.configuration_act import ACTConfig

def main():
    """Run training with manually constructed config to bypass draccus issues."""
    
    # We need to import the train function directly and call it without the wrapper
    # to avoid the draccus parsing that causes circular imports
    import lerobot.scripts.train as train_module
    
    # Get the unwrapped train function
    train_func = train_module.train.__wrapped__  # This gets the original function without the @parser.wrap() decorator
    
    # Manually construct the config to avoid draccus parsing issues
    # Use the local dataset instead of trying to download from hub
    dataset_config = DatasetConfig(
        repo_id="[YOUR_DATASET_REPO_PATH]"  # Absolute local path
    )
    
    # Create ACT policy config - we'll use the basic config and let the system auto-detect robot features
    policy_config = ACTConfig(
        n_obs_steps=1,
        chunk_size=8,
        n_action_steps=8,
        dim_model=512,
        dim_feedforward=3200,
        n_heads=8,
        n_encoder_layers=4,
        n_decoder_layers=1,
        dropout=0.1,
        pre_norm=False,
        vision_backbone="resnet18",
        use_vae=True,
        kl_weight=10.0,
        repo_id="[REPO_ID_ACT]",  # Add repo_id to satisfy validation
        push_to_hub=False  # Disable pushing to hub
    )
    
    # Create wandb config
    wandb_config = WandBConfig(
        enable=True,
        project="[YOUR_WANDB_PROJECT]"
    )
    
    # Create training pipeline config
    config = TrainPipelineConfig(
        dataset=dataset_config,
        env=None,  # No environment for offline training
        policy=policy_config,
        output_dir=Path("./outputs/policy/[YOUR_OUTPUT_DIR]"),
        job_name="[YOUR_JOB_NAME]",
        batch_size=16,
        steps=100000,
        eval_freq=2000,
        save_freq=10000,
        wandb=wandb_config
    )
    
    print(f"Dataset: {config.dataset.repo_id}")
    print(f"Policy: ACT")
    print(f"Output dir: {config.output_dir}")
    
    # Call the unwrapped train function directly
    train_func(config)
    
    print("Training completed!")

if __name__ == "__main__":
    main()