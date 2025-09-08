#!/usr/bin/env python3
"""
Training script for Diffusion Policy that bypasses draccus for config parsing.
Uses the official lerobot training pipeline like the ACT script.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import torch
from lerobot.configs.train import TrainPipelineConfig
from lerobot.configs.default import DatasetConfig, WandBConfig
from lerobot.policies.diffusion.configuration_diffusion import DiffusionConfig
from lerobot.configs.types import PolicyFeature, FeatureType, NormalizationMode

def main():
    """Run diffusion policy training with manually constructed config."""
    
    # Import the train function directly
    import lerobot.scripts.train as train_module
    
    # Get the unwrapped train function
    train_func = train_module.train.__wrapped__
    
    # Dataset config with cropped dataset path
    dataset_config = DatasetConfig(
        repo_id="[YOUR_DATASET_REPO_PATH]"  # Absolute local path
    )
    
    # Create Diffusion policy config with manually specified input features
    # to avoid the automatic detection that causes shape mismatch
    policy_config = DiffusionConfig(
        # Use only tpv camera to avoid preprocessing complexity
        input_features={
            "observation.state": PolicyFeature(
                type=FeatureType.STATE,
                shape=(54,)  # Combined arm (14) + hand (12) + ee_pose (16) state + 12 torque
            ),
            "observation.images.tpv": PolicyFeature(
                type=FeatureType.VISUAL,
                shape=(3, 240, 320)  # tpv camera original size
            ),
            "observation.images.wrist": PolicyFeature(
                type=FeatureType.VISUAL,
                shape=(3, 240, 320)  # wrist camera original size
            ),
            "observation.images.overhead": PolicyFeature(
                type=FeatureType.VISUAL,
                shape=(3, 240, 320)  # overhead camera original size
            )
        },
        output_features={
            "action": PolicyFeature(
                type=FeatureType.ACTION,
                shape=(19,)  # Arm (7) + hand (12) actions
            )
        },
        
        # Normalization mapping
        normalization_mapping={
            "STATE": NormalizationMode.MIN_MAX,
            "VISUAL": NormalizationMode.MEAN_STD, 
            "ACTION": NormalizationMode.MIN_MAX,
        },
        
        # Adjust the config for the robot and task as you need
        # Observation and action configuration
        n_obs_steps=2,
        horizon=16,
        n_action_steps=8,

        # Diffusion specific - optimized for fast inference
        num_inference_steps=25,
        num_train_timesteps=100,
        diffusion_step_embed_dim=128,
        noise_scheduler_type="DDPM",
        prediction_type="epsilon",
        
        # Vision backbone architecture
        vision_backbone="resnet18",
        pretrained_backbone_weights="ResNet18_Weights.IMAGENET1K_V1",
        use_group_norm=False,
        crop_shape=None,
        crop_is_random=False,
        spatial_softmax_num_keypoints=80,  # 8×10 feature map = 80 positions for 320×240 input
        
        # UNet diffusion architecture (for action denoising) - optimized for speed
        down_dims=(256, 512), 
        kernel_size=3,
        n_groups=8,
        diffusion_step_embed_dim=128,
        use_film_scale_modulation=True,
        
        # Training
        repo_id="diffusion-policy-orange_cube_with_force",
        push_to_hub=False
    )
    
    # Create wandb config
    wandb_config = WandBConfig(
        enable=True,  # Enable wandb for tracking
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
        save_freq=2000,
        wandb=wandb_config
    )
    
    print("Starting Diffusion Policy training...")
    print(f"Dataset: {config.dataset.repo_id}")
    print(f"Policy: Diffusion with ResNet18 backbone")
    print(f"Output dir: {config.output_dir}")
    
    # Call the unwrapped train function
    train_func(config)
    
    print("Training completed!")

if __name__ == "__main__":
    main()