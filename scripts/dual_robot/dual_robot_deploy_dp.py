#!/usr/bin/env python3
"""
Direct deployment script for combo robot that bypasses draccus CLI parsing.
"""

import sys
from pathlib import Path
import time
import torch
import numpy as np
import rerun as rr

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from lerobot.robots.franka_fer_xhand.franka_fer_xhand import FrankaFERXHand
from lerobot.robots.franka_fer_xhand.franka_fer_xhand_config import FrankaFERXHandConfig
from lerobot.robots.franka_fer.franka_fer_config import FrankaFERConfig
from lerobot.robots.xhand.xhand_config import XHandConfig
from lerobot.cameras.opencv.configuration_opencv import OpenCVCameraConfig
from lerobot.cameras.configs import ColorMode
from lerobot.policies.diffusion.modeling_diffusion import DiffusionPolicy
from lerobot.policies.diffusion.configuration_diffusion import DiffusionConfig
from lerobot.configs.types import PolicyFeature, FeatureType, NormalizationMode

def main():
    print("=== Combo Robot Policy Deployment ===")
    
    # Tunable parameters
    ACTION_SCALE = 1.0  # Scale actions (< 1.0 for safer/slower movements)
    SMOOTHING_ALPHA = 0.5  # 1.0 = no smoothing (try this first for DP)
    OVERRIDE_INFERENCE_STEPS = 25  # Reduce from 50 to speed up (trade-off with quality)
    USE_DDIM = False  # Use DDIM scheduler for faster inference (deterministic)
    
    print(f"Settings: ACTION_SCALE={ACTION_SCALE}, SMOOTHING_ALPHA={SMOOTHING_ALPHA}")
    print(f"Inference optimization: steps={OVERRIDE_INFERENCE_STEPS}, DDIM={USE_DDIM}")
    
    # Create robot configuration
    arm_config = FrankaFERConfig(
        server_ip="[YOUR_ROBOT_IP]",  # USE YOUR ROBOT IP
        server_port=5000,
        home_position=[0, -0.785, 0, -2.356, 0, 1.571, -0.9],
        cameras={}
    )
    
    hand_config = XHandConfig(
        protocol="RS485",
        serial_port="/dev/ttyUSB0",
        baud_rate=3000000,
        hand_id=0,
        control_frequency=30.0,
        max_torque=250.0,
        cameras={}
    )
    
    cameras = {
        "tpv": OpenCVCameraConfig(
            index_or_path="/dev/video11",
            fps=30,
            width=320,
            height=240,
            color_mode=ColorMode.RGB
        ),
        # Add more cameras as needed
    }
    
    robot_config = FrankaFERXHandConfig(
        arm_config=arm_config,
        hand_config=hand_config,
        cameras=cameras,
        synchronize_actions=True,
        action_timeout=0.2,
        check_arm_hand_collision=True,
        emergency_stop_both=True
    )
    
    # Create robot
    robot = FrankaFERXHand(robot_config)
    
    # Load policy
    policy_path = Path("[YOUR_POLICY_PATH]")  # USE YOUR POLICY PATH
    print(f"Loading policy from {policy_path}")
    
    # Find the latest checkpoint directory
    checkpoint_dirs = sorted([d for d in policy_path.iterdir() if d.is_dir() and d.name.isdigit()])
    if not checkpoint_dirs:
        print("No checkpoint directories found!")
        return 1
    
    latest_checkpoint_dir = checkpoint_dirs[-1]
    print(f"Using checkpoint: {latest_checkpoint_dir}")
    
    # Load config
    config_path = latest_checkpoint_dir / "pretrained_model" / "config.json"
    with open(config_path) as f:
        import json
        config_dict = json.load(f)
    
    # Convert input_features and output_features to PolicyFeature objects
    input_features = {}
    for key, value in config_dict.get('input_features', {}).items():
        input_features[key] = PolicyFeature(
            type=FeatureType[value['type']],
            shape=tuple(value['shape'])
        )
    
    output_features = {}
    for key, value in config_dict.get('output_features', {}).items():
        output_features[key] = PolicyFeature(
            type=FeatureType[value['type']],
            shape=tuple(value['shape'])
        )
    
    # Convert normalization_mapping strings to NormalizationMode enums
    normalization_mapping = {}
    for key, value in config_dict.get('normalization_mapping', {}).items():
        normalization_mapping[key] = NormalizationMode[value]
    
    # Create DiffusionConfig with only the fields it expects
    config = DiffusionConfig(
        n_obs_steps=config_dict.get('n_obs_steps', 2),
        normalization_mapping=normalization_mapping,
        input_features=input_features,
        output_features=output_features,
        device=config_dict.get('device', 'cuda'),
        horizon=config_dict.get('horizon', 16),
        n_action_steps=config_dict.get('n_action_steps', 8),
        vision_backbone=config_dict.get('vision_backbone', 'resnet18'),
        pretrained_backbone_weights=config_dict.get('pretrained_backbone_weights'),
        crop_shape=config_dict.get('crop_shape'),
        crop_is_random=config_dict.get('crop_is_random', True),
        use_group_norm=config_dict.get('use_group_norm', True),
        spatial_softmax_num_keypoints=config_dict.get('spatial_softmax_num_keypoints', 80),
        use_separate_rgb_encoder_per_camera=config_dict.get('use_separate_rgb_encoder_per_camera', False),
        down_dims=tuple(config_dict.get('down_dims', [256, 512])),
        kernel_size=config_dict.get('kernel_size', 3),
        n_groups=config_dict.get('n_groups', 8),
        diffusion_step_embed_dim=config_dict.get('diffusion_step_embed_dim', 128),
        use_film_scale_modulation=config_dict.get('use_film_scale_modulation', True),
        noise_scheduler_type=config_dict.get('noise_scheduler_type', 'DDPM').upper(),
        num_train_timesteps=config_dict.get('num_train_timesteps', 100),
        beta_schedule=config_dict.get('beta_schedule', 'squaredcos_cap_v2'),
        beta_start=config_dict.get('beta_start', 0.0001),
        beta_end=config_dict.get('beta_end', 0.02),
        prediction_type=config_dict.get('prediction_type', 'epsilon'),
        clip_sample=config_dict.get('clip_sample', True),
        clip_sample_range=config_dict.get('clip_sample_range', 1.0),
        num_inference_steps=config_dict.get('num_inference_steps'),
        do_mask_loss_for_padding=config_dict.get('do_mask_loss_for_padding', False)
    )
    
    # Try to load actual stats from dataset if available
    stats_path = policy_path / "stats.json"
    if not stats_path.exists():
        # Try parent directory
        stats_path = policy_path.parent / "stats.json"
    if stats_path.exists():
        import json
        with open(stats_path) as f:
            raw_stats = json.load(f)
        # Convert to torch tensors
        stats = {}
        for key in ["observation.state", "observation.images.tpv", "observation.images.wrist", "observation.images.overhead", "action"]:
            if key in raw_stats:
                stats[key] = {}
                # Load all available statistics (mean/std for images, min/max for state/action)
                for stat_type in ["mean", "std", "min", "max"]:
                    if stat_type in raw_stats[key]:
                        stats[key][stat_type] = torch.tensor(raw_stats[key][stat_type])
        print("Loaded dataset statistics from stats.json")
        # Debug action stats
        if "action" in stats:
            action_stats = stats["action"]
            print(f"Action stats loaded:")
            for stat_name, stat_value in action_stats.items():
                if hasattr(stat_value, 'shape'):
                    print(f"  {stat_name}: shape={stat_value.shape}, range=[{stat_value.min():.3f}, {stat_value.max():.3f}]")
                else:
                    print(f"  {stat_name}: {stat_value}")
        else:
            print("WARNING: No action stats found!")
    else:
        # Create policy with dummy stats (will be normalized by the model)
        print("Warning: No stats.json found, using dummy normalization")
        stats = {
            "observation.state": {
                "min": torch.ones(54) * -1,
                "max": torch.ones(54),
                "mean": torch.zeros(54),
                "std": torch.ones(54)
            },
            "observation.images.tpv": {
                "mean": torch.zeros(3, 240, 320),
                "std": torch.ones(3, 240, 320)
            },
            "observation.images.wrist": {
                "mean": torch.zeros(3, 240, 320),
                "std": torch.ones(3, 240, 320)
            },
            "observation.images.overhead": {
                "mean": torch.zeros(3, 240, 320),
                "std": torch.ones(3, 240, 320)
            },
            "action": {
                "min": torch.ones(19) * -1,
                "max": torch.ones(19),
                "mean": torch.zeros(19),
                "std": torch.ones(19)
            }
        }
    
    # Override settings for faster inference
    if OVERRIDE_INFERENCE_STEPS:
        config.num_inference_steps = OVERRIDE_INFERENCE_STEPS
        print(f"Overriding num_inference_steps to {OVERRIDE_INFERENCE_STEPS} (was {config_dict.get('num_inference_steps', 50)})")
    
    if USE_DDIM:
        config.noise_scheduler_type = "DDIM"
        print(f"Using DDIM scheduler for faster inference (was {config_dict.get('noise_scheduler_type', 'DDPM')})")
    
    # Use the original crop_shape from training
    print(f"Using crop_shape={config.crop_shape} (matches training)")
    # Note: The model was actually trained with 84x84 crops despite the training script saying None
    
    policy = DiffusionPolicy(config, dataset_stats=stats)
    
    # Load model weights from safetensors
    model_path = latest_checkpoint_dir / "pretrained_model" / "model.safetensors"
    from safetensors.torch import load_file
    state_dict = load_file(model_path)
    policy.load_state_dict(state_dict)
    policy.eval()
    policy.to("cuda")
    
    print("Policy loaded successfully")
    
    # Connect robot
    print("Connecting to robot...")
    robot.connect(calibrate=False)
    
    if not robot.is_connected:
        print("Failed to connect to robot!")
        return 1
    
    print("Robot connected successfully")
    
    # Initialize Rerun
    rr.init("combo_robot_deployment", spawn=True)
    
    # Home robot
    print("Homing robot...")
    robot.reset_to_home()
    time.sleep(2)
    
    # Main control loop
    print("\n=== Starting control loop ===")
    print("Press Ctrl+C to stop")
    
    fps = 30
    dt = 1.0 / fps
    frame_idx = 0
    
    # Action smoothing
    action_smoothing_alpha = SMOOTHING_ALPHA
    prev_action = None
    
    # Observation history for diffusion policy (needs n_obs_steps)
    n_obs_steps = config.n_obs_steps
    obs_history_states = []
    obs_history_images = {
        "tpv": [],
        # Add more video streams as needed
    }
    
    # Action chunking
    action_chunk = None
    chunk_idx = 0
    n_action_steps = config.n_action_steps  # Use all 8 actions from the chunk
    
    try:
        while True:
            start_time = time.perf_counter()
            
            # Set rerun time
            rr.set_time_sequence("frame", frame_idx)
            rr.set_time_seconds("time", time.time())
            
            # Get observation
            obs = robot.get_observation()
            
            # Prepare observation for policy
            # Combine arm and hand states into environment_state
            env_state = []
            
            # Add arm joint positions (7)
            for i in range(7):
                env_state.append(obs[f"arm_joint_{i}.pos"])
            
            # Add arm joint velocities (7)
            for i in range(7):
                env_state.append(obs[f"arm_joint_{i}.vel"])
            
            # Add ee_pose (16)
            for i in range(16):
                env_state.append(obs[f"arm_ee_pose.{i:02d}"])
            
            # Add hand joint positions (12)
            for i in range(12):
                env_state.append(obs[f"hand_joint_{i}.pos"])

            # Add hand joint torque (12)
            for i in range(12):
                env_state.append(obs[f"hand_joint_{i}.torque"])
            
            env_state = np.array(env_state, dtype=np.float32)
            
            # Process current observation
            import torch.nn.functional as F
            tpv_image = torch.FloatTensor(obs["tpv"]).permute(2, 0, 1) / 255.0
            wrist_image = torch.FloatTensor(obs["wrist"]).permute(2, 0, 1) / 255.0
            overhead_image = torch.FloatTensor(obs["overhead"]).permute(2, 0, 1) / 255.0
            
            # Resize to expected size
            tpv_image = F.interpolate(tpv_image.unsqueeze(0), size=(240, 320), mode='bilinear', align_corners=False).squeeze(0)
            wrist_image = F.interpolate(wrist_image.unsqueeze(0), size=(240, 320), mode='bilinear', align_corners=False).squeeze(0)
            overhead_image = F.interpolate(overhead_image.unsqueeze(0), size=(240, 320), mode='bilinear', align_corners=False).squeeze(0)
            
            # Update observation history
            obs_history_states.append(env_state)
            obs_history_images["tpv"].append(tpv_image)
            obs_history_images["wrist"].append(wrist_image)
            obs_history_images["overhead"].append(overhead_image)
            
            # Keep only last n_obs_steps
            if len(obs_history_states) > n_obs_steps:
                obs_history_states = obs_history_states[-n_obs_steps:]
                for key in obs_history_images:
                    obs_history_images[key] = obs_history_images[key][-n_obs_steps:]
            
            # Generate new action chunk when needed
            if action_chunk is None or chunk_idx >= n_action_steps:
                if len(obs_history_states) == n_obs_steps:
                    # Stack observation history
                    state_stack = torch.FloatTensor(np.stack(obs_history_states)).unsqueeze(0).cuda()  # (1, n_obs_steps, state_dim)
                    tpv_stack = torch.stack(obs_history_images["tpv"]).unsqueeze(0).cuda()  # (1, n_obs_steps, 3, H, W)
                    wrist_stack = torch.stack(obs_history_images["wrist"]).unsqueeze(0).cuda()
                    overhead_stack = torch.stack(obs_history_images["overhead"]).unsqueeze(0).cuda()
                    
                    batch = {
                        "observation.state": state_stack,
                        "observation.images.tpv": tpv_stack,
                        "observation.images.wrist": wrist_stack,
                        "observation.images.overhead": overhead_stack
                    }
                    
                    # Normalize inputs
                    batch = policy.normalize_inputs(batch)
                    
                    # Stack images for the model
                    batch["observation.images"] = torch.stack([
                        batch["observation.images.tpv"],
                        batch["observation.images.wrist"],
                        batch["observation.images.overhead"]
                    ], dim=-4)  # (1, n_obs_steps, 3, 3, H, W)
                    
                    # Generate action chunk using the diffusion model
                    with torch.no_grad():
                        inference_start = time.perf_counter()
                        actions_raw = policy.diffusion.generate_actions(batch)  # (1, horizon, action_dim)
                        
                        # Debug: Check raw actions
                        print(f"Raw actions shape: {actions_raw.shape}, range: [{actions_raw.min():.3f}, {actions_raw.max():.3f}]")
                        
                        # Unnormalize actions
                        actions = policy.unnormalize_outputs({"action": actions_raw})["action"]
                        
                        # Debug: Check unnormalized actions
                        print(f"Unnormalized actions range: [{actions.min():.3f}, {actions.max():.3f}]")
                        
                        # Extract the action chunk (first n_action_steps from horizon)
                        action_chunk = actions[0, :n_action_steps].cpu().numpy()  # (n_action_steps, 19)
                        
                        inference_time = (time.perf_counter() - inference_start) * 1000
                        print(f"Generated new action chunk: {action_chunk.shape}, inference time: {inference_time:.1f}ms")
                        print(f"Action chunk range: [{action_chunk.min():.3f}, {action_chunk.max():.3f}]")
                    
                    chunk_idx = 0
                else:
                    # Not enough history yet
                    action_chunk = np.zeros((n_action_steps, 19))
                    chunk_idx = 0
            
            # Use current action from chunk
            action = action_chunk[chunk_idx]
            
            # Debug: Check extracted action
            if frame_idx % 10 == 0:  # More frequent debugging
                print(f"Using action {chunk_idx+1}/{n_action_steps} from chunk")
                print(f"Extracted action shape: {action.shape}, range: [{action.min():.3f}, {action.max():.3f}]")
                print(f"Action chunk shape: {action_chunk.shape}")
                print(f"First few actions: {action[:5]}")
            
            chunk_idx += 1
            
            # Store raw action for debugging
            raw_action = action.copy()
            
            # Apply exponential smoothing
            if prev_action is not None:
                action_before_smooth = action.copy()
                action = action_smoothing_alpha * action + (1 - action_smoothing_alpha) * prev_action
                if frame_idx % 10 == 0:
                    print(f"Before smooth: [{action_before_smooth.min():.3f}, {action_before_smooth.max():.3f}]")
                    print(f"After smooth: [{action.min():.3f}, {action.max():.3f}]")
            prev_action = action.copy()
            
            # Apply action scaling for safety
            action = action * ACTION_SCALE
            
            # Debug final action
            if frame_idx % 10 == 0:
                print(f"Final action range: [{action.min():.3f}, {action.max():.3f}]")
            
            # Split into arm and hand actions
            action_dict = {}
            
            # Arm actions (first 7)
            for i in range(7):
                action_dict[f"arm_joint_{i}.pos"] = float(action[i])
            
            # Hand actions (next 12)
            for i in range(12):
                action_dict[f"hand_joint_{i}.pos"] = float(action[7 + i])
            
            # Debug: Print hand actions periodically to see if they're changing
            if frame_idx % 30 == 0:  # Every second
                raw_hand = [raw_action[7+i] for i in range(12)]
                smooth_hand = [action[7+i] for i in range(12)]
                print(f"RAW hand actions: mean={np.mean(raw_hand):.3f}, std={np.std(raw_hand):.3f}, range=[{min(raw_hand):.3f}, {max(raw_hand):.3f}]")
                print(f"SMOOTH hand actions: mean={np.mean(smooth_hand):.3f}, std={np.std(smooth_hand):.3f}")
                # Print specific joints to see if they're trying to move
                for i in [0, 4, 6, 8]:
                    print(f"  Joint {i}: current={obs[f'hand_joint_{i}.pos']:.3f}, raw_target={raw_action[7+i]:.3f}, smooth_target={action[7+i]:.3f}")
            
            # Send action to robot
            robot.send_action(action_dict)
            
            # Log to Rerun
            # Log arm joint positions
            for i in range(7):
                rr.log(f"robot/arm/joint_{i}/position", rr.Scalar(obs[f"arm_joint_{i}.pos"]))
                rr.log(f"robot/arm/joint_{i}/velocity", rr.Scalar(obs[f"arm_joint_{i}.vel"]))
                rr.log(f"robot/arm/joint_{i}/action", rr.Scalar(action_dict[f"arm_joint_{i}.pos"]))
            
            # Log hand joint positions
            for i in range(12):
                rr.log(f"robot/hand/joint_{i}/position", rr.Scalar(obs[f"hand_joint_{i}.pos"]))
                rr.log(f"robot/hand/joint_{i}/action", rr.Scalar(action_dict[f"hand_joint_{i}.pos"]))
            
            # Log camera images if available
            if "tpv" in obs:
                rr.log("cameras/tpv", rr.Image(obs["tpv"]))
            if "wrist" in obs:
                rr.log("cameras/wrist", rr.Image(obs["wrist"]))
            if "overhead" in obs:
                rr.log("cameras/overhead", rr.Image(obs["overhead"]))
            
            # Log end-effector pose as 4x4 matrix
            ee_pose = np.array([obs[f"arm_ee_pose.{i:02d}"] for i in range(16)]).reshape(4, 4)
            rr.log("robot/arm/ee_pose", rr.Transform3D(mat3x3=ee_pose[:3, :3], translation=ee_pose[:3, 3]))
            
            # Maintain loop timing
            elapsed = time.perf_counter() - start_time
            if elapsed < dt:
                time.sleep(dt - elapsed)
            
            # Debug print every second
            if frame_idx % fps == 0:
                print(f"Control loop running... (frame {frame_idx}, loop time: {elapsed*1000:.1f}ms)")
            
            frame_idx += 1
    
    except KeyboardInterrupt:
        print("\n=== Stopping control loop ===")
    except Exception as e:
        print(f"Error in control loop: {e}")
        import traceback
        traceback.print_exc()
    finally:
        # Cleanup
        print("Disconnecting robot...")
        if robot.is_connected:
            robot.stop()
            robot.disconnect()
        print("Done!")
    
    return 0

if __name__ == "__main__":
    sys.exit(main())