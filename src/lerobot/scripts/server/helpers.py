# Copyright 2025 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import logging
import logging.handlers
import os
import time
from dataclasses import dataclass
from pathlib import Path

import torch

from lerobot.configs.types import PolicyFeature
from lerobot.constants import OBS_IMAGES, OBS_STATE
from lerobot.datasets.utils import build_dataset_frame, hw_to_dataset_features

# NOTE: Configs need to be loaded for the client to be able to instantiate the policy config
from lerobot.policies import ACTConfig, DiffusionConfig, PI0Config, SmolVLAConfig, VQBeTConfig  # noqa: F401
from lerobot.robots.robot import Robot
from lerobot.utils.utils import init_logging

Action = torch.Tensor
ActionChunk = torch.Tensor

# observation as received from the robot
RawObservation = dict[str, torch.Tensor]

# observation as those recorded in LeRobot dataset (keys are different)
LeRobotObservation = dict[str, torch.Tensor]

# observation, ready for policy inference (image keys resized)
Observation = dict[str, torch.Tensor]


def visualize_action_queue_size(action_queue_size: list[int], inference_times: list[float] = None) -> None:
    """Display action queue and inference time statistics with optional plots"""
    if not action_queue_size and not inference_times:
        print("📊 PERFORMANCE STATS: No data collected")
        return
    
    # Calculate statistics
    import numpy as np
    
    print("\n" + "="*80)
    print("📊 POLICY DEPLOYMENT PERFORMANCE STATISTICS")
    print("="*80)
    
    # Action queue statistics
    if action_queue_size:
        queue_sizes = np.array(action_queue_size)
        total_steps = len(queue_sizes)
        avg_size = np.mean(queue_sizes)
        min_size = np.min(queue_sizes)
        max_size = np.max(queue_sizes)
        std_size = np.std(queue_sizes)
        
        # Calculate percentage of time queue was empty
        empty_percentage = (queue_sizes == 0).sum() / total_steps * 100
        
        print("🔄 ACTION QUEUE STATISTICS")
        print("-" * 40)
        print(f"Total control steps: {total_steps}")
        print(f"Average queue size: {avg_size:.2f}")
        print(f"Min queue size: {min_size}")
        print(f"Max queue size: {max_size}")
        print(f"Queue size std dev: {std_size:.2f}")
        print(f"Queue empty {empty_percentage:.1f}% of the time")
        
        # Performance assessment
        if empty_percentage > 50:
            queue_performance = "🔴 POOR - Queue often empty, inference too slow"
        elif empty_percentage > 20:
            queue_performance = "🟡 FAIR - Some queue depletion"
        elif empty_percentage > 5:
            queue_performance = "🟢 GOOD - Minimal queue depletion"
        else:
            queue_performance = "🟢 EXCELLENT - Queue well maintained"
        
        print(f"Queue performance: {queue_performance}")
        print()
    
    # Inference time statistics
    if inference_times:
        inference_ms = np.array(inference_times)
        total_inferences = len(inference_ms)
        avg_inference = np.mean(inference_ms)
        min_inference = np.min(inference_ms)
        max_inference = np.max(inference_ms)
        std_inference = np.std(inference_ms)
        p95_inference = np.percentile(inference_ms, 95)
        
        # Calculate inference rate (inferences per second)
        inference_rate = 1000.0 / avg_inference if avg_inference > 0 else 0
        
        print("⚡ INFERENCE TIME STATISTICS")
        print("-" * 40)
        print(f"Total inferences: {total_inferences}")
        print(f"Average time: {avg_inference:.1f}ms")
        print(f"Min time: {min_inference:.1f}ms")
        print(f"Max time: {max_inference:.1f}ms")
        print(f"95th percentile: {p95_inference:.1f}ms")
        print(f"Std deviation: {std_inference:.1f}ms")
        print(f"Inference rate: {inference_rate:.1f} Hz")
        
        # Performance assessment
        if avg_inference < 50:
            inference_performance = "🟢 EXCELLENT - Real-time capable"
        elif avg_inference < 100:
            inference_performance = "🟡 GOOD - Acceptable for most use cases"
        elif avg_inference < 200:
            inference_performance = "🟠 FAIR - May cause some delays"
        else:
            inference_performance = "🔴 SLOW - Significant delays expected"
        
        print(f"Inference performance: {inference_performance}")
        print()
    
    # Overall assessment
    if action_queue_size and inference_times:
        print("🎯 OVERALL ASSESSMENT")
        print("-" * 40)
        if "EXCELLENT" in queue_performance and "EXCELLENT" in inference_performance:
            overall = "🟢 EXCELLENT - Optimal performance"
        elif "GOOD" in queue_performance or "GOOD" in inference_performance:
            overall = "🟡 GOOD - Solid performance"
        elif "FAIR" in queue_performance or "FAIR" in inference_performance:
            overall = "🟠 FAIR - Acceptable but could be improved"
        else:
            overall = "🔴 NEEDS IMPROVEMENT - Performance issues detected"
        print(f"Overall: {overall}")
    
    print("="*80)
    
    # Try to generate plots if display is available
    try:
        import matplotlib
        matplotlib.use('Agg')  # Use non-interactive backend
        import matplotlib.pyplot as plt
        
        # Determine number of subplots needed
        num_plots = 0
        if action_queue_size:
            num_plots += 1
        if inference_times:
            num_plots += 1
        
        if num_plots > 0:
            fig, axes = plt.subplots(num_plots, 1, figsize=(12, 4 * num_plots))
            if num_plots == 1:
                axes = [axes]  # Make it consistent for single plot
            
            plot_idx = 0
            
            # Action queue size plot
            if action_queue_size:
                ax = axes[plot_idx]
                ax.set_title("Action Queue Size Over Time", fontsize=14, fontweight='bold')
                ax.set_xlabel("Environment steps")
                ax.set_ylabel("Queue Size")
                ax.set_ylim(0, max(queue_sizes) * 1.1 if max(queue_sizes) > 0 else 1)
                ax.grid(True, alpha=0.3)
                ax.plot(range(len(queue_sizes)), queue_sizes, linewidth=2, color='blue', label='Queue Size')
                ax.legend()
                plot_idx += 1
            
            # Inference time plot
            if inference_times:
                ax = axes[plot_idx]
                ax.set_title("Inference Time Over Time", fontsize=14, fontweight='bold')
                ax.set_xlabel("Inference number")
                ax.set_ylabel("Inference Time (ms)")
                ax.grid(True, alpha=0.3)
                
                # Plot inference times with moving average
                ax.plot(range(len(inference_ms)), inference_ms, linewidth=1, alpha=0.7, color='red', label='Inference Time')
                
                # Add moving average if we have enough data points
                if len(inference_ms) >= 10:
                    window_size = min(10, len(inference_ms) // 4)
                    moving_avg = np.convolve(inference_ms, np.ones(window_size)/window_size, mode='same')
                    ax.plot(range(len(moving_avg)), moving_avg, linewidth=3, color='darkred', label=f'Moving Average ({window_size})')
                
                # Add horizontal line for average
                ax.axhline(y=avg_inference, color='orange', linestyle='--', alpha=0.8, label=f'Average ({avg_inference:.1f}ms)')
                ax.legend()
            
            plt.tight_layout()
            
            # Save plot to file instead of showing
            timestamp = int(time.time())
            plot_filename = f"performance_stats_{timestamp}.png"
            plt.savefig(plot_filename, dpi=150, bbox_inches='tight')
            plt.close()
            print(f"📈 Performance plots saved to: {plot_filename}")
        
    except Exception as e:
        print(f"Note: Could not generate plots ({e})")
    
    print()


def validate_robot_cameras_for_policy(
    lerobot_observation_features: dict[str, dict], policy_image_features: dict[str, PolicyFeature]
) -> None:
    image_keys = list(filter(is_image_key, lerobot_observation_features))
    assert set(image_keys) == set(policy_image_features.keys()), (
        f"Policy image features must match robot cameras! Received {list(policy_image_features.keys())} != {image_keys}"
    )


def map_robot_keys_to_lerobot_features(robot: Robot) -> dict[str, dict]:
    return hw_to_dataset_features(robot.observation_features, "observation", use_video=False)


def is_image_key(k: str) -> bool:
    return k.startswith(OBS_IMAGES)


def resize_robot_observation_image(image: torch.tensor, resize_dims: tuple[int, int, int]) -> torch.tensor:
    assert image.ndim == 3, f"Image must be (C, H, W)! Received {image.shape}"
    # (H, W, C) -> (C, H, W) for resizing from robot obsevation resolution to policy image resolution
    image = image.permute(2, 0, 1)
    dims = (resize_dims[1], resize_dims[2])
    # Add batch dimension for interpolate: (C, H, W) -> (1, C, H, W)
    image_batched = image.unsqueeze(0)
    # Interpolate and remove batch dimension: (1, C, H, W) -> (C, H, W)
    resized = torch.nn.functional.interpolate(image_batched, size=dims, mode="bilinear", align_corners=False)

    return resized.squeeze(0)


def raw_observation_to_observation(
    raw_observation: RawObservation,
    lerobot_features: dict[str, dict],
    policy_image_features: dict[str, PolicyFeature],
    device: str,
) -> Observation:
    observation = {}

    observation = prepare_raw_observation(raw_observation, lerobot_features, policy_image_features)
    for k, v in observation.items():
        if isinstance(v, torch.Tensor):  # VLAs present natural-language instructions in observations
            if "image" in k:
                # Policy expects images in shape (B, C, H, W)
                observation[k] = prepare_image(v).unsqueeze(0).to(device)
            else:
                observation[k] = v.to(device)
        else:
            observation[k] = v

    return observation


def prepare_image(image: torch.Tensor) -> torch.Tensor:
    """Minimal preprocessing to turn int8 images to float32 in [0, 1], and create a memory-contiguous tensor"""
    image = image.type(torch.float32) / 255
    image = image.contiguous()

    return image


def extract_state_from_raw_observation(
    lerobot_obs: RawObservation,
) -> torch.Tensor:
    """Extract the state from a raw observation."""
    state = torch.tensor(lerobot_obs[OBS_STATE])

    if state.ndim == 1:
        state = state.unsqueeze(0)

    return state


def extract_images_from_raw_observation(
    lerobot_obs: RawObservation,
    camera_key: str,
) -> dict[str, torch.Tensor]:
    """Extract the images from a raw observation."""
    return torch.tensor(lerobot_obs[camera_key])


def make_lerobot_observation(
    robot_obs: RawObservation,
    lerobot_features: dict[str, dict],
) -> LeRobotObservation:
    """Make a lerobot observation from a raw observation."""
    return build_dataset_frame(lerobot_features, robot_obs, prefix="observation")


def prepare_raw_observation(
    robot_obs: RawObservation,
    lerobot_features: dict[str, dict],
    policy_image_features: dict[str, PolicyFeature],
) -> Observation:
    """Matches keys from the raw robot_obs dict to the keys expected by a given policy (passed as
    policy_image_features)."""
    # 1. {motor.pos1:value1, motor.pos2:value2, ..., laptop:np.ndarray} ->
    # -> {observation.state:[value1,value2,...], observation.images.laptop:np.ndarray}
    lerobot_obs = make_lerobot_observation(robot_obs, lerobot_features)

    # 2. Greps all observation.images.<> keys
    image_keys = list(filter(is_image_key, lerobot_obs))
    # state's shape is expected as (B, state_dim)
    state_dict = {OBS_STATE: extract_state_from_raw_observation(lerobot_obs)}
    image_dict = {
        image_k: extract_images_from_raw_observation(lerobot_obs, image_k) for image_k in image_keys
    }

    # Turns the image features to (C, H, W) with H, W matching the policy image features.
    # This reduces the resolution of the images
    image_dict = {
        key: resize_robot_observation_image(torch.tensor(lerobot_obs[key]), policy_image_features[key].shape)
        for key in image_keys
    }

    if "task" in robot_obs:
        state_dict["task"] = robot_obs["task"]

    return {**state_dict, **image_dict}


def get_logger(name: str, log_to_file: bool = True) -> logging.Logger:
    """
    Get a logger using the standardized logging setup from utils.py.

    Args:
        name: Logger name (e.g., 'policy_server', 'robot_client')
        log_to_file: Whether to also log to a file

    Returns:
        Configured logger instance
    """
    # Create logs directory if logging to file
    if log_to_file:
        os.makedirs("logs", exist_ok=True)
        log_file = Path(f"logs/{name}_{int(time.time())}.log")
    else:
        log_file = None

    # Initialize the standardized logging
    init_logging(log_file=log_file, display_pid=False)

    # Return a named logger
    return logging.getLogger(name)


@dataclass
class TimedData:
    """A data object with timestamp and timestep information.

    Args:
        timestamp: Unix timestamp relative to data's creation.
        data: The actual data to wrap a timestamp around.
        timestep: The timestep of the data.
    """

    timestamp: float
    timestep: int

    def get_timestamp(self):
        return self.timestamp

    def get_timestep(self):
        return self.timestep


@dataclass
class TimedAction(TimedData):
    action: Action

    def get_action(self):
        return self.action


@dataclass
class TimedObservation(TimedData):
    observation: RawObservation
    must_go: bool = False

    def get_observation(self):
        return self.observation


@dataclass
class FPSTracker:
    """Utility class to track FPS metrics over time."""

    target_fps: float
    first_timestamp: float = None
    total_obs_count: int = 0

    def calculate_fps_metrics(self, current_timestamp: float) -> dict[str, float]:
        """Calculate average FPS vs target"""
        self.total_obs_count += 1

        # Initialize first observation time
        if self.first_timestamp is None:
            self.first_timestamp = current_timestamp

        # Calculate overall average FPS (since start)
        total_duration = current_timestamp - self.first_timestamp
        avg_fps = (self.total_obs_count - 1) / total_duration if total_duration > 1e-6 else 0.0

        return {"avg_fps": avg_fps, "target_fps": self.target_fps}

    def reset(self):
        """Reset the FPS tracker state"""
        self.first_timestamp = None
        self.total_obs_count = 0


@dataclass
class RemotePolicyConfig:
    policy_type: str
    pretrained_name_or_path: str
    lerobot_features: dict[str, PolicyFeature]
    actions_per_chunk: int
    device: str = "cpu"


def _compare_observation_states(obs1_state: torch.Tensor, obs2_state: torch.Tensor, atol: float) -> bool:
    """Check if two observation states are similar, under a tolerance threshold"""
    return bool(torch.linalg.norm(obs1_state - obs2_state) < atol)


def observations_similar(
    obs1: TimedObservation, obs2: TimedObservation, lerobot_features: dict[str, dict], atol: float = 1
) -> bool:
    """Check if two observations are similar, under a tolerance threshold. Measures distance between
    observations as the difference in joint-space between the two observations.

    NOTE(fracapuano): This is a very simple check, and it is enough for the current use case.
    An immediate next step is to use (fast) perceptual difference metrics comparing some camera views,
    to surpass this joint-space similarity check.
    """
    obs1_state = extract_state_from_raw_observation(
        make_lerobot_observation(obs1.get_observation(), lerobot_features)
    )
    obs2_state = extract_state_from_raw_observation(
        make_lerobot_observation(obs2.get_observation(), lerobot_features)
    )

    return _compare_observation_states(obs1_state, obs2_state, atol=atol)
