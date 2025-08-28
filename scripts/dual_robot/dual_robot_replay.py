#!/usr/bin/env python3
"""
Replay script for dual VR recorded data with Franka FER + XHand.

This script loads a recorded dataset and replays the actions on the robot,
allowing you to see the recorded manipulation behaviors.
"""

import logging
import time
import sys
import argparse
from pathlib import Path
import torch

from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.robots.franka_fer_xhand.franka_fer_xhand import FrankaFERXHand
from lerobot.robots.franka_fer_xhand.franka_fer_xhand_config import FrankaFERXHandConfig
from lerobot.robots.franka_fer.franka_fer_config import FrankaFERConfig
from lerobot.robots.xhand.xhand_config import XHandConfig
from lerobot.utils.utils import init_logging, log_say
from lerobot.utils.visualization_utils import _init_rerun
from lerobot.utils.control_utils import init_keyboard_listener

# Default replay parameters
DEFAULT_ROBOT_IP = "192.168.18.1"
DEFAULT_EPISODE = 0
DEFAULT_REPLAY_SPEED = 1.0
DEFAULT_CONFIRM = True

# Initialize logging
init_logging()
logger = logging.getLogger(__name__)

def load_dataset(dataset_path: str) -> LeRobotDataset:
    """Load the recorded dataset."""
    dataset_path = Path(dataset_path)

    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset not found. Looking for: {dataset_path}")
    
    logger.info(f"Loading dataset from: {dataset_path}")
    dataset = LeRobotDataset(str(dataset_path))
    return dataset

def get_episode_actions(dataset: LeRobotDataset, episode_idx: int) -> list:
    """Extract actions from a specific episode."""
    # Check available attributes
    logger.info(f"Dataset attributes: {dir(dataset)}")
    
    # Try different ways to access episode data
    if hasattr(dataset, 'episode_data_index'):
        logger.info(f"Episode data index keys: {dataset.episode_data_index.keys()}")
    
    # Get episode bounds - try different methods
    if hasattr(dataset, 'episode_data_index') and 'from' in dataset.episode_data_index:
        # Method 1: Direct episode_data_index access
        start_idx = dataset.episode_data_index['from'][episode_idx]
        end_idx = dataset.episode_data_index['to'][episode_idx]
    elif hasattr(dataset, 'get_episode_data_index'):
        # Method 2: get_episode_data_index method
        episode_data = dataset.get_episode_data_index()
        start_idx = episode_data['from'][episode_idx]
        end_idx = episode_data['to'][episode_idx]
    else:
        # Method 3: Assume single episode covers entire dataset
        logger.warning("Cannot find episode boundaries, assuming single episode")
        start_idx = 0
        end_idx = len(dataset)
    
    logger.info(f"Episode {episode_idx}: frames {start_idx} to {end_idx} ({end_idx - start_idx} frames)")
    
    # Extract actions for this episode
    actions = []
    for frame_idx in range(start_idx, end_idx):
        frame_data = dataset[frame_idx]
        action = {}
        
        # Extract action data - the actions are stored under the 'action' key
        if 'action' in frame_data:
            action_data = frame_data['action']
            
            # Debug: Show first frame action structure
            if frame_idx == start_idx:
                logger.info(f"First frame keys: {list(frame_data.keys())}")
                logger.info(f"Action data type: {type(action_data)}")
                if hasattr(action_data, 'shape'):
                    logger.info(f"Action shape: {action_data.shape}")
                
            # Handle tensor actions (convert to dict format expected by robot)
            if isinstance(action_data, torch.Tensor):
                # The actions should correspond to the robot's action features
                # We need to map tensor indices to action keys
                action_features = ['arm_joint_0.pos', 'arm_joint_1.pos', 'arm_joint_2.pos', 'arm_joint_3.pos', 
                                 'arm_joint_4.pos', 'arm_joint_5.pos', 'arm_joint_6.pos',
                                 'hand_joint_0.pos', 'hand_joint_1.pos', 'hand_joint_2.pos', 'hand_joint_3.pos',
                                 'hand_joint_4.pos', 'hand_joint_5.pos', 'hand_joint_6.pos', 'hand_joint_7.pos',
                                 'hand_joint_8.pos', 'hand_joint_9.pos', 'hand_joint_10.pos', 'hand_joint_11.pos']
                
                for i, action_key in enumerate(action_features):
                    if i < len(action_data):
                        action[action_key] = action_data[i].item()
            elif isinstance(action_data, dict):
                # Action is already a dictionary
                action.update(action_data)
        
        actions.append(action)
    
    return actions

def setup_robot(robot_ip: str) -> FrankaFERXHand:
    """Set up the composite robot for replay."""
    # Create arm configuration
    arm_config = FrankaFERConfig(
        server_ip=robot_ip,
        server_port=5000,
        home_position=[0, -0.785, 0, -2.356, 0, 1.571, -0.9],
        max_relative_target=None,
        cameras={}
    )
    
    # Create hand configuration
    hand_config = XHandConfig(
        protocol="RS485",
        serial_port="/dev/ttyUSB0",
        baud_rate=3000000,
        hand_id=0,
        control_frequency=200.0,
        max_torque=250.0,
        cameras={}
    )
    
    # Create composite robot configuration
    robot_config = FrankaFERXHandConfig(
        arm_config=arm_config,
        hand_config=hand_config,
        cameras={},
        synchronize_actions=True,
        action_timeout=0.1,
        check_arm_hand_collision=True,
        emergency_stop_both=True
    )
    
    # Create and connect robot
    robot = FrankaFERXHand(robot_config)
    return robot

def replay_episode(robot: FrankaFERXHand, actions: list, fps: float, speed_multiplier: float = 1.0):
    """Replay a sequence of actions on the robot."""
    dt = 1.0 / fps / speed_multiplier
    
    logger.info(f"Replaying {len(actions)} actions at {fps * speed_multiplier:.1f} FPS")
    log_say(f"Starting replay in 3 seconds...")
    time.sleep(3)
    
    log_say("Replay starting now!")
    
    start_time = time.perf_counter()
    
    for i, action in enumerate(actions):
        loop_start = time.perf_counter()
        
        try:
            # Send action to robot
            performed_action = robot.send_action(action)
            
            # Log progress periodically
            if i % 30 == 0:  # Every ~1 second at 30fps
                elapsed = time.perf_counter() - start_time
                progress = (i / len(actions)) * 100
                logger.info(f"Replay progress: {progress:.1f}% ({i}/{len(actions)}) - {elapsed:.1f}s elapsed")
        
        except Exception as e:
            logger.error(f"Error sending action {i}: {e}")
            logger.warning("Continuing with next action...")
            continue
        
        # Maintain timing
        elapsed = time.perf_counter() - loop_start
        sleep_time = dt - elapsed
        if sleep_time > 0:
            time.sleep(sleep_time)
        elif sleep_time < -0.01:  # More than 10ms behind
            logger.warning(f"Replay running behind schedule by {-sleep_time*1000:.1f}ms")
    
    total_time = time.perf_counter() - start_time
    log_say(f"Replay complete! Total time: {total_time:.1f}s")

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Replay recorded dual robot dataset")
    parser.add_argument("dataset_path", help="Path to the dataset directory")
    parser.add_argument("--episode", type=int, default=DEFAULT_EPISODE, 
                       help=f"Episode to replay (default: {DEFAULT_EPISODE})")
    parser.add_argument("--speed", type=float, default=DEFAULT_REPLAY_SPEED,
                       help=f"Replay speed multiplier (default: {DEFAULT_REPLAY_SPEED})")
    parser.add_argument("--robot-ip", default=DEFAULT_ROBOT_IP,
                       help=f"Robot IP address (default: {DEFAULT_ROBOT_IP})")
    parser.add_argument("--no-confirm", action="store_true",
                       help="Skip confirmation before starting replay")
    return parser.parse_args()

def main():
    """Main replay function."""
    args = parse_args()
    
    logger.info("Setting up dual robot replay...")
    logger.info(f"Dataset path: {args.dataset_path}")
    logger.info(f"Episode: {args.episode}")
    logger.info(f"Speed: {args.speed}")
    logger.info(f"Robot IP: {args.robot_ip}")
    
    try:
        # Load dataset
        logger.info(f"Loading dataset from: {args.dataset_path}")
        dataset = load_dataset(args.dataset_path)
        
        logger.info(f"Dataset loaded. Episodes: {dataset.num_episodes}, Total frames: {len(dataset)}")
        logger.info(f"FPS: {dataset.fps}")
        
        # Check episode exists
        if args.episode >= dataset.num_episodes:
            raise ValueError(f"Episode {args.episode} not found. Dataset has {dataset.num_episodes} episodes.")
        
        # Extract actions from episode
        logger.info(f"Extracting actions from episode {args.episode}")
        actions = get_episode_actions(dataset, args.episode)
        
        if not actions:
            raise ValueError(f"No actions found in episode {args.episode}")
        
        logger.info(f"Extracted {len(actions)} actions")
        logger.info(f"Action keys: {list(actions[0].keys()) if actions else 'None'}")
        
        # Set up robot
        logger.info("Setting up robot...")
        robot = setup_robot(args.robot_ip)
        
        # Connect robot
        logger.info("Connecting to robot...")
        robot.connect(calibrate=False)
        
        if not robot.is_connected:
            raise ValueError("Failed to connect to robot")
        
        logger.info("Robot connected successfully")
        
        # Initialize visualization
        _init_rerun(session_name="dual_robot_replay")
        
        # Set up keyboard listener for emergency stop
        listener, events = init_keyboard_listener()
        
        # Confirmation before starting
        if not args.no_confirm:
            log_say("Ready to replay. Starting in 5 seconds (Ctrl+C to cancel)...")
            time.sleep(5)
        
        # Replay the episode
        replay_episode(robot, actions, dataset.fps, args.speed)
        
    except KeyboardInterrupt:
        logger.info("Replay cancelled by user")
    except Exception as e:
        logger.error(f"Error during replay: {e}")
        raise
    finally:
        # Clean up
        logger.info("Cleaning up...")
        
        try:
            if 'robot' in locals() and robot.is_connected:
                robot.disconnect()
            logger.info("Robot disconnected")
        except Exception as e:
            logger.error(f"Error disconnecting robot: {e}")
        
        try:
            if 'listener' in locals():
                listener.stop()
        except Exception as e:
            logger.error(f"Error stopping keyboard listener: {e}")
        
        logger.info("Replay session complete!")

if __name__ == "__main__":
    main()