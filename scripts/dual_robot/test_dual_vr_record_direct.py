#!/usr/bin/env python3
"""
Direct Python script for recording data with Franka FER + XHand using dual VR teleoperator.

This script bypasses the CLI argument parsing issues and directly instantiates 
the robot and teleoperator configurations.
"""

import logging
import time
from pathlib import Path

from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.datasets.utils import hw_to_dataset_features
from lerobot.record import record_loop
from lerobot.robots.franka_fer_xhand.franka_fer_xhand import FrankaFERXHand
from lerobot.robots.franka_fer_xhand.franka_fer_xhand_config import FrankaFERXHandConfig
from lerobot.robots.franka_fer.franka_fer_config import FrankaFERConfig
from lerobot.robots.xhand.xhand_config import XHandConfig
from lerobot.teleoperators.franka_fer_xhand_vr.franka_fer_xhand_vr_teleoperator import FrankaFERXHandVRTeleoperator
from lerobot.teleoperators.franka_fer_xhand_vr.config_franka_fer_xhand_vr import FrankaFERXHandVRTeleoperatorConfig
from lerobot.utils.control_utils import init_keyboard_listener
from lerobot.utils.utils import init_logging, log_say
from lerobot.utils.visualization_utils import _init_rerun

# Recording parameters
NUM_EPISODES = 1
FPS = 30
EPISODE_TIME_SEC = 10
RESET_TIME_SEC = 5
TASK_DESCRIPTION = "Teleoperate dual arm-hand system to manipulate objects"
DATASET_NAME = "test_dual_vr_recording_5"
ROBOT_IP = "192.168.18.1"  # Update with your robot IP

# Initialize logging
init_logging()
logger = logging.getLogger(__name__)

def main():
    """Main function to run the recording test."""
    
    logger.info("Setting up dual VR recording test...")
    
    # Create arm configuration
    arm_config = FrankaFERConfig(
        server_ip=ROBOT_IP,
        server_port=5000,
        home_position=[0, -0.785, 0, -2.356, 0, 1.571, -0.9],  # Modified for XHand
        max_relative_target=None,
        cameras={}  # Use composite robot cameras instead
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
        cameras={},  # No cameras for this test
        synchronize_actions=True,
        action_timeout=0.1,
        check_arm_hand_collision=True,
        emergency_stop_both=True
    )
    
    # Create the composite robot
    robot = FrankaFERXHand(robot_config)
    
    # Create teleoperator configuration
    teleop_config = FrankaFERXHandVRTeleoperatorConfig(
        vr_tcp_port=8000,
        setup_adb=True,
        vr_verbose=False,
        # Arm settings
        arm_smoothing_factor=0.1,
        manipulability_weight=0.1,
        neutral_distance_weight=1.0,
        current_distance_weight=10.0,
        arm_joint_weights=[1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
        q7_min=-2.8973,
        q7_max=2.8973,
        # Hand settings
        hand_robot_name="xhand_right",
        hand_retargeting_type="dexpilot",
        hand_type="right",
        hand_control_frequency=30.0,
        hand_smoothing_alpha=0.3
    )
    
    # Create the teleoperator
    teleop = FrankaFERXHandVRTeleoperator(teleop_config)
    
    # Configure dataset features
    logger.info("Setting up dataset...")
    action_features = hw_to_dataset_features(robot.action_features, "action")
    obs_features = hw_to_dataset_features(robot.observation_features, "observation")
    dataset_features = {**action_features, **obs_features}
    
    logger.info(f"Robot action features: {list(robot.action_features.keys())}")
    logger.info(f"Robot observation features: {list(robot.observation_features.keys())}")
    
    # Create the dataset (local for testing)
    dataset = LeRobotDataset.create(
        repo_id=f"local/{DATASET_NAME}",
        fps=FPS,
        features=dataset_features,
        robot_type=robot.name,
        use_videos=False,  # Disable videos for initial testing
        image_writer_threads=4,
    )
    
    try:
        # Connect robot and teleoperator
        logger.info("Connecting robot...")
        robot.connect(calibrate=False)
        
        logger.info("Connecting teleoperator...")
        teleop.connect(calibrate=False)
        
        # Set robot reference in teleoperator
        teleop.set_robot(robot)
        
        # Initialize visualization
        _init_rerun(session_name="dual_vr_record_test")
        
        # Initialize keyboard listener for control events
        listener, events = init_keyboard_listener()
        
        # Verify connections
        if not robot.is_connected:
            raise ValueError("Robot is not connected!")
        if not teleop.is_connected:
            raise ValueError("Teleoperator is not connected!")
        
        logger.info("Starting recording session...")
        log_say(f"Recording {NUM_EPISODES} episodes")
        log_say("Press 's' to stop recording, 'r' to re-record current episode")
        
        # Recording loop
        recorded_episodes = 0
        while recorded_episodes < NUM_EPISODES and not events.get("stop_recording", False):
            log_say(f"Recording episode {recorded_episodes + 1} of {NUM_EPISODES}")
            
            # Record the episode
            record_loop(
                robot=robot,
                events=events,
                fps=FPS,
                dataset=dataset,
                teleop=teleop,
                control_time_s=EPISODE_TIME_SEC,
                single_task=TASK_DESCRIPTION,
                display_data=True,
            )
            
            # Reset environment between episodes (except after last one)
            if not events.get("stop_recording", False) and recorded_episodes < NUM_EPISODES - 1:
                log_say("Reset the environment")
                record_loop(
                    robot=robot,
                    events=events,
                    fps=FPS,
                    teleop=teleop,
                    control_time_s=RESET_TIME_SEC,
                    single_task=TASK_DESCRIPTION,
                    display_data=True,
                )
            
            # Handle re-recording if requested
            if events.get("rerecord_episode", False):
                log_say("Re-recording episode")
                events["rerecord_episode"] = False
                events["exit_early"] = False
                dataset.clear_episode_buffer()
                continue
            
            # Save the episode
            dataset.save_episode()
            recorded_episodes += 1
            logger.info(f"Episode {recorded_episodes} saved successfully")
            
    except KeyboardInterrupt:
        logger.info("Recording interrupted by user")
    except Exception as e:
        logger.error(f"Error during recording: {e}")
        raise
    finally:
        # Clean up
        logger.info("Cleaning up...")
        
        # Save dataset locally
        logger.info(f"Dataset saved locally at: {dataset.root}")
        
        # Disconnect devices
        try:
            if teleop.is_connected:
                teleop.disconnect()
        except Exception as e:
            logger.error(f"Error disconnecting teleoperator: {e}")
            
        try:
            if robot.is_connected:
                robot.disconnect()
        except Exception as e:
            logger.error(f"Error disconnecting robot: {e}")
        
        # Stop keyboard listener
        try:
            listener.stop()
        except Exception as e:
            logger.error(f"Error stopping keyboard listener: {e}")
        
        logger.info("Recording session complete!")
        logger.info(f"Recorded {recorded_episodes} episodes")
        logger.info(f"Dataset saved to: {dataset.root}")


if __name__ == "__main__":
    main()