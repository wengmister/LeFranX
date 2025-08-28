#!/usr/bin/env python3
"""
Test VR control of Franka arm using real robot and VR device via teleoperator class
"""

import sys
import time
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from lerobot.robots.franka_fer.franka_fer import FrankaFER
from lerobot.robots.franka_fer.franka_fer_config import FrankaFERConfig
from lerobot.teleoperators.franka_fer_vr.franka_fer_vr_teleoperator import FrankaFERVRTeleoperator
from lerobot.teleoperators.franka_fer_vr.config_franka_fer_vr import FrankaFERVRTeleoperatorConfig

def test_vr_arm_control():
    print("Testing VR control of Franka arm using FrankaFERVRTeleoperator...")
    
    try:
        # Initialize Franka robot with proper config
        print("Connecting to Franka robot...")
        robot_config = FrankaFERConfig()
        robot = FrankaFER(robot_config)
        robot.connect()
        print("Connected to Franka robot")
        
        # Home robot to neutral position
        print("Homing robot to neutral position...")
        home_action = {f"joint_{i}.pos": robot_config.home_position[i] for i in range(7)}
        robot.send_action(home_action)
        time.sleep(2.0)  # Wait for robot to reach home position
        print("Robot homed")
        
        # Initialize FrankaFER VR teleoperator with config
        print("Setting up FrankaFER VR teleoperator...")
        teleop_config = FrankaFERVRTeleoperatorConfig(
            tcp_port=8000,
            setup_adb=True,  # Automatic ADB setup
            smoothing_factor=0.1,  # Reduced for more responsive movement
            verbose=False,  # Suppress IK solver logging
            manipulability_weight=0.1,
            neutral_distance_weight=1.5,
            current_distance_weight=3.0,
            joint_weights=[3.0, 3.0, 1.0, 1.0, 1.0, 1.0, 1.0],
            q7_min=-2.89,  # Use full Franka range
            q7_max=2.89
        )
        
        teleop = FrankaFERVRTeleoperator(teleop_config)
        
        # Connect teleoperator (this handles ADB setup automatically)
        print("Connecting teleoperator (includes ADB setup)...")
        teleop.connect(calibrate=False)
        print("Teleoperator connected")
        
        # Set robot reference for IK processing
        print("Setting robot reference...")
        teleop.set_robot(robot)
        print("Robot reference set")
        
        # Debug: Check what observation keys are available
        debug_obs = robot.get_observation()
        print(f"Available observation keys: {list(debug_obs.keys())}")
        print(f"Sample values: {[(k, v) for k, v in list(debug_obs.items())[:5]]}")
        
        print("Connect your VR device and move your hand to control the robot")
        print("Press Ctrl+C to stop")
        
        # Control loop using teleoperator
        last_update_time = time.time()
        update_rate = 100.0  # 100 Hz control rate
        frame_count = 0
        
        try:
            while True:
                current_time = time.time()
                
                # Control at fixed rate
                if current_time - last_update_time >= 1.0 / update_rate:
                    # Get action from teleoperator
                    action = teleop.get_action()
                    
                    if action is not None and robot.is_connected:
                        # Send action to robot
                        try:
                            robot.send_action(action)
                        except Exception as e:
                            print(f"Error sending action to robot: {e}")
                            break
                        
                        # Get current robot state for logging
                        try:
                            current_obs = robot.get_observation()
                            current_joints = [current_obs[f"joint_{i}.pos"] for i in range(7)]
                            target_joints = [action[f"joint_{i}.pos"] for i in range(7)]
                            max_diff = max(abs(t - c) for t, c in zip(target_joints, current_joints))
                            
                            # Print status every 30 frames (3 times per second at 100Hz)
                            if frame_count % 30 == 0:
                                print(f"Frame {frame_count}: Max joint diff: {max_diff:.3f} rad ({np.degrees(max_diff):.1f}°)")
                        except Exception as e:
                            print(f"Error getting robot observation: {e}")
                            break
                    else:
                        # Print waiting message every 100 frames (once per second at 100Hz)
                        if frame_count % 100 == 0:
                            print("Waiting for VR connection or valid data...")
                    
                    last_update_time = current_time
                    frame_count += 1
                
                time.sleep(0.01)  # Small sleep to prevent CPU spinning
                
        except KeyboardInterrupt:
            print("\nStopping VR control...")
    
    finally:
        # Cleanup
        try:
            if 'teleop' in locals() and teleop.is_connected:
                teleop.disconnect()
                print("Teleoperator disconnected (includes ADB cleanup)")
        except Exception as e:
            print(f"Error disconnecting teleoperator: {e}")
        
        try:
            if 'robot' in locals() and robot.is_connected:
                robot.disconnect()
                print("Robot disconnected")
        except Exception as e:
            print(f"Error disconnecting robot: {e}")
    
    print("VR arm control test completed")
    return True

if __name__ == "__main__":
    success = test_vr_arm_control()
    sys.exit(0 if success else 1)