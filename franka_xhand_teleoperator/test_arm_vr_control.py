#!/usr/bin/env python3
"""
Test VR control of Franka arm using real robot and VR device
"""

import sys
import time
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "src"))
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import vr_message_router
from lerobot.teleoperators.franka_fer_vr.arm_ik_processor import ArmIKProcessor
from lerobot.robots.franka_fer.franka_fer import FrankaFER
from lerobot.robots.franka_fer.franka_fer_config import FrankaFERConfig

from src.adb_setup import setup_adb_reverse, cleanup_adb_reverse

def test_vr_arm_control():
    print("Testing VR control of Franka arm...")
    
    # Setup ADB for VR connection
    print("Setting up ADB reverse port forwarding...")
    setup_adb_reverse(tcp_port=8000)
    
    try:
        # Initialize VR message router
        config = vr_message_router.VRRouterConfig()
        config.tcp_port = 8000
        config.verbose = False  # Reduce verbosity
        config.message_timeout_ms = 200.0  # Increase timeout
        
        router = vr_message_router.VRMessageRouter(config)
        print("VR message router created")
        
        # Initialize arm IK processor
        ik_config = {
            'verbose': True,  # Enable for debugging translation issues
            'smoothing_factor': 0.1  # Reduced from 0.8 for more responsive movement
        }
        
        arm_processor = ArmIKProcessor(ik_config)
        print("Arm IK processor created")
        
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
        
        # Get current robot state
        current_obs = robot.get_observation()
        current_joints = [current_obs[f"joint_{i}.pos"] for i in range(7)]
        
        print(f"Current robot joints: {[f'{j:.3f}' for j in current_joints]}")
        
        # Setup arm IK processor with proper neutral pose from config
        neutral_pose = robot_config.home_position
        
        # Get initial robot pose from current end-effector position
        # FrankaFER stores ee_pose as ee_pose.00 through ee_pose.15
        initial_robot_pose = [current_obs[f"ee_pose.{i:02d}"] for i in range(16)]
        
        print(f"Initial end-effector pose: {[f'{p:.3f}' for p in initial_robot_pose[:4]]} ...")  # Show first row
        
        success = arm_processor.setup(
            neutral_pose=neutral_pose,
            initial_robot_pose=initial_robot_pose,
            manipulability_weight=1.0,
            neutral_distance_weight=1.5,
            current_distance_weight=3.0,
            joint_weights=[3.0, 3.0, 2.0, 2.0, 1.0, 1.0, 1.0],
            q7_min=-2.8,
            q7_max=2.8
        )
        
        if not success:
            print("Failed to setup arm IK processor")
            return False
        
        print("Arm IK processor setup complete")
        
        # Start VR TCP server
        if not router.start_tcp_server():
            print("Failed to start VR TCP server")
            return False
        
        print("VR TCP server started")
        print("Connect your VR device and move your hand to control the robot")
        print("Press Ctrl+C to stop")
        
        # Control loop
        last_update_time = time.time()
        update_rate = 100.0  # 100 Hz control rate (much faster than 30Hz VR)
        
        try:
            while True:
                current_time = time.time()
                
                # Control at fixed rate
                if current_time - last_update_time >= 1.0 / update_rate:
                    # Get VR messages
                    messages = router.get_messages()
                    status = router.get_status()
                    
                    if status['tcp_connected'] and messages.wrist_valid:
                        # Get current robot state
                        current_obs = robot.get_observation()
                        current_joints = [current_obs[f"joint_{i}.pos"] for i in range(7)]
                        
                        # Process VR wrist data through IK
                        arm_action = arm_processor.process_wrist_data(
                            messages.wrist_data, 
                            current_joints
                        )
                        
                        # Convert action format from arm_joint_X.pos to joint_X.pos for FrankaFER
                        franka_action = {}
                        for i in range(7):
                            franka_action[f"joint_{i}.pos"] = arm_action[f"arm_joint_{i}.pos"]
                        
                        # Send action to robot
                        robot.send_action(franka_action)
                        
                        # Print status
                        wrist = messages.wrist_data
                        target_joints = [arm_action[f"arm_joint_{i}.pos"] for i in range(7)]
                        max_diff = max(abs(t - c) for t, c in zip(target_joints, current_joints))
                        
                        print(f"VR pos: ({wrist.position[0]:.3f}, {wrist.position[1]:.3f}, {wrist.position[2]:.3f}) "
                              f"| Max joint diff: {max_diff:.3f} rad ({np.degrees(max_diff):.1f}°)")
                        
                        # Store the first VR position to track if it changes
                        if not hasattr(test_vr_arm_control, 'first_vr_pos'):
                            test_vr_arm_control.first_vr_pos = wrist.position.copy()
                            print(f"  STORED first VR position: {test_vr_arm_control.first_vr_pos}")
                    
                    elif status['tcp_connected']:
                        print("VR connected but no valid wrist data")
                    else:
                        print("Waiting for VR connection...")
                    
                    last_update_time = current_time
                
                time.sleep(0.01)  # Small sleep to prevent CPU spinning
                
        except KeyboardInterrupt:
            print("\nStopping VR control...")
    
    finally:
        # Cleanup
        try:
            router.stop()
            print("VR router stopped")
        except:
            pass
        
        try:
            robot.disconnect()
            print("Robot disconnected")
        except:
            pass
        
        print("Cleaning up ADB...")
        cleanup_adb_reverse(tcp_port=8000)
    
    print("VR arm control test completed")
    return True

if __name__ == "__main__":
    success = test_vr_arm_control()
    sys.exit(0 if success else 1)