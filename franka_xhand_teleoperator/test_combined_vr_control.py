#!/usr/bin/env python3
"""
Test combined VR control of Franka arm + XHand using shared VR router
"""

import sys
import time
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "src"))
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from adb_setup import setup_adb_reverse, cleanup_adb_reverse

def test_combined_vr_control():
    print("Testing combined VR control of Franka arm + XHand...")
    
    try:
        # Import components
        import vr_message_router
        from arm_ik_processor import ArmIKProcessor
        from hand_retargeting_processor import HandRetargetingProcessor
        from lerobot.robots.franka_fer.franka_fer import FrankaFER
        from lerobot.robots.franka_fer.franka_fer_config import FrankaFERConfig
        from lerobot.robots.xhand.xhand import XHand
        from lerobot.robots.xhand.xhand_config import XHandConfig
        
        print("Successfully imported all components")
    except ImportError as e:
        print(f"Failed to import: {e}")
        return False
    
    # Setup ADB for VR connection
    print("Setting up ADB reverse port forwarding...")
    setup_adb_reverse(tcp_port=8000)
    
    try:
        # Initialize VR message router (shared for arm + hand)
        config = vr_message_router.VRRouterConfig()
        config.tcp_port = 8000
        config.verbose = False  # Reduce verbosity
        config.message_timeout_ms = 200.0
        
        router = vr_message_router.VRMessageRouter(config)
        print("VR message router created")
        
        # Initialize arm IK processor
        arm_config = {
            'verbose': True,
            'smoothing_factor': 0.1
        }
        arm_processor = ArmIKProcessor(arm_config)
        print("Arm IK processor created")
        
        # Initialize hand retargeting processor
        hand_config = {
            'verbose': True,
            'smoothing_factor': 0.3
        }
        hand_processor = HandRetargetingProcessor(hand_config)
        print("Hand retargeting processor created")
        
        # Initialize Franka robot
        print("Connecting to Franka robot...")
        franka_config = FrankaFERConfig()
        franka_robot = FrankaFER(franka_config)
        franka_robot.connect()
        print("Connected to Franka robot")
        
        # Initialize XHand robot
        print("Connecting to XHand robot...")
        xhand_config = XHandConfig()
        xhand_robot = XHand(xhand_config)
        xhand_robot.connect()
        print("Connected to XHand robot")
        
        # Home both robots
        print("Homing robots to neutral positions...")
        
        # Home Franka
        franka_home_action = {f"joint_{i}.pos": franka_config.home_position[i] for i in range(7)}
        franka_robot.send_action(franka_home_action)
        
        # Home XHand
        xhand_home_action = {f"joint_{i}.pos": 0.0 for i in range(12)}
        xhand_robot.send_action(xhand_home_action)
        
        time.sleep(2.0)
        print("Robots homed")
        
        # Setup arm IK processor
        franka_obs = franka_robot.get_observation()
        current_arm_joints = [franka_obs[f"joint_{i}.pos"] for i in range(7)]
        initial_robot_pose = [franka_obs[f"ee_pose.{i:02d}"] for i in range(16)]
        
        arm_success = arm_processor.setup(
            neutral_pose=franka_config.home_position,
            initial_robot_pose=initial_robot_pose,
            manipulability_weight=1.0,
            neutral_distance_weight=1.5,
            current_distance_weight=3.0,
            joint_weights=[3.0, 3.0, 2.0, 2.0, 1.0, 1.0, 1.0],
            q7_min=-2.8,
            q7_max=2.8
        )
        
        if not arm_success:
            print("Failed to setup arm IK processor")
            return False
        print("Arm IK processor setup complete")
        
        # Setup hand retargeting processor
        robot_dir = str(Path(__file__).parent.parent / "dex_retargeting" / "assets" / "robots")
        config_path = str(Path(__file__).parent.parent / "dex_retargeting" / "src" / "dex_retargeting" / "configs" / "teleop" / "xhand_right_dexpilot.yml")
        
        hand_success = hand_processor.setup(
            robot_dir=robot_dir,
            config_path=config_path,
            hand_type="Right"
        )
        
        if not hand_success:
            print("Failed to setup hand retargeting processor")
            return False
        print("Hand retargeting processor setup complete")
        
        # Start VR TCP server
        if not router.start_tcp_server():
            print("Failed to start VR TCP server")
            return False
        
        print("VR TCP server started")
        print("Connect your VR device and move your hand to control both arm and hand")
        print("Press Ctrl+C to stop")
        
        # Control loop
        last_update_time = time.time()
        update_rate = 60.0  # 60 Hz control rate
        
        try:
            while True:
                current_time = time.time()
                
                # Control at fixed rate
                if current_time - last_update_time >= 1.0 / update_rate:
                    # Get VR messages
                    messages = router.get_messages()
                    status = router.get_status()
                    
                    if status['tcp_connected']:
                        # Get current robot states
                        franka_obs = franka_robot.get_observation()
                        current_arm_joints = [franka_obs[f"joint_{i}.pos"] for i in range(7)]
                        
                        xhand_obs = xhand_robot.get_observation()
                        current_hand_joints = [xhand_obs[f"joint_{i}.pos"] for i in range(12)]
                        
                        # Process arm control (wrist data)
                        if messages.wrist_valid:
                            arm_action = arm_processor.process_wrist_data(
                                messages.wrist_data, 
                                current_arm_joints
                            )
                            
                            # Convert action format for FrankaFER
                            franka_action = {}
                            for i in range(7):
                                franka_action[f"joint_{i}.pos"] = arm_action[f"arm_joint_{i}.pos"]
                            
                            # Send action to Franka
                            franka_robot.send_action(franka_action)
                        
                        # Process hand control (landmarks data)
                        if messages.landmarks_valid:
                            hand_action = hand_processor.process_landmarks_data(
                                messages.landmarks_data, 
                                current_hand_joints
                            )
                            
                            # Convert action format for XHand
                            xhand_action = {}
                            for i in range(12):
                                xhand_action[f"joint_{i}.pos"] = hand_action[f"hand_joint_{i}.pos"]
                            
                            # Send action to XHand
                            xhand_robot.send_action(xhand_action)
                        
                        # Print status
                        arm_status = "ARM: OK" if messages.wrist_valid else "ARM: No wrist data"
                        hand_status = "HAND: OK" if messages.landmarks_valid else "HAND: No landmarks"
                        print(f"{arm_status} | {hand_status}")
                    
                    else:
                        print("Waiting for VR connection...")
                    
                    last_update_time = current_time
                
                time.sleep(0.01)  # Small sleep to prevent CPU spinning
                
        except KeyboardInterrupt:
            print("\\nStopping combined VR control...")
    
    finally:
        # Cleanup
        try:
            router.stop()
            print("VR router stopped")
        except:
            pass
        
        try:
            franka_robot.disconnect()
            print("Franka robot disconnected")
        except:
            pass
        
        try:
            xhand_robot.disconnect()
            print("XHand robot disconnected")
        except:
            pass
        
        print("Cleaning up ADB...")
        cleanup_adb_reverse(tcp_port=8000)
    
    print("Combined VR control test completed")
    return True

if __name__ == "__main__":
    success = test_combined_vr_control()
    sys.exit(0 if success else 1)