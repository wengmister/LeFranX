#!/usr/bin/env python3
"""
Test VR control of XHand using real robot and VR device
"""

import sys
import time
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "src"))
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from adb_setup import setup_adb_reverse, cleanup_adb_reverse

def test_vr_hand_control():
    print("Testing VR control of XHand...")
    
    try:
        # Import components
        import vr_message_router
        from src.hand_retargeting_processor import HandRetargetingProcessor
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
        # Initialize VR message router
        config = vr_message_router.VRRouterConfig()
        config.tcp_port = 8000
        config.verbose = False  # Reduce verbosity
        config.message_timeout_ms = 100.0
        
        router = vr_message_router.VRMessageRouter(config)
        print("VR message router created")
        
        # Initialize hand retargeting processor
        retargeting_config = {
            'verbose': True,  # Enable verbose logging for debugging
            'smoothing_factor': 0.8
        }
        
        hand_processor = HandRetargetingProcessor(retargeting_config)
        print("Hand retargeting processor created")
        
        # Initialize XHand robot
        print("Connecting to XHand robot...")
        xhand_config = XHandConfig()
        xhand_robot = XHand(xhand_config)
        xhand_robot.connect()
        print("Connected to XHand robot")
        
        # Home robot to neutral position
        print("Homing XHand to neutral position...")
        home_action = {f"joint_{i}.pos": 0.0 for i in range(12)}  # Neutral position
        xhand_robot.send_action(home_action)
        time.sleep(2.0)  # Wait for robot to reach home position
        print("XHand homed")
        
        # Get current robot state
        current_obs = xhand_robot.get_observation()
        current_joints = [current_obs[f"joint_{i}.pos"] for i in range(12)]
        
        print(f"Current XHand joints: {[f'{j:.3f}' for j in current_joints]}")
        
        # Setup hand retargeting processor
        robot_dir = str(Path(__file__).parent.parent / "dex_retargeting" / "assets" / "robots")
        config_path = str(Path(__file__).parent.parent / "dex_retargeting" / "src" / "dex_retargeting" / "configs" / "teleop" / "xhand_right_dexpilot.yml")
        
        success = hand_processor.setup(
            robot_dir=robot_dir,
            config_path=config_path,
            hand_type="Right"
        )
        
        if not success:
            print("Failed to setup hand retargeting processor")
            return False
        
        print("Hand retargeting processor setup complete")
        
        # Start VR TCP server
        if not router.start_tcp_server():
            print("Failed to start VR TCP server")
            return False
        
        print("VR TCP server started")
        print("Connect your VR device and move your hand to control XHand")
        print("Press Ctrl+C to stop")
        
        # Control loop
        last_update_time = time.time()
        update_rate = 50.0  # 50 Hz control rate (good for hand control)
        
        try:
            while True:
                current_time = time.time()
                
                # Control at fixed rate
                if current_time - last_update_time >= 1.0 / update_rate:
                    # Get VR messages
                    messages = router.get_messages()
                    status = router.get_status()
                    
                    if status['tcp_connected'] and messages.landmarks_valid:
                        # Get current robot state
                        current_obs = xhand_robot.get_observation()
                        current_joints = [current_obs[f"joint_{i}.pos"] for i in range(12)]
                        
                        # Process VR landmarks data through retargeting
                        hand_action = hand_processor.process_landmarks_data(
                            messages.landmarks_data, 
                            current_joints
                        )
                        
                        # Convert action format from hand_joint_X.pos to joint_X.pos for XHand
                        xhand_action = {}
                        for i in range(12):
                            xhand_action[f"joint_{i}.pos"] = hand_action[f"hand_joint_{i}.pos"]
                        
                        # Send action to robot
                        xhand_robot.send_action(xhand_action)
                        
                        # Print status
                        landmarks = messages.landmarks_data
                        target_joints = [hand_action[f"hand_joint_{i}.pos"] for i in range(12)]
                        max_diff = max(abs(t - c) for t, c in zip(target_joints, current_joints))
                        
                        print(f"Landmarks: {len(landmarks.landmarks)} points "
                              f"| Max joint diff: {max_diff:.3f} rad ({np.degrees(max_diff):.1f}°)")
                    
                    elif status['tcp_connected']:
                        print("VR connected but no valid landmarks data")
                    else:
                        print("Waiting for VR connection...")
                    
                    last_update_time = current_time
                
                time.sleep(0.01)  # Small sleep to prevent CPU spinning
                
        except KeyboardInterrupt:
            print("\nStopping VR hand control...")
    
    finally:
        # Cleanup
        try:
            router.stop()
            print("VR router stopped")
        except:
            pass
        
        try:
            xhand_robot.disconnect()
            print("XHand robot disconnected")
        except:
            pass
        
        print("Cleaning up ADB...")
        cleanup_adb_reverse(tcp_port=8000)
    
    print("VR hand control test completed")
    return True

if __name__ == "__main__":
    success = test_vr_hand_control()
    sys.exit(0 if success else 1)