#!/usr/bin/env python3
"""
Test script for the dual VR teleoperator (FrankaFER + XHand).

This script tests the combined VR teleoperator without requiring the full robot setup.
"""

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from lerobot.robots.franka_fer_xhand.franka_fer_xhand import FrankaFERXHand
from lerobot.robots.franka_fer_xhand.franka_fer_xhand_config import FrankaFERXHandConfig
from lerobot.robots.franka_fer.franka_fer_config import FrankaFERConfig
from lerobot.robots.xhand.xhand_config import XHandConfig
from lerobot.teleoperators.franka_fer_xhand_vr.franka_fer_xhand_vr_teleoperator import FrankaFERXHandVRTeleoperator
from lerobot.teleoperators.franka_fer_xhand_vr.config_franka_fer_xhand_vr import FrankaFERXHandVRTeleoperatorConfig
from dex_retargeting.constants import RobotName, RetargetingType, HandType


def test_dual_vr_teleoperator():
    """Test the dual VR teleoperator functionality with real robots."""
    print("Testing Dual VR Teleoperator (FrankaFER + XHand) with Real Robots")
    print("=" * 70)
    
    robot = None
    teleop = None
    
    try:
        # ===== SETUP REAL ROBOT =====
        print("\nSetting up FrankaFER + XHand composite robot...")
        
        robot_config = FrankaFERXHandConfig(
            arm_config=FrankaFERConfig(
                server_ip="192.168.18.1",  # Update with your Franka IP
                server_port=5000
            ),
            hand_config=XHandConfig(
                protocol="RS485",
                serial_port="/dev/ttyUSB0"
            )
        )
        
        robot = FrankaFERXHand(robot_config)
        robot.connect(calibrate=True)
        print("  Composite robot connected")
        
        # ===== SETUP DUAL VR TELEOPERATOR =====
        print("\nSetting up dual VR teleoperator...")
        
        config = FrankaFERXHandVRTeleoperatorConfig(
            vr_tcp_port=8000,
            setup_adb=True,
            vr_verbose=False,
            # Arm settings
            arm_smoothing_factor=0.1,
            manipulability_weight=0.1,
            neutral_distance_weight=1.0,
            current_distance_weight=10.0,
            arm_joint_weights=[3.0, 3.0, 1.0, 1.0, 1.0, 1.0, 1.0],
            # Hand settings
            hand_robot_name=RobotName.xhand,
            hand_retargeting_type=RetargetingType.dexpilot,
            hand_type=HandType.right,
            hand_control_frequency=30.0,
            hand_smoothing_alpha=0.5
        )
        
        teleop = FrankaFERXHandVRTeleoperator(config)
        print("  Dual VR teleoperator created")
        
        # ===== CONNECT TELEOPERATOR =====
        print("\nConnecting dual VR teleoperator...")
        teleop.connect(calibrate=False)
        teleop.set_robot(robot)  # Set robot reference for arm IK
        print("  Dual VR teleoperator connected with robot reference")
        
        # ===== CHECK ACTION FEATURES =====
        print("\nChecking action features...")
        action_features = teleop.action_features
        
        arm_features = {k: v for k, v in action_features.items() if k.startswith('arm_')}
        hand_features = {k: v for k, v in action_features.items() if k.startswith('hand_')}
        
        print(f"  Arm features ({len(arm_features)}):")
        for feature in sorted(arm_features.keys()):
            print(f"    - {feature}")
        
        print(f"  Hand features ({len(hand_features)}):")
        for feature in sorted(hand_features.keys()):
            print(f"    - {feature}")
        
        
        # ===== CHECK CONNECTION STATUS =====
        print("\nChecking connection status...")
        status = teleop.get_status()
        print(f"  Connected: {status.get('connected', False)}")
        
        if 'arm_status' in status:
            arm_status = status['arm_status']
            print(f"  Arm VR Ready: {arm_status.get('vr_ready', False)}")
            print(f"  Arm TCP Connected: {arm_status.get('tcp_connected', False)}")
        
        if 'hand_status' in status:
            hand_status = status['hand_status'] 
            print(f"  Hand Connected: {hand_status.get('connected', False)}")
        
        # ===== RESET TO HOME POSITION =====
        print("\nResetting robots to home position...")
        home_success = robot.reset_to_home()
        if home_success:
            print("  Both arm and hand reset to home position")
        else:
            print("  Warning: Home reset may have had issues")
        
        # Wait a moment for reset to complete
        time.sleep(2.0)
        
        # ===== TEST ACTION GENERATION =====
        print("\nTesting VR control with robot movement...")
        print("Connect your VR device and move your hand to control both robots!")
        print("Press Ctrl+C to stop")
        
        frame_count = 0
        start_time = time.time()
        
        while True:
            loop_start = time.perf_counter()
            
            try:
                # Get action from dual teleoperator
                action = teleop.get_action()
                
                # Send action to robot to actually move it!
                robot.send_action(action)
                
                # Count arm and hand actions
                arm_actions = {k: v for k, v in action.items() if k.startswith('arm_')}
                hand_actions = {k: v for k, v in action.items() if k.startswith('hand_')}
                
                # Print status every 30 frames (once per second at 30Hz)
                if frame_count % 30 == 0:
                    elapsed = time.time() - start_time
                    actual_freq = frame_count / elapsed if elapsed > 0 else 0
                    
                    print(f"Frame {frame_count}: {actual_freq:.1f} Hz")
                    print(f"  Arm actions: {len(arm_actions)} (sample: {list(arm_actions.keys())[:3]})")
                    print(f"  Hand actions: {len(hand_actions)} (sample: {list(hand_actions.keys())[:3]})")
                    
                    # Show sample values
                    if arm_actions:
                        sample_arm = {k: f"{v:.3f}" for k, v in list(arm_actions.items())[:2]}
                        print(f"     Sample arm values: {sample_arm}")
                    
                    if hand_actions:
                        sample_hand = {k: f"{v:.3f}" for k, v in list(hand_actions.items())[:2]}
                        print(f"     Sample hand values: {sample_hand}")
                        
                        # Check if hand values are actually changing
                        hand_sum = sum(abs(v) for v in hand_actions.values())
                        print(f"     Hand motion magnitude: {hand_sum:.3f}")
                
                frame_count += 1
                
                # Maintain ~30Hz rate
                elapsed = time.perf_counter() - loop_start
                sleep_time = max(0, 1.0/30.0 - elapsed)
                if sleep_time > 0:
                    time.sleep(sleep_time)
                    
            except Exception as e:
                print(f"Error in action generation: {e}")
                break
                
    except KeyboardInterrupt:
        print("\nTest stopped by user")
        
    except Exception as e:
        print(f"\nTest failed: {e}")
        import traceback
        traceback.print_exc()
        return False
        
    finally:
        # ===== CLEANUP =====
        print("\nCleaning up...")
        
        if teleop:
            try:
                teleop.disconnect()
                print("  Dual VR teleoperator disconnected")
            except Exception as e:
                print(f"  Error disconnecting teleoperator: {e}")
        
        if robot:
            try:
                robot.disconnect()
                print("  Composite robot disconnected")
            except Exception as e:
                print(f"  Error disconnecting robot: {e}")
    
    print("\nDual VR teleoperator test completed!")
    return True


if __name__ == "__main__":
    success = test_dual_vr_teleoperator()
    sys.exit(0 if success else 1)