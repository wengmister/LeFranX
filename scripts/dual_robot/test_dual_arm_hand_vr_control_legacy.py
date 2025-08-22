#!/usr/bin/env python3
"""
Test script for dual arm+hand VR control using shared VR router.

This script demonstrates how both FrankaFER arm and XHand can be controlled
simultaneously from the same VR source using the shared VRRouterManager.
"""

import sys
import time
import threading
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from lerobot.robots.franka_fer.franka_fer import FrankaFER
from lerobot.robots.franka_fer.franka_fer_config import FrankaFERConfig
from lerobot.teleoperators.franka_fer_vr.franka_fer_vr_teleoperator import FrankaFERVRTeleoperator
from lerobot.teleoperators.franka_fer_vr.config_franka_fer_vr import FrankaFERVRTeleoperatorConfig

from lerobot.robots.xhand.xhand import XHand
from lerobot.robots.xhand.xhand_config import XHandConfig
from lerobot.teleoperators.xhand_vr.xhand_vr_teleoperator import XHandVRTeleoperator
from lerobot.teleoperators.xhand_vr.config_xhand_vr import XHandVRTeleoperatorConfig
from dex_retargeting.constants import RobotName, RetargetingType, HandType

from lerobot.teleoperators.vr_router_manager import get_vr_router_manager


def test_dual_vr_control(use_stub=False):
    """
    Test dual arm+hand VR control using shared VR router manager.
    
    Args:
        use_stub: If True, use stub mode for testing without hardware
    """
    print("🚀 Testing Dual Arm+Hand VR Control with Shared Router")
    print("=" * 60)
    
    # Get the shared VR router manager
    vr_manager = get_vr_router_manager()
    print(f"📡 VR Router Manager: {vr_manager}")
    
    arm_robot = None
    hand_robot = None
    arm_teleop = None
    hand_teleop = None
    
    try:
        # ===== SETUP ARM CONTROL =====
        print("\n🦾 Setting up Franka FER arm control...")
        
        # Initialize arm robot
        arm_config = FrankaFERConfig()
        arm_robot = FrankaFER(arm_config)
        
        arm_robot.connect()
        print("  ✅ Arm robot connected")
        
        # Initialize arm teleoperator
        arm_teleop_config = FrankaFERVRTeleoperatorConfig(
            tcp_port=8000,  # Shared port
            setup_adb=True,
            smoothing_factor=0.1,
            verbose=False
        )
        arm_teleop = FrankaFERVRTeleoperator(arm_teleop_config)
        arm_teleop.connect(calibrate=False)
        arm_teleop.set_robot(arm_robot)
        print("  ✅ Arm teleoperator connected")
        
        # ===== SETUP HAND CONTROL =====
        print("\n✋ Setting up XHand control...")
        
        # Initialize hand robot  
        hand_config = XHandConfig(
            protocol="RS485",
            serial_port="/dev/ttyUSB0",
        )
        hand_robot = XHand(hand_config)
        
        hand_robot.connect(calibrate=True)
        print("  ✅ Hand robot connected")
        
        # Initialize hand teleoperator
        hand_teleop_config = XHandVRTeleoperatorConfig(
            robot_name=RobotName.xhand,
            retargeting_type=RetargetingType.dexpilot,
            hand_type=HandType.right,
            vr_tcp_port=8000,  # Same shared port
            setup_adb=False,   # ARM teleop already set it up
            vr_verbose=False
        )
        hand_teleop = XHandVRTeleoperator(hand_teleop_config)
        hand_teleop.connect(calibrate=False)
        print("  ✅ Hand teleoperator connected")
        
        # ===== VERIFY SHARED ROUTER =====
        print("\n📊 VR Router Manager Status:")
        status = vr_manager.get_status()
        for key, value in status.items():
            print(f"  {key}: {value}")
        
        print(f"\n🔗 Reference Count: {vr_manager.reference_count} teleoperators")
        print("  Both teleoperators should be sharing the same VR router!")
        
        # ===== DUAL CONTROL LOOP =====
        print("\n🎮 Starting dual control loop...")
        print("📱 Connect your VR device and move your hand to control both arm and hand")
        print("⏹️  Press Ctrl+C to stop")
        
        frame_count = 0
        start_time = time.time()
        control_rate = 30.0  # 30 Hz
        
        while True:
            loop_start = time.perf_counter()
            
            # Get actions from both teleoperators simultaneously
            try:
                arm_action = arm_teleop.get_action()
                hand_action = hand_teleop.get_action()
                
                # Send actions to robots
                if arm_action:
                    arm_robot.send_action(arm_action)
                
                if hand_action:
                    hand_robot.send_action(hand_action)
                
                # Print status every 30 frames (once per second at 30Hz)
                if frame_count % 30 == 0:
                    elapsed = time.time() - start_time
                    actual_freq = frame_count / elapsed if elapsed > 0 else 0
                    
                    print(f"📊 Frame {frame_count}: {actual_freq:.1f} Hz")
                    print(f"  🦾 Arm action keys: {list(arm_action.keys()) if arm_action else 'None'}")
                    print(f"  ✋ Hand action keys: {list(hand_action.keys()) if hand_action else 'None'}")
                    
                    # Show VR manager status
                    vr_status = vr_manager.get_status()
                    tcp_connected = vr_status.get('tcp_connected', False)
                    print(f"  📡 VR Connected: {tcp_connected}")
                
                frame_count += 1
                
                # Maintain control rate
                elapsed = time.perf_counter() - loop_start
                sleep_time = max(0, 1.0/control_rate - elapsed)
                if sleep_time > 0:
                    time.sleep(sleep_time)
                    
            except Exception as e:
                print(f"❌ Error in control loop: {e}")
                break
                
    except KeyboardInterrupt:
        print("\n⏹️  Stopping dual VR control...")
        
    except Exception as e:
        print(f"\n❌ Error during setup: {e}")
        return False
        
    finally:
        # ===== CLEANUP =====
        print("\n🧹 Cleaning up...")
        
        # Disconnect teleoperators first (this will unregister from VR manager)
        if hand_teleop:
            try:
                hand_teleop.disconnect()
                print("  ✅ Hand teleoperator disconnected")
            except Exception as e:
                print(f"  ⚠️  Error disconnecting hand teleoperator: {e}")
        
        if arm_teleop:
            try:
                arm_teleop.disconnect()
                print("  ✅ Arm teleoperator disconnected")
            except Exception as e:
                print(f"  ⚠️  Error disconnecting arm teleoperator: {e}")
        
        # Disconnect robots
        if hand_robot:
            try:
                hand_robot.disconnect()
                print("  ✅ Hand robot disconnected")
            except Exception as e:
                print(f"  ⚠️  Error disconnecting hand robot: {e}")
                
        if arm_robot:
            try:
                arm_robot.disconnect()
                print("  ✅ Arm robot disconnected")
            except Exception as e:
                print(f"  ⚠️  Error disconnecting arm robot: {e}")
        
        # Check final VR manager status
        final_status = vr_manager.get_status()
        print(f"\n📊 Final VR Manager Status:")
        print(f"  Reference Count: {vr_manager.reference_count}")
        print(f"  Router Started: {final_status.get('router_started', False)}")
        print("  (Should be 0 and False after cleanup)")
    
    print("\n🎉 Dual VR control test completed!")
    return True


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Test dual arm+hand VR control")
    parser.add_argument(
        "--hardware", 
        action="store_true", 
        help="Use real hardware (default: stub mode)"
    )
    
    args = parser.parse_args()
    
    success = test_dual_vr_control(use_stub=False)
    sys.exit(0 if success else 1)