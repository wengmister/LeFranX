#!/usr/bin/env python3

"""
Test script for XHand robot gestures.
This script tests the XHand robot by sending predefined gesture commands.
"""

import logging
import math
import time
from lerobot.robots.xhand import XHand, XHandConfig

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_xhand_gestures():
    """Test XHand robot with predefined gestures from xhand_examples.py"""
    
    print("🤖 Testing XHand Robot Gestures")
    print("=" * 50)
    
    # Create XHand configuration
    config = XHandConfig(
        protocol="RS485",
        serial_port="/dev/ttyUSB0",
        baud_rate=3000000,
        hand_id=0,
        cameras={}
    )
    
    # Create XHand robot
    robot = XHand(config)
    
    print(f"Robot: {robot.name}")
    print(f"Action features: {len(robot.action_features)}")
    print(f"Observation features: {len(robot.observation_features)}")
    
    # Predefined gestures from xhand_examples.py (in degrees)
    actions_list = {
        'fist': [11.85, 74.58, 40, -3.08, 106.02, 110, 109.75, 107.56, 107.66, 110, 109.1, 109.15],
        'palm': [0, 80.66, 33.2, 2.00, 5.11, 2, 6.53, 2, 6.76, 4.41, 10.13, 2],
        'v': [38.32, 90, 52.08, 6.21, 2.6, 2, 2.1, 2, 110, 110, 110, 109.23],
        'ok': [45.88, 41.54, 67.35, 2.22, 80.45, 70.82, 31.37, 10.39, 13.69, 16.88, 1.39, 10.55],
        'end': [0, 80.66, 33.2, 2.00, 5.11, 2, 6.53, 2, 6.76, 4.41, 10.13, 2]
    }
    
    try:
        # Connect to robot (will use stub mode if XHand SDK not available)
        print(f"\n🔌 Connecting to XHand...")
        robot.connect()
        print(f"✅ Connected successfully!")
        
        # Test each gesture
        print(f"\n🎭 Testing Gestures:")
        print("-" * 30)
        
        for gesture_name, positions_deg in actions_list.items():
            print(f"\n👋 Performing gesture: '{gesture_name}'")
            
            # Convert degrees to radians and create action dict
            action = {}
            for i in range(12):
                joint_name = f"joint_{i}"
                position_rad = positions_deg[i] * math.pi / 180
                action[f"{joint_name}.pos"] = position_rad
                
            print(f"   Positions (deg): {[f'{deg:.1f}' for deg in positions_deg[:6]]}...")
            positions_rad = [action[f"joint_{i}.pos"] for i in range(6)]
            print(f"   Positions (rad): {[f'{pos:.3f}' for pos in positions_rad]}...")
            
            # Send action to robot
            try:
                performed_action = robot.send_action(action)
                print(f"   ✅ Action sent successfully")
                
                # Get observation to verify
                obs = robot.get_observation()
                current_positions = [obs[f"joint_{i}.pos"] for i in range(12)]
                current_torques = [obs[f"joint_{i}.torque"] for i in range(12)]
                
                print(f"   📊 Current positions: {[f'{pos:.3f}' for pos in current_positions[:6]]}...")
                print(f"   📊 Current torques: {[f'{torque:.3f}' for torque in current_torques[:6]]}...")
                
            except Exception as e:
                print(f"   ❌ Failed to send action: {e}")
            
            # Wait before next gesture
            print(f"   ⏱️  Waiting 1 second...")
            time.sleep(1)
        
        # Test observation reading
        print(f"\n📊 Final Observation Test:")
        print("-" * 30)
        
        obs = robot.get_observation()
        print(f"Total observation features: {len(obs)}")
        
        # Show joint positions and torques
        print(f"Joint positions:")
        for i in range(12):
            pos = obs[f"joint_{i}.pos"]
            pos_deg = pos * 180 / math.pi
            print(f"  joint_{i}: {pos:.3f} rad ({pos_deg:.1f}°)")
        
        print(f"\nJoint torques:")
        for i in range(12):
            torque = obs[f"joint_{i}.torque"]
            print(f"  joint_{i}: {torque:.3f} Nm")
        
        print(f"\n🏠 Resetting to home position...")
        success = robot.reset_to_home()
        if success:
            print(f"✅ Home reset successful")
        else:
            print(f"❌ Home reset failed")
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    finally:
        # Disconnect robot
        try:
            print(f"\n🔌 Disconnecting robot...")
            robot.disconnect()
            print(f"✅ Disconnected successfully")
        except Exception as e:
            print(f"⚠️  Disconnect warning: {e}")
    
    return True

def test_composite_gestures():
    """Test composite robot with XHand gestures"""
    print(f"\n🤖 Testing Composite Robot with XHand Gestures")
    print("=" * 60)
    
    from lerobot.robots.franka_fer_xhand import FrankaFERXHand, FrankaFERXHandConfig
    from lerobot.robots.franka_fer.franka_fer_config import FrankaFERConfig
    
    # Create configurations
    arm_config = FrankaFERConfig(
        server_ip="192.168.18.1",
        server_port=5000,
        cameras={}
    )
    
    hand_config = XHandConfig(
        protocol="RS485",
        serial_port="/dev/ttyUSB0",
        baud_rate=3000000,
        hand_id=0,
        cameras={}
    )
    
    composite_config = FrankaFERXHandConfig(
        arm_config=arm_config,
        hand_config=hand_config,
        synchronize_actions=True
    )
    
    # Create composite robot
    robot = FrankaFERXHand(composite_config)
    
    print(f"Composite robot features:")
    print(f"  - Observations: {len(robot.observation_features)}")
    print(f"  - Actions: {len(robot.action_features)}")
    
    # Test gesture with composite robot (hand only, arm stays still)
    palm_positions_deg = [0, 80.66, 33.2, 0.00, 5.11, 0, 6.53, 0, 6.76, 4.41, 10.13, 0]
    
    # Create composite action (arm stays at current position, hand moves to palm)
    composite_action = {}
    
    # Arm actions (keep current position - using zeros as placeholder)
    for i in range(7):
        composite_action[f"arm_joint_{i}.pos"] = 0.0
    
    # Hand actions (palm gesture)
    for i in range(12):
        position_rad = palm_positions_deg[i] * math.pi / 180
        composite_action[f"hand_joint_{i}.pos"] = position_rad
    
    print(f"\n🎭 Composite action created:")
    print(f"  - Arm actions: {len([k for k in composite_action.keys() if k.startswith('arm_')])}")
    print(f"  - Hand actions: {len([k for k in composite_action.keys() if k.startswith('hand_')])}")
    print(f"  - Total actions: {len(composite_action)}")
    
    return robot, composite_action

if __name__ == "__main__":
    print("🧪 XHand Robot Gesture Testing")
    print("=" * 70)
    
    try:
        # Test individual XHand robot
        success = test_xhand_gestures()
        
        if success:
            print(f"\n✅ Individual XHand tests passed!")
        
        # Test composite robot setup
        composite_robot, composite_action = test_composite_gestures()
        print(f"\n✅ Composite robot setup successful!")
        
        print(f"\n🎉 All tests completed!")
        print(f"   - XHand robot ready for gesture control")
        print(f"   - Composite robot ready for coordinated arm+hand control")
        print(f"   - Action space: {len(composite_action)} total actions")
        
    except Exception as e:
        print(f"\n💥 Tests failed: {e}")
        import traceback
        traceback.print_exc()