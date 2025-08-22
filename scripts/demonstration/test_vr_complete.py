#!/usr/bin/env python3
"""
Complete VR teleoperator test with robot connection.
Tests the full VR teleoperation pipeline.
"""

import time
import sys
from pathlib import Path

# Add franka_teleoperator to path for importing the module
sys.path.insert(0, str(Path(__file__).parent / "franka_teleoperator"))

from lerobot.robots.franka_fer.franka_fer_config import FrankaFERConfig
from lerobot.robots.franka_fer.franka_fer import FrankaFER
from lerobot.teleoperators.vr_teleoperator import VRTeleoperator, VRTeleoperatorConfig

def test_complete_vr_system():
    """Test complete VR system with robot connection"""
    print("🤖 Testing Complete VR Teleoperation System")
    print("=" * 50)
    
    # Create robot configuration
    robot_config = FrankaFERConfig(
        server_ip="192.168.18.1",
        server_port=5000,
        cameras={}  # No cameras for this test
    )
    
    # Create VR teleoperator configuration
    vr_config = VRTeleoperatorConfig(
        tcp_port=8000,
        verbose=False,  # Disable verbose mode to reduce debug output
        smoothing_factor=0.7,
        use_bidexhand_limits=False
    )
    
    robot = None
    vr_teleop = None
    
    try:
        # Step 1: Connect to robot
        print("🔄 Connecting to robot...")
        robot = FrankaFER(robot_config)
        robot.connect(calibrate=False)
        print("✓ Robot connected successfully")
        
        # Step 2: Create and connect VR teleoperator
        print("🔄 Creating VR teleoperator...")
        vr_teleop = VRTeleoperator(vr_config)
        print("✓ VR teleoperator created")
        
        print("🔄 Connecting VR teleoperator to robot...")
        # Set robot reference for VR teleoperator
        vr_teleop._robot_reference = robot
        vr_teleop.connect()
        print("✓ VR teleoperator connected")
        print(f"✓ VR TCP server listening on port {vr_config.tcp_port}")
        
        # Step 3: Test basic functionality
        print("\n📊 Testing basic functionality...")
        
        # Test action features
        features = vr_teleop.action_features
        print(f"✓ Action features: {list(features.keys())}")
        
        # Test connection status
        print(f"✓ VR teleoperator connected: {vr_teleop.is_connected}")
        print(f"✓ VR teleoperator calibrated: {vr_teleop.is_calibrated}")
        
        # Test VR status
        status = vr_teleop.get_status()
        print(f"✓ VR status: connected={status.get('connected', False)}")
        
        # Step 4: Test action loop for a few iterations
        print("\n🎯 Testing action generation...")
        print("📱 Waiting for VR input...")
        print("Connect your VR app to localhost:8000 or press Ctrl+C to skip")
        
        start_time = time.time()
        test_duration = 10.0  # Test for 10 seconds
        iteration = 0
        
        while time.time() - start_time < test_duration:
            try:
                # Get action from VR
                action = vr_teleop.get_action()
                
                # Extract joint positions
                joint_positions = [action[f"joint_{i}.pos"] for i in range(7)]
                
                # Get VR status
                status = vr_teleop.get_status()
                
                # Print status every 0.1 second
                if iteration % 2 == 0:  # Every 0.1 second at 25Hz
                    print(f"\n📊 Iteration {iteration}:")
                    print(f"  VR ready: {'✓' if status.get('vr_ready', False) else '✗'}")
                    print(f"  Joint targets: {[f'{j:.3f}' for j in joint_positions]}")
                    
                    if status.get('position'):
                        pos = status['position']
                        print(f"  VR position: [{pos[0]:.3f}, {pos[1]:.3f}, {pos[2]:.3f}]")
                
                # Optional: Send action to robot (BE CAREFUL!)
                robot.send_action(action)
                
                iteration += 1
                time.sleep(0.04)  # 25Hz
                
            except KeyboardInterrupt:
                print("\n🛑 Test interrupted by user")
                break
            except Exception as e:
                print(f"✗ Error in action loop: {e}")
                break
        
        print(f"\n✅ Completed {iteration} action cycles")
        return True
        
    except Exception as e:
        print(f"✗ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False
        
    finally:
        # Cleanup
        print("\n🔌 Cleaning up...")
        if vr_teleop and vr_teleop.is_connected:
            try:
                vr_teleop.disconnect()
                print("✓ VR teleoperator disconnected")
            except Exception as e:
                print(f"⚠️ VR cleanup error: {e}")
        
        if robot and robot.is_connected:
            try:
                robot.disconnect()
                print("✓ Robot disconnected")
            except Exception as e:
                print(f"⚠️ Robot cleanup error: {e}")

def main():
    """Main test function"""
    print("Complete VR Teleoperation Test")
    print("=" * 35)
    
    success = test_complete_vr_system()
    
    print("\n" + "=" * 35)
    if success:
        print("🎉 VR teleoperation test completed!")
        print("\n💡 System is ready for:")
        print("1. Real VR teleoperation with your VR app")
        print("2. Data collection with lerobot-record")
        print("3. Diffusion policy training and deployment")
    else:
        print("❌ VR teleoperation test failed!")
        print("Check the error messages above for debugging")

if __name__ == "__main__":
    main()