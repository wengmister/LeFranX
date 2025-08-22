#!/usr/bin/env python3
"""
Test script for XHandVRTeleoperator (lerobot teleoperator framework version).

This script tests the teleoperator class that integrates with lerobot's 
teleoperator framework, as opposed to the standalone version.

Usage:
    python test_scripts/test_xhand_vr_teleoperator.py --stub --control-freq 30
"""

import argparse
import logging
import sys
import time
from pathlib import Path
import numpy as np

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).parent.parent / "franka_xhand_teleoperator" / "src"))

from lerobot.robots.xhand.xhand import XHand
from lerobot.robots.xhand.xhand_config import XHandConfig
from lerobot.teleoperators.xhand_vr.xhand_vr_teleoperator import XHandVRTeleoperator
from lerobot.teleoperators.xhand_vr.config_xhand_vr import XHandVRTeleoperatorConfig
from dex_retargeting.constants import RobotName, RetargetingType, HandType


logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def print_joint_visualization(action):
    """
    Print a simple visualization of joint angles.
    
    Args:
        action: Dict with joint positions in radians
    """
    # Extract joint positions and convert to degrees
    joint_degrees = []
    for i in range(12):
        key = f"joint_{i}.pos"
        if key in action:
            angle_rad = action[key]
            angle_deg = angle_rad * 180.0 / np.pi
            joint_degrees.append(angle_deg)
        else:
            joint_degrees.append(0.0)
    
    # Create simple bar visualization
    print("\n" + "="*80)
    print("XHand VR Teleoperator - Joint Angles (degrees):")
    print("-" * 80)
    
    joint_names = [
        "Thumb Bend  ", "Thumb Rot1  ", "Thumb Rot2  ",
        "Index Bend  ", "Index J1    ", "Index J2    ",
        "Middle J1   ", "Middle J2   ",
        "Ring J1     ", "Ring J2     ",
        "Pinky J1    ", "Pinky J2    "
    ]
    
    for i in range(12):
        angle_deg = joint_degrees[i]
        # Create a simple bar chart (scale: -90 to +90 degrees)
        bar_length = int(abs(angle_deg) / 5)  # 5 degrees per character
        bar_length = min(bar_length, 18)  # Max 18 characters
        
        if angle_deg >= 0:
            bar = "+" * bar_length
            spaces = " " * (18 - bar_length)
            bar_display = f"|{spaces}{bar}|"
        else:
            bar = "-" * bar_length
            spaces = " " * (18 - bar_length)
            bar_display = f"|{bar}{spaces}|"
        
        print(f"{joint_names[i]}: {angle_deg:6.1f}° {bar_display}")
    
    print("="*80)
    
    # Also print raw values on one line for compact view
    angles_str = " ".join([f"{a:5.1f}" for a in joint_degrees])
    print(f"Raw angles: [{angles_str}]")
    print()


def main():
    parser = argparse.ArgumentParser(description="Test XHandVRTeleoperator (lerobot framework)")
    parser.add_argument(
        "--control-freq",
        type=float,
        default=30.0,
        help="Control frequency in Hz (default: 30.0)"
    )
    parser.add_argument(
        "--smoothing",
        type=float,
        default=0.3,
        help="Position smoothing factor 0-1 (default: 0.3)"
    )
    parser.add_argument(
        "--protocol",
        choices=["RS485", "EtherCAT"],
        default="RS485",
        help="Communication protocol (default: RS485)"
    )
    parser.add_argument(
        "--serial-port",
        default="/dev/ttyUSB0",
        help="Serial port for RS485 (default: /dev/ttyUSB0)"
    )
    parser.add_argument(
        "--stub",
        action="store_true",
        help="Use stub mode (no hardware required)"
    )
    parser.add_argument(
        "--vr-port",
        type=int,
        default=8000,
        help="TCP port for VR communication (default: 8000, avoids conflict with franka_fer_vr on 8000)"
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose VR debug output"
    )
    
    args = parser.parse_args()
    
    logger.info("="*80)
    logger.info("Testing XHandVRTeleoperator (lerobot teleoperator framework)")
    logger.info("="*80)
    logger.info("ADB setup is now handled automatically by the teleoperator")
    
    try:
        # Initialize XHand robot
        logger.info("Step 1: Initializing XHand robot...")
        robot_config = XHandConfig(
            protocol=args.protocol,
            serial_port=args.serial_port,
        )
        robot = XHand(robot_config)
        
        # Connect to robot
        logger.info("Step 2: Connecting to XHand robot...")
        if args.stub:
            logger.info("Using stub mode (no hardware)")
            robot._connect_stub()
        else:
            logger.info(f"Connecting via {args.protocol} on {args.serial_port}")
            robot.connect(calibrate=True)
        
        # Initialize teleoperator with config
        logger.info("Step 3: Setting up XHandVRTeleoperator...")
        teleop_config = XHandVRTeleoperatorConfig(
            robot_name=RobotName.xhand,
            retargeting_type=RetargetingType.dexpilot,
            hand_type=HandType.right,
            control_frequency=args.control_freq,
            smoothing_alpha=args.smoothing,
            vr_tcp_port=args.vr_port,
            vr_verbose=args.verbose
        )
        
        logger.info(f"  - Robot: {teleop_config.robot_name}")
        logger.info(f"  - Retargeting: {teleop_config.retargeting_type}")
        logger.info(f"  - Hand: {teleop_config.hand_type}")
        logger.info(f"  - Control freq: {teleop_config.control_frequency} Hz")
        logger.info(f"  - Smoothing: {teleop_config.smoothing_alpha}")
        logger.info(f"  - VR TCP port: {teleop_config.vr_tcp_port}")
        
        teleop = XHandVRTeleoperator(teleop_config)
        
        # Connect teleoperator
        logger.info("Step 4: Connecting teleoperator...")
        teleop.connect(calibrate=False)
        
        # Check connection status
        logger.info(f"Teleoperator connected: {teleop.is_connected}")
        logger.info(f"Teleoperator calibrated: {teleop.is_calibrated}")
        
        # Print action and feedback features
        logger.info("\nAction features:")
        for feature, dtype in teleop.action_features.items():
            logger.info(f"  - {feature}: {dtype.__name__}")
        
        logger.info("\nFeedback features:")
        if teleop.feedback_features:
            for feature, dtype in teleop.feedback_features.items():
                logger.info(f"  - {feature}: {dtype.__name__}")
        else:
            logger.info("  (none)")
        
        # Reset robot to home position
        logger.info("\nStep 5: Resetting robot to home position...")
        robot.reset_to_home()
        
        # Start control loop
        logger.info("\nStep 6: Starting control loop...")
        logger.info("Move your hand in front of the VR tracker to control the robot.")
        logger.info("Press Ctrl+C to stop.\n")
        
        control_period = 1.0 / args.control_freq
        frame_count = 0
        start_time = time.time()
        
        while True:
            loop_start = time.perf_counter()
            
            # Get action from teleoperator
            action = teleop.get_action()
            
            # Send action to robot
            robot.send_action(action)
            
            # Get observation from robot (for logging, currently unused)
            _ = robot.get_observation()
            
            # Visualization
            if args.stub or args.verbose:
                # In stub mode or verbose, show visualization every 10 frames
                if frame_count % 10 == 0:
                    print_joint_visualization(action)
            
            # Log statistics every second
            frame_count += 1
            if frame_count % int(args.control_freq) == 0:
                elapsed = time.time() - start_time
                actual_freq = frame_count / elapsed
                logger.info(f"Frame {frame_count}: {actual_freq:.1f} Hz (target: {args.control_freq} Hz)")
            
            # Maintain control frequency
            elapsed = time.perf_counter() - loop_start
            sleep_time = max(0, control_period - elapsed)
            if sleep_time > 0:
                time.sleep(sleep_time)
        
    except KeyboardInterrupt:
        logger.info("\nTeleoperation stopped by user")
    except Exception as e:
        logger.error(f"Error during teleoperation: {e}", exc_info=True)
        return 1
    finally:
        # Clean shutdown
        logger.info("\nShutting down...")
        try:
            if 'teleop' in locals() and teleop.is_connected:
                logger.info("Disconnecting teleoperator...")
                teleop.disconnect()
            
            if 'robot' in locals() and robot.is_connected:
                logger.info("Stopping and disconnecting robot...")
                robot.stop()
                robot.disconnect()
        except Exception as e:
            logger.warning(f"Error during shutdown: {e}")
        
        # ADB cleanup is now handled automatically by teleoperator.disconnect()
    
    logger.info("Test completed successfully")
    return 0


if __name__ == "__main__":
    sys.exit(main())