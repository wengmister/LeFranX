#!/usr/bin/env python3
"""
Simple test script for VR Message Router
"""

import time
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "src"))

from adb_setup import setup_adb_reverse, cleanup_adb_reverse

def test_vr_message_router():
    print("Testing VR Message Router...")
    
    try:
        import vr_message_router
        print("Successfully imported vr_message_router")
    except ImportError as e:
        print(f"Failed to import vr_message_router: {e}")
        return False
    
    # Setup ADB first
    print("Setting up ADB reverse port forwarding...")
    setup_adb_reverse(tcp_port=8000)
    
    # Test configuration
    config = vr_message_router.VRRouterConfig()
    config.tcp_port = 8000  # Use real VR port
    config.verbose = True
    
    # Create router
    router = vr_message_router.VRMessageRouter(config)
    print(f"Created VRMessageRouter on port {config.tcp_port}")
    
    # Start TCP server
    if not router.start_tcp_server():
        print("Failed to start TCP server")
        cleanup_adb_reverse(tcp_port=8000)
        return False
    
    print("TCP server started, waiting for VR connection...")
    print("Connect your VR device and move your hand")
    print("Press Ctrl+C to stop")
    
    # Test receiving real VR messages
    try:
        while True:
            messages = router.get_messages()
            status = router.get_status()
            
            if status['tcp_connected']:
                print("VR device connected")
                
                if messages.wrist_valid:
                    wrist = messages.wrist_data
                    print(f"Wrist: pos=({wrist.position[0]:.3f}, {wrist.position[1]:.3f}, {wrist.position[2]:.3f}) fist={wrist.fist_state}")
                
                if messages.landmarks_valid:
                    landmarks = messages.landmarks_data
                    print(f"Landmarks: {len(landmarks.landmarks)} points")
            else:
                print("Waiting for VR connection...")
            
            time.sleep(0.1)
            
    except KeyboardInterrupt:
        print("\nStopping...")
    
    # Stop router
    router.stop()
    print("Router stopped")
    
    # Cleanup ADB
    print("Cleaning up ADB reverse port forwarding...")
    cleanup_adb_reverse(tcp_port=8000)
    
    return True

if __name__ == "__main__":
    success = test_vr_message_router()
    sys.exit(0 if success else 1)