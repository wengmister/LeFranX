#!/usr/bin/env python3
"""
ADB setup utilities for VR Message Router
Handles Android device port forwarding setup
"""

import subprocess
import logging

logger = logging.getLogger(__name__)

def setup_adb_reverse(tcp_port=8000):
    """Setup adb reverse port forwarding for Android VR apps"""
    try:
        # Check if adb is available
        result = subprocess.run(['adb', 'devices'], 
                              capture_output=True, text=True, timeout=5)
        if result.returncode != 0:
            logger.warning("adb command not found. Please install Android SDK platform-tools.")
            return False
        
        # Check if device is connected
        if "device" not in result.stdout:
            logger.warning("No Android device connected via adb.")
            return False
        
        # Setup reverse port forwarding
        cmd = ['adb', 'reverse', f'tcp:{tcp_port}', f'tcp:{tcp_port}']
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=10)
        
        if result.returncode == 0:
            logger.info(f"Successfully setup adb reverse tcp:{tcp_port}")
            return True
        else:
            logger.warning(f"Failed to setup adb reverse: {result.stderr}")
            return False
            
    except Exception as e:
        logger.warning(f"Error setting up adb reverse: {e}")
        return False

def cleanup_adb_reverse(tcp_port=8000):
    """Remove adb reverse port forwarding"""
    try:
        cmd = ['adb', 'reverse', '--remove', f'tcp:{tcp_port}']
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=10)
        
        if result.returncode == 0:
            logger.info(f"Successfully removed adb reverse tcp:{tcp_port}")
        else:
            logger.warning(f"Failed to remove adb reverse: {result.stderr}")
            
    except Exception as e:
        logger.warning(f"Error cleaning up adb reverse: {e}")

if __name__ == "__main__":
    # Test ADB setup
    print("Testing ADB setup...")
    
    if setup_adb_reverse(8000):
        print("ADB reverse setup successful")
        
        input("Press Enter to cleanup...")
        cleanup_adb_reverse(8000)
        print("ADB cleanup completed")
    else:
        print("ADB setup failed")