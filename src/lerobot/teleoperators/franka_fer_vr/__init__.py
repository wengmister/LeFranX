"""
Franka FER VR Teleoperator Module.

This module provides VR-based teleoperation for Franka FER robots using hand tracking
and C++ IK optimization. It includes automatic ADB setup for Meta Quest devices
and integrates seamlessly with the LeRobot framework.
"""

from .config_franka_fer_vr import FrankaFERVRTeleoperatorConfig
from .franka_fer_vr_teleoperator import FrankaFERVRTeleoperator

__all__ = ["FrankaFERVRTeleoperatorConfig", "FrankaFERVRTeleoperator"]