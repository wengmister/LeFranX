"""
Configuration for Franka FER VR Teleoperator.

This module defines the configuration class for the Franka FER VR teleoperator,
including TCP connection settings, VR processing parameters, and IK solver options.
"""

from dataclasses import dataclass
from typing import List, Optional

from lerobot.teleoperators.config import TeleoperatorConfig


@TeleoperatorConfig.register_subclass("franka_fer_vr")
@dataclass
class FrankaFERVRTeleoperatorConfig(TeleoperatorConfig):
    """
    Configuration for Franka FER VR teleoperator using C++ IK bridge.
    
    This configuration class contains all parameters needed to set up VR-based
    teleoperation of a Franka FER robot, including network settings, motion
    processing parameters, and IK solver weights.
    """
    
    # TCP connection settings
    tcp_port: int = 8000
    setup_adb: bool = True  # Automatically setup adb reverse port forwarding using adb_setup.py
    
    # VR processing settings
    smoothing_factor: float = 0.7  # Higher = more smoothing (0-1)
    position_deadzone: float = 0.001  # 1mm deadzone to prevent drift
    orientation_deadzone: float = 0.03  # ~1.7 degrees deadzone
    max_position_offset: float = 0.75  # 75cm max workspace
    
    # IK solver settings
    manipulability_weight: float = 1.0
    neutral_distance_weight: float = 2.0
    current_distance_weight: float = 2.0
    joint_weights: Optional[List[float]] = None  # Will default to [2.0, 2.0, 1.0, 1.0, 1.0, 1.0, 1.0]
    
    # Q7 limits (can be configured for different end effectors)
    q7_min: float = -2.89  # Full Franka range
    q7_max: float = 2.89
    use_bidexhand_limits: bool = False  # If True, use [-0.2, 1.9] for BiDexHand
    
    # Debug settings
    verbose: bool = False
    
    def __post_init__(self):
        """Initialize default values and apply BiDexHand limits if requested."""
        # Set default joint weights if not provided
        if self.joint_weights is None:
            # Higher weights for base joints for stability
            self.joint_weights = [2.0, 2.0, 1.0, 1.0, 1.0, 1.0, 1.0]
        
        # Apply BiDexHand limits if requested
        if self.use_bidexhand_limits:
            self.q7_min = -0.2
            self.q7_max = 1.9