from dataclasses import dataclass, field
from typing import Dict

from lerobot.cameras.utils import CameraConfig
from lerobot.robots.config import RobotConfig


@RobotConfig.register_subclass("xhand")
@dataclass
class XHandConfig(RobotConfig):
    """Configuration for XHand robot with 12 DOF"""
    
    # Communication settings (based on xhand_examples.py)
    protocol: str = "RS485"  # "RS485" or "EtherCAT"
    serial_port: str = "/dev/ttyUSB0"  # For RS485 communication
    baud_rate: int = 3000000  # Default baud rate for RS485
    
    # XHand specific settings
    hand_id: int = 0  # Default hand ID
    
    # Control parameters (from examples) - XHand API expects integers
    default_kp: int = 80  # Proportional gain
    default_ki: int = 0   # Integral gain  
    default_kd: int = 0   # Derivative gain
    default_tor_max: int = 400  # Maximum torque (mA) (limited at 400mA torque-current limit)
    default_mode: int = 3     # Control mode
    
    # Robot limits and safety
    max_torque: float = 300.0  # Maximum torque per joint (mA) - from examples
    
    # Control settings
    control_frequency: float = 30.0  # Hz
    timeout: float = 1.0  # Communication timeout in seconds
    
    # Home position (12 joint angles in degrees, converted to radians in code)
    # Based on 'palm' position from examples
    home_position_deg: tuple = (0, 80.66, 33.2, 0.00, 5.11, 5.0, 
                               6.53, 5.0, 6.76, 5.0, 10.13, 5.0)
    
    # Cameras (if any are attached to the hand)
    cameras: Dict[str, CameraConfig] = field(default_factory=dict)
    
    @property
    def position_limits(self) -> Dict[str, tuple]:
        """XHand joint limits in radians (from actual XHand specifications)"""
        import math
        # Joint limits provided as [min0, max0, min1, max1, ...]
        # joint_limits_deg = [0, 105, -60, 90, -10, 105, -10, 10, 0, 110, 0, 110, 
        #                    0, 110, 0, 110, 0, 110, 0, 110, 0, 110, 0, 110]
        joint_limits_deg = [0, 105, -60, 90, -10, 105, -10, 10, 0, 110, 5, 110, 
                           0, 110, 5, 110, 0, 110, 5, 110, 0, 110, 5, 110] # Updated to prevent mechanical clogging

        limits = {}
        for i in range(12):
            min_deg = joint_limits_deg[i * 2]
            max_deg = joint_limits_deg[i * 2 + 1]
            min_rad = min_deg * math.pi / 180
            max_rad = max_deg * math.pi / 180
            limits[f"joint_{i}"] = (min_rad, max_rad)
        return limits
    
    @property
    def home_position_rad(self) -> tuple:
        """Convert home position from degrees to radians"""
        import math
        return tuple(deg * math.pi / 180 for deg in self.home_position_deg)