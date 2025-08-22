from dataclasses import dataclass, field
from typing import Dict

from lerobot.cameras.utils import CameraConfig
from lerobot.robots.config import RobotConfig
from lerobot.robots.franka_fer.franka_fer_config import FrankaFERConfig
from lerobot.robots.xhand.xhand_config import XHandConfig


@RobotConfig.register_subclass("franka_fer_xhand")
@dataclass
class FrankaFERXHandConfig(RobotConfig):
    """Configuration for combined Franka FER arm + XHand robot"""
    
    # Sub-robot configurations (with empty cameras to avoid circular imports)
    arm_config: FrankaFERConfig = field(default_factory=lambda: FrankaFERConfig(cameras={}))
    hand_config: XHandConfig = field(default_factory=lambda: XHandConfig(cameras={}))
    
    # Data collection cameras (empty for now)
    cameras: Dict[str, CameraConfig] = field(default_factory=dict)
    
    # Combined robot settings
    synchronize_actions: bool = True  # Send arm and hand actions simultaneously
    action_timeout: float = 0.1  # Timeout for synchronized actions (seconds)
    
    # Safety settings
    check_arm_hand_collision: bool = True  # Check for arm-hand collisions
    emergency_stop_both: bool = True  # Stop both arm and hand on any error
    
    def __post_init__(self):
        """Validate configuration after initialization"""
        super().__post_init__()
        
        if not isinstance(self.arm_config, FrankaFERConfig):
            raise TypeError("arm_config must be a FrankaFERConfig")
        if not isinstance(self.hand_config, XHandConfig):
            raise TypeError("hand_config must be a XHandConfig")
    
    @property
    def all_cameras(self) -> Dict[str, CameraConfig]:
        """Get all cameras from both arm and hand"""
        cameras = {}
        
        # Add arm cameras with 'arm_' prefix
        for name, config in self.arm_config.cameras.items():
            cameras[f"arm_{name}"] = config
            
        # Add hand cameras with 'hand_' prefix  
        for name, config in self.hand_config.cameras.items():
            cameras[f"hand_{name}"] = config
            
        return cameras