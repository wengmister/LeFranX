from dataclasses import dataclass
from typing import List, Optional

from lerobot.teleoperators.config import TeleoperatorConfig


@TeleoperatorConfig.register_subclass("franka_fer_xhand_vr")
@dataclass
class FrankaFERXHandVRTeleoperatorConfig(TeleoperatorConfig):
    """Configuration for dual VR teleoperator controlling FrankaFER arm + XHand."""
    
    # VR connection settings (shared between arm and hand)
    vr_tcp_port: int = 8000
    setup_adb: bool = True
    vr_verbose: bool = False
    
    # Arm VR teleoperator settings
    arm_smoothing_factor: float = 0.1
    manipulability_weight: float = 0.1
    neutral_distance_weight: float = 1.0  
    current_distance_weight: float = 10.0
    arm_joint_weights: Optional[List[float]] = None
    q7_min: float = -2.8973
    q7_max: float = 2.8973
    
    # Hand VR teleoperator settings
    hand_robot_name: str = "xhand_right"
    hand_retargeting_type: str = "dexpilot"
    hand_type: str = "right"
    hand_control_frequency: float = 30.0
    hand_smoothing_alpha: float = 0.3
    
    def __post_init__(self):
        """Set default joint weights if not provided."""
        if self.arm_joint_weights is None:
            # Default joint weights for Franka robot
            self.arm_joint_weights = [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]