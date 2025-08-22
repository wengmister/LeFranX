from dataclasses import dataclass

from dex_retargeting.constants import RobotName, RetargetingType, HandType
from lerobot.teleoperators.config import TeleoperatorConfig


@TeleoperatorConfig.register_subclass("xhand_vr")
@dataclass 
class XHandVRTeleoperatorConfig(TeleoperatorConfig):
    """Configuration for XHand VR teleoperation using dex-retargeting."""
    
    # Retargeting settings
    robot_name: RobotName = RobotName.xhand
    retargeting_type: RetargetingType = RetargetingType.dexpilot
    hand_type: HandType = HandType.right
    
    # Control settings
    control_frequency: float = 30.0  # Hz
    smoothing_alpha: float = 0.3  # Exponential smoothing factor
    
    # Robot directory for URDF files (optional, will use default if None)
    robot_dir: str | None = None
    
    # VR message router settings
    vr_tcp_port: int = 8001  # TCP port for VR message router (8001 to avoid conflict with franka_fer_vr on 8000)
    vr_verbose: bool = False  # Enable verbose VR logging
    setup_adb: bool = True  # Automatically setup adb reverse port forwarding using adb_setup.py