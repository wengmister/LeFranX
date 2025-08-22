#!/usr/bin/env python3
"""
Dual VR Teleoperator for FrankaFER + XHand composite robot using shared VR router.

This teleoperator combines arm and hand VR control into a unified interface for
controlling the FrankaFERXHand composite robot. It uses the shared VR router manager
to coordinate data between arm and hand control.
"""

import logging
from typing import Dict, Any

from lerobot.teleoperators.teleoperator import Teleoperator
from lerobot.teleoperators.franka_fer_vr.franka_fer_vr_teleoperator import FrankaFERVRTeleoperator
from lerobot.teleoperators.xhand_vr.xhand_vr_teleoperator import XHandVRTeleoperator
from lerobot.teleoperators.franka_fer_vr.config_franka_fer_vr import FrankaFERVRTeleoperatorConfig
from lerobot.teleoperators.xhand_vr.config_xhand_vr import XHandVRTeleoperatorConfig
from dex_retargeting.constants import RobotName, RetargetingType, HandType

from .config_franka_fer_xhand_vr import FrankaFERXHandVRTeleoperatorConfig

logger = logging.getLogger(__name__)


class FrankaFERXHandVRTeleoperator(Teleoperator):
    """
    Dual VR teleoperator for controlling FrankaFER arm + XHand simultaneously.
    
    This teleoperator combines the arm and hand VR teleoperators into a unified
    interface that outputs actions in the format expected by the FrankaFERXHand
    composite robot (with 'arm_' and 'hand_' prefixes).
    
    Features:
    - Unified VR control for both arm and hand
    - Shared VR router manager for coordinated data access
    - Action prefix mapping for composite robot compatibility
    - Coordinated connection/disconnection lifecycle
    
    The data flow is:
    1. Single VR source provides both wrist and hand landmark data
    2. Arm teleoperator processes wrist data for arm control
    3. Hand teleoperator processes landmarks data for hand control  
    4. Combined actions are output with proper prefixes
    """
    
    config_class = FrankaFERXHandVRTeleoperatorConfig
    name = "franka_fer_xhand_vr"
    
    def __init__(self, config: FrankaFERXHandVRTeleoperatorConfig):
        super().__init__(config)
        self.config = config
        
        # Create arm VR teleoperator
        arm_config = FrankaFERVRTeleoperatorConfig(
            tcp_port=config.vr_tcp_port,
            setup_adb=config.setup_adb,
            verbose=config.vr_verbose,
            smoothing_factor=config.arm_smoothing_factor,
            manipulability_weight=config.manipulability_weight,
            neutral_distance_weight=config.neutral_distance_weight,
            current_distance_weight=config.current_distance_weight,
            joint_weights=config.arm_joint_weights,
            q7_min=config.q7_min,
            q7_max=config.q7_max
        )
        self.arm_teleop = FrankaFERVRTeleoperator(arm_config)
        
        # Create hand VR teleoperator - convert string constants to enums
        from dex_retargeting.constants import RobotName, RetargetingType, HandType
        
        # Convert string robot name to enum
        robot_name_map = {
            "xhand_left": RobotName.xhand,
            "xhand_right": RobotName.xhand,
            "xhand": RobotName.xhand
        }
        robot_name_enum = robot_name_map.get(config.hand_robot_name, RobotName.xhand)
        
        # Convert string retargeting type to enum
        retargeting_map = {
            "vector": RetargetingType.vector,
            "dexpilot": RetargetingType.dexpilot
        }
        retargeting_enum = retargeting_map.get(config.hand_retargeting_type, RetargetingType.dexpilot)
        
        # Convert string hand type to enum
        hand_type_map = {
            "left": HandType.left,
            "right": HandType.right
        }
        hand_type_enum = hand_type_map.get(config.hand_type, HandType.right)
        
        hand_config = XHandVRTeleoperatorConfig(
            robot_name=robot_name_enum,
            retargeting_type=retargeting_enum,
            hand_type=hand_type_enum,
            vr_tcp_port=config.vr_tcp_port,  # Same port - shared VR
            setup_adb=False,  # Arm teleop handles ADB setup
            vr_verbose=config.vr_verbose,
            control_frequency=config.hand_control_frequency,
            smoothing_alpha=config.hand_smoothing_alpha
        )
        self.hand_teleop = XHandVRTeleoperator(hand_config)
        
        # State tracking
        self._is_connected = False
        self._robot_reference = None
        
        logger.info("FrankaFERXHandVRTeleoperator initialized")
    
    @property
    def action_features(self) -> Dict[str, type]:
        """Combined action features with proper prefixes."""
        features = {}
        
        # Add arm action features with 'arm_' prefix
        arm_features = self.arm_teleop.action_features
        for key, value in arm_features.items():
            features[f"arm_{key}"] = value
            
        # Add hand action features with 'hand_' prefix
        hand_features = self.hand_teleop.action_features
        for key, value in hand_features.items():
            features[f"hand_{key}"] = value
        
        return features
    
    @property
    def feedback_features(self) -> Dict[str, type]:
        """Combined feedback features (empty for VR teleoperators)."""
        return {}
    
    @property
    def is_connected(self) -> bool:
        """Check if both teleoperators are connected."""
        return self._is_connected and self.arm_teleop.is_connected and self.hand_teleop.is_connected
    
    @property
    def is_calibrated(self) -> bool:
        """VR teleoperators don't require calibration."""
        return True
    
    def connect(self, calibrate: bool = True) -> None:
        """Connect both arm and hand VR teleoperators."""
        if self._is_connected:
            raise RuntimeError("FrankaFERXHandVRTeleoperator is already connected")
        
        logger.info("Connecting dual VR teleoperator...")
        
        try:
            # Connect arm teleoperator first (handles ADB setup)
            logger.info("Connecting arm VR teleoperator...")
            self.arm_teleop.connect(calibrate=calibrate)
            
            # Connect hand teleoperator (will use shared VR router)
            logger.info("Connecting hand VR teleoperator...")
            self.hand_teleop.connect(calibrate=calibrate)
            
            self._is_connected = True
            logger.info("Dual VR teleoperator connected successfully")
            
        except Exception as e:
            logger.error(f"Failed to connect dual VR teleoperator: {e}")
            # Try to disconnect any connected components
            try:
                if self.arm_teleop.is_connected:
                    self.arm_teleop.disconnect()
                if self.hand_teleop.is_connected:
                    self.hand_teleop.disconnect()
            except:
                pass
            raise ConnectionError(f"Failed to connect dual VR teleoperator: {e}")
    
    def disconnect(self) -> None:
        """Disconnect both arm and hand VR teleoperators."""
        if not self._is_connected:
            return
        
        logger.info("Disconnecting dual VR teleoperator...")
        
        # Disconnect both teleoperators
        try:
            self.hand_teleop.disconnect()
        except Exception as e:
            logger.error(f"Error disconnecting hand teleoperator: {e}")
        
        try:
            self.arm_teleop.disconnect() 
        except Exception as e:
            logger.error(f"Error disconnecting arm teleoperator: {e}")
        
        self._is_connected = False
        self._robot_reference = None
        logger.info("Dual VR teleoperator disconnected")
    
    def set_robot(self, robot):
        """Set robot reference for both teleoperators."""
        self._robot_reference = robot
        
        # The composite robot has arm and hand sub-robots
        if hasattr(robot, 'arm') and hasattr(robot, 'hand'):
            self.arm_teleop.set_robot(robot.arm)
            # Hand teleoperator doesn't need robot reference for VR control
            logger.info("Robot references set for dual VR teleoperator")
        else:
            logger.warning("Robot doesn't have expected 'arm' and 'hand' attributes")
    
    def get_action(self) -> Dict[str, Any]:
        """Get combined action from both arm and hand VR teleoperators."""
        if not self._is_connected:
            raise RuntimeError("Dual VR teleoperator not connected")
        
        action = {}
        
        try:
            # Get arm action and add 'arm_' prefix
            arm_action = self.arm_teleop.get_action()
            for key, value in arm_action.items():
                action[f"arm_{key}"] = value
            
            # Get hand action and add 'hand_' prefix
            hand_action = self.hand_teleop.get_action()
            for key, value in hand_action.items():
                action[f"hand_{key}"] = value
            
            return action
            
        except Exception as e:
            logger.error(f"Error getting dual VR action: {e}")
            # Return safe default actions
            safe_action = {}
            # Arm: current positions (will be handled by individual teleoperators)
            for i in range(7):
                safe_action[f"arm_joint_{i}.pos"] = 0.0
            # Hand: open position
            for i in range(12):
                safe_action[f"hand_joint_{i}.pos"] = 0.0
            return safe_action
    
    def calibrate(self) -> None:
        """Calibrate teleoperators (no-op for VR)."""
        pass
    
    def configure(self) -> None:
        """Configure teleoperators."""
        pass
    
    def send_feedback(self, feedback: Dict[str, Any]) -> None:
        """Send feedback to teleoperators (no-op for VR)."""
        pass
    
    def get_status(self) -> Dict[str, Any]:
        """Get combined status from both teleoperators."""
        status = {
            "connected": self._is_connected,
            "arm_status": {},
            "hand_status": {}
        }
        
        if self._is_connected:
            try:
                status["arm_status"] = self.arm_teleop.get_status()
                status["hand_status"] = {"connected": self.hand_teleop.is_connected}
            except Exception as e:
                logger.error(f"Error getting teleoperator status: {e}")
        
        return status
    
    def reset_initial_pose(self) -> bool:
        """Reset VR initial pose for arm teleoperator."""
        if not self._is_connected:
            return False
        
        # Only arm teleoperator has pose reset functionality
        if hasattr(self.arm_teleop, 'reset_initial_pose'):
            return self.arm_teleop.reset_initial_pose()
        
        return True