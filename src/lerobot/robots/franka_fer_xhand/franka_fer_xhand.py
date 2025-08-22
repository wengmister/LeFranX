import logging
import time
from functools import cached_property
from typing import Any, Dict

from lerobot.cameras.utils import make_cameras_from_configs
from lerobot.errors import DeviceAlreadyConnectedError, DeviceNotConnectedError
from lerobot.robots.franka_fer import FrankaFER
from lerobot.robots.robot import Robot
from lerobot.robots.xhand import XHand

from .franka_fer_xhand_config import FrankaFERXHandConfig

logger = logging.getLogger(__name__)


class FrankaFERXHand(Robot):
    """
    Composite robot combining Franka FER arm with XHand end-effector.
    
    Provides unified interface for:
    - Arm control (7 DOF Franka FER)  
    - Hand control (12 DOF XHand)
    - Synchronized observations and actions
    """
    
    config_class = FrankaFERXHandConfig
    name = "franka_fer_xhand"
    
    def __init__(self, config: FrankaFERXHandConfig):
        super().__init__(config)
        self.config = config
        
        # Initialize sub-robots
        self.arm = FrankaFER(config.arm_config)
        self.hand = XHand(config.hand_config)
        
        # Initialize cameras for data collection
        self.cameras = make_cameras_from_configs(config.cameras)
        
        # Track connection state
        self._is_connected = False
    
    @cached_property
    def observation_features(self) -> Dict[str, type]:
        """Combined observation features from arm, hand, and cameras"""
        features = {}
        
        # Add arm features with 'arm_' prefix
        arm_features = self.arm.observation_features
        for key, value in arm_features.items():
            # Skip arm cameras since we use our own
            if not key.startswith(('camera', 'cam')):
                features[f"arm_{key}"] = value
        
        # Add hand features with 'hand_' prefix
        hand_features = self.hand.observation_features  
        for key, value in hand_features.items():
            # Skip hand cameras since we use our own
            if not key.startswith(('camera', 'cam')):
                features[f"hand_{key}"] = value
        
        # Add composite robot cameras
        for cam_name, cam_config in self.config.cameras.items():
            features[cam_name] = (cam_config.height, cam_config.width, 3)
        
        return features
    
    @cached_property
    def action_features(self) -> Dict[str, type]:
        """Combined action features from arm and hand"""
        features = {}
        
        # Add arm action features with 'arm_' prefix
        arm_actions = self.arm.action_features
        for key, value in arm_actions.items():
            features[f"arm_{key}"] = value
            
        # Add hand action features with 'hand_' prefix  
        hand_actions = self.hand.action_features
        for key, value in hand_actions.items():
            features[f"hand_{key}"] = value
        
        return features
    
    @property
    def is_connected(self) -> bool:
        """Check if both arm and hand are connected"""
        return self._is_connected and self.arm.is_connected and self.hand.is_connected
    
    def connect(self, calibrate: bool = True) -> None:
        """Connect to both arm and hand robots"""
        if self.is_connected:
            raise DeviceAlreadyConnectedError(f"{self} already connected")
        
        logger.info("Connecting Franka FER + XHand composite robot...")
        
        try:
            # Connect arm first
            logger.info("Connecting Franka FER arm...")
            self.arm.connect(calibrate=calibrate)
            
            # Connect hand
            logger.info("Connecting XHand...")
            self.hand.connect(calibrate=calibrate)
            
            # Connect cameras
            logger.info("Connecting cameras...")
            for cam in self.cameras.values():
                cam.connect()
            
            self._is_connected = True
            logger.info(f"{self} connected successfully")
            
        except Exception as e:
            logger.error(f"Failed to connect composite robot: {e}")
            # Try to disconnect any connected components
            try:
                if self.arm.is_connected:
                    self.arm.disconnect()
                if self.hand.is_connected:
                    self.hand.disconnect()
                for cam in self.cameras.values():
                    if cam.is_connected:
                        cam.disconnect()
            except:
                pass
            raise ConnectionError(f"Failed to connect Franka FER + XHand: {e}")
    
    @property
    def is_calibrated(self) -> bool:
        """Check if both arm and hand are calibrated"""
        return self.arm.is_calibrated and self.hand.is_calibrated
    
    def configure(self) -> None:
        """Configure both arm and hand"""
        if not self.is_connected:
            raise DeviceNotConnectedError(f"{self} is not connected")
        
        logger.info("Configuring composite robot...")
        # Note: Individual robots configure themselves during connect()
        # This method can be used for any composite-specific configuration
        logger.info("Composite robot configuration completed")
    
    def calibrate(self) -> None:
        """Calibrate both arm and hand"""
        if not self.is_connected:
            raise DeviceNotConnectedError(f"{self} is not connected")
        
        logger.info("Calibrating composite robot...")
        self.arm.calibrate()
        self.hand.calibrate()
        logger.info("Composite robot calibration completed")
    
    def get_observation(self) -> Dict[str, Any]:
        """Get combined observations from arm and hand"""
        if not self.is_connected:
            raise DeviceNotConnectedError(f"{self} is not connected")
        
        obs_dict = {}
        
        # Get arm observations with 'arm_' prefix
        start = time.perf_counter()
        arm_obs = self.arm.get_observation()
        arm_time = time.perf_counter() - start
        
        for key, value in arm_obs.items():
            # Skip arm cameras since we use our own
            if not key.startswith(('camera', 'cam')):
                obs_dict[f"arm_{key}"] = value
        
        # Get hand observations with 'hand_' prefix
        start = time.perf_counter()
        hand_obs = self.hand.get_observation()
        hand_time = time.perf_counter() - start
        
        for key, value in hand_obs.items():
            # Skip hand cameras since we use our own
            if not key.startswith(('camera', 'cam')):
                obs_dict[f"hand_{key}"] = value
        
        # Get camera observations
        start = time.perf_counter()
        for cam_name, cam in self.cameras.items():
            obs_dict[cam_name] = cam.read()
        cam_time = time.perf_counter() - start
        
        logger.debug(f"Arm obs: {arm_time*1000:.1f}ms, Hand obs: {hand_time*1000:.1f}ms, Cameras: {cam_time*1000:.1f}ms")
        
        return obs_dict
    
    def send_action(self, action: Dict[str, Any]) -> Dict[str, Any]:
        """Send combined actions to arm and hand"""
        if not self.is_connected:
            raise DeviceNotConnectedError(f"{self} is not connected")
        
        # Split action into arm and hand components
        arm_action = {}
        hand_action = {}
        
        for key, value in action.items():
            if key.startswith("arm_"):
                arm_key = key[4:]  # Remove 'arm_' prefix
                arm_action[arm_key] = value
            elif key.startswith("hand_"):
                hand_key = key[5:]  # Remove 'hand_' prefix  
                hand_action[hand_key] = value
            else:
                logger.warning(f"Unknown action key: {key} (should start with 'arm_' or 'hand_')")
        
        performed_action = {}
        
        if self.config.synchronize_actions:
            # Send actions simultaneously
            try:
                start = time.perf_counter()
                
                # TODO: Could implement true parallel execution with threading
                # For now, send sequentially but quickly
                if arm_action:
                    arm_result = self.arm.send_action(arm_action)
                    for key, value in arm_result.items():
                        performed_action[f"arm_{key}"] = value
                
                if hand_action:
                    hand_result = self.hand.send_action(hand_action)
                    for key, value in hand_result.items():
                        performed_action[f"hand_{key}"] = value
                
                action_time = time.perf_counter() - start
                logger.debug(f"Synchronized action time: {action_time*1000:.1f}ms")
                
            except Exception as e:
                logger.error(f"Error in synchronized action: {e}")
                if self.config.emergency_stop_both:
                    self.stop()
                raise
        else:
            # Send actions independently
            if arm_action:
                arm_result = self.arm.send_action(arm_action)
                for key, value in arm_result.items():
                    performed_action[f"arm_{key}"] = value
                    
            if hand_action:
                hand_result = self.hand.send_action(hand_action)  
                for key, value in hand_result.items():
                    performed_action[f"hand_{key}"] = value
        
        return performed_action
    
    def disconnect(self) -> None:
        """Disconnect both arm and hand"""
        if not self.is_connected:
            raise DeviceNotConnectedError(f"{self} is not connected")
        
        logger.info("Disconnecting composite robot...")
        
        # Disconnect both components (continue even if one fails)
        arm_success = True
        hand_success = True
        
        try:
            self.arm.disconnect()
        except Exception as e:
            logger.error(f"Failed to disconnect arm: {e}")
            arm_success = False
        
        try:
            self.hand.disconnect()
        except Exception as e:
            logger.error(f"Failed to disconnect hand: {e}")
            hand_success = False
        
        # Disconnect cameras
        try:
            for cam in self.cameras.values():
                cam.disconnect()
        except Exception as e:
            logger.error(f"Failed to disconnect cameras: {e}")
        
        self._is_connected = False
        
        if arm_success and hand_success:
            logger.info(f"{self} disconnected successfully")
        else:
            logger.warning(f"{self} disconnected with errors")
    
    def reset_to_home(self) -> bool:
        """Reset both arm and hand to home positions"""
        if not self.is_connected:
            return False
        
        logger.info("Resetting composite robot to home position...")
        
        # Reset both components
        arm_success = self.arm.reset_to_home()
        hand_success = self.hand.reset_to_home()
        
        success = arm_success and hand_success
        
        if success:
            logger.info("Composite robot reset to home position")
        else:
            logger.warning("Composite robot reset completed with errors")
        
        return success
    
    def stop(self) -> bool:
        """Emergency stop both arm and hand"""
        if not self.is_connected:
            return False
        
        logger.info("Emergency stop - composite robot")
        
        # Stop both components
        arm_success = self.arm.stop()
        hand_success = self.hand.stop()
        
        return arm_success and hand_success
    
    def recover_from_errors(self) -> bool:
        """Recover from errors on both arm and hand"""
        if not self.is_connected:
            return False
        
        logger.info("Recovering from errors - composite robot")
        
        # Attempt recovery on both components
        arm_success = self.arm.recover_from_errors()
        hand_success = self.hand.recover_from_errors()
        
        return arm_success and hand_success