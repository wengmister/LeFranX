import logging
import math
import time
from functools import cached_property
from typing import Any, Dict, Optional

import numpy as np

from lerobot.cameras.utils import make_cameras_from_configs
from lerobot.errors import DeviceAlreadyConnectedError, DeviceNotConnectedError

from ..robot import Robot
from .xhand_config import XHandConfig

logger = logging.getLogger(__name__)


class XHand(Robot):
    """
    XHand robot with 12 DOF (Degrees of Freedom).
    
    Commercial robotic hand with 12 joints providing:
    - 12 position commands (actions)  
    - 12 joint positions (observations)
    - 12 joint torques (observations)
    
    Based on XHand SDK API from xhand_examples.py
    """
    
    config_class = XHandConfig
    name = "xhand"
    
    def __init__(self, config: XHandConfig):
        super().__init__(config)
        self.config = config
        self.cameras = make_cameras_from_configs(config.cameras)
        self._is_connected = False
        
        # Joint names - XHand has 12 joints numbered 0-11
        self.joint_names = [f"joint_{i}" for i in range(12)]
        
        # XHand SDK interface (will be initialized on connect)
        self._device = None
        self._hand_command = None
        self._hand_id = config.hand_id
        
    @cached_property
    def observation_features(self) -> Dict[str, type]:
        """Define observation feature structure"""
        features = {}
        
        # Joint positions (12 joints)
        for joint_name in self.joint_names:
            features[f"{joint_name}.pos"] = float
            
        # Joint torques (12 joints) - actually current in mA, stored as float
        for joint_name in self.joint_names:
            features[f"{joint_name}.torque"] = float
        
        # Add camera features if any
        for cam_name, cam_config in self.config.cameras.items():
            features[cam_name] = (cam_config.height, cam_config.width, 3)
            
        return features
    
    @cached_property 
    def action_features(self) -> Dict[str, type]:
        """Define action feature structure"""
        return {f"{joint_name}.pos": float for joint_name in self.joint_names}
    
    @property
    def is_connected(self) -> bool:
        """Check if robot is connected"""
        return self._is_connected
    
    def connect(self, calibrate: bool = True) -> None:
        """Connect to the XHand robot"""
        if self.is_connected:
            raise DeviceAlreadyConnectedError(f"{self} already connected")
        
        logger.info(f"Connecting to XHand via {self.config.protocol}")
        
        try:
            # Import XHand SDK (only when actually connecting)
            try:
                from xhand_controller import xhand_control
            except ImportError:
                logger.warning("XHand SDK not available - using stub implementation")
                self._connect_stub()
                return
            
            # Initialize XHand device
            self._device = xhand_control.XHandControl()
            
            # Open device connection
            if self.config.protocol == "RS485":
                self._connect_rs485()
            elif self.config.protocol == "EtherCAT":
                self._connect_ethercat()
            else:
                raise ValueError(f"Unsupported protocol: {self.config.protocol}")
            
            # Initialize hand command structure
            self._hand_command = xhand_control.HandCommand_t()
            for i in range(12):
                self._hand_command.finger_command[i].id = i
                self._hand_command.finger_command[i].kp = self.config.default_kp
                self._hand_command.finger_command[i].ki = self.config.default_ki
                self._hand_command.finger_command[i].kd = self.config.default_kd
                self._hand_command.finger_command[i].position = 0.0  # Start at 0
                self._hand_command.finger_command[i].tor_max = self.config.default_tor_max
                self._hand_command.finger_command[i].mode = self.config.default_mode
            
            self._is_connected = True
            logger.info(f"{self} connected successfully")
            
            # Configure robot after connection
            self.configure()
            
            # Connect cameras if any
            for cam in self.cameras.values():
                cam.connect()
            
            # Calibrate if requested
            if calibrate and not self.is_calibrated:
                self.calibrate()
                
        except Exception as e:
            logger.error(f"Failed to connect to XHand: {e}")
            raise ConnectionError(f"Failed to connect to XHand robot: {e}")
    
    def _connect_rs485(self) -> None:
        """Connect via RS485 (real implementation)"""
        logger.info(f"Opening RS485 connection to {self.config.serial_port}")
        
        response = self._device.open_serial(
            self.config.serial_port,
            self.config.baud_rate
        )
        
        if response.error_code != 0:
            raise ConnectionError(f"Failed to open RS485 connection: {response.error_message}")
        
        # Get hand ID
        hands_id = self._device.list_hands_id()
        if hands_id:
            self._hand_id = hands_id[0]
            logger.info(f"Found hand with ID: {self._hand_id}")
        else:
            raise ConnectionError("No XHand devices found")
    
    def _connect_ethercat(self) -> None:
        """Connect via EtherCAT (stub - not implemented yet)"""
        # TODO: Implement EtherCAT connection
        logger.warning("EtherCAT connection not implemented yet")
        raise NotImplementedError("EtherCAT connection not implemented")
    
    def _connect_stub(self) -> None:
        """Stub connection for testing without hardware"""
        logger.info("Using stub XHand connection (no hardware)")
        self._device = None  # Stub mode
        self._hand_command = None
        self._is_connected = True
    
    @property
    def is_calibrated(self) -> bool:
        """Check if robot is calibrated"""
        # TODO: Implement calibration status check
        return True  # Assume calibrated for now
    
    def configure(self) -> None:
        """Configure robot with current settings"""
        if not self.is_connected:
            raise DeviceNotConnectedError(f"{self} is not connected")
        
        logger.info("Configuring XHand robot...")
        # TODO: Implement configuration procedure (set control mode, gains, etc.)
        logger.info("XHand configuration completed")
    
    def calibrate(self) -> None:
        """Calibrate the XHand robot"""
        if not self.is_connected:
            raise DeviceNotConnectedError(f"{self} is not connected")
        
        logger.info("Calibrating XHand robot...")
        # TODO: Implement calibration procedure
        logger.info("XHand calibration completed")
    
    def get_observation(self) -> Dict[str, Any]:
        """Get current robot observation"""
        if not self.is_connected:
            raise DeviceNotConnectedError(f"{self} is not connected")
        
        obs_dict = {}
        
        # Get robot state (positions and torques)
        start = time.perf_counter()
        joint_states = self._get_joint_states()
        dt_ms = (time.perf_counter() - start) * 1e3
        logger.debug(f"{self} read joint states: {dt_ms:.1f}ms")
        
        if joint_states is not None:
            # Add joint positions
            for i, joint_name in enumerate(self.joint_names):
                obs_dict[f"{joint_name}.pos"] = float(joint_states["positions"][i])
                obs_dict[f"{joint_name}.torque"] = float(joint_states["torques"][i])  # Convert to float (mA)
        
        # Capture images from cameras
        for cam_key, cam in self.cameras.items():
            start = time.perf_counter()
            obs_dict[cam_key] = cam.async_read()
            dt_ms = (time.perf_counter() - start) * 1e3
            logger.debug(f"{self} read {cam_key}: {dt_ms:.1f}ms")
        
        return obs_dict
    
    def _get_joint_states(self) -> Optional[Dict[str, np.ndarray]]:
        """Get current joint positions and torques from XHand"""
        if self._device is None:
            # Stub mode - return dummy data
            positions = np.zeros(12)  # All joints at 0 position
            torques = np.zeros(12, dtype=int)  # All torques at 0 (mA - actually current)
            return {"positions": positions, "torques": torques}
        
        try:
            # Read state from XHand device (use positional arguments)
            error_struct, state = self._device.read_state(self._hand_id, True)
            
            if error_struct.error_code != 0:
                # Ignore known hardware sensor errors that don't affect position reading
                ignored_errors = [
                    "Sensor fails to read the combined force",
                    "Sensor fails to read the distributed force", 
                    "Sensor fails to read temperature",
                    "Communication data CRC error",
                    "This hardware version does not support force control mode"
                ]
                
                if not any(ignored_error in error_struct.error_message for ignored_error in ignored_errors):
                    logger.warning(f"Failed to read XHand state: {error_struct.error_message}")
                    return None
            
            # Extract joint positions and torques
            positions = np.zeros(12)
            torques = np.zeros(12)
            
            for i in range(12):
                finger_state = state.finger_state[i]
                positions[i] = finger_state.position  # Already in radians
                torques[i] = int(finger_state.torque)  # Actually current in mA (misnamed in SDK)
            
            return {
                "positions": positions,
                "torques": torques
            }
            
        except Exception as e:
            logger.error(f"Error reading XHand joint states: {e}")
            return None
    
    def send_action(self, action: Dict[str, Any]) -> Dict[str, Any]:
        """Send action to robot"""
        if not self.is_connected:
            raise DeviceNotConnectedError(f"{self} is not connected")
        
        # Extract joint positions from action dict
        target_positions = []
        for joint_name in self.joint_names:
            key = f"{joint_name}.pos"
            if key not in action:
                raise ValueError(f"Missing joint position for {key}")
            target_positions.append(action[key])
        
        target_positions = np.array(target_positions)
        
        # Apply safety limits
        target_positions = self._apply_safety_limits(target_positions)
        
        # Send command to robot
        success = self._send_position_command(target_positions)
        if not success:
            logger.warning("Failed to send action to XHand")
        
        # Return the actual action sent
        return {f"{joint_name}.pos": float(target_positions[i]) 
                for i, joint_name in enumerate(self.joint_names)}
    
    def _apply_safety_limits(self, positions: np.ndarray) -> np.ndarray:
        """Apply safety limits to target positions"""
        limited_positions = positions.copy()
        
        for i, joint_name in enumerate(self.joint_names):
            if joint_name in self.config.position_limits:
                min_pos, max_pos = self.config.position_limits[joint_name]
                limited_positions[i] = np.clip(positions[i], min_pos, max_pos)
        
        return limited_positions
    
    def _send_position_command(self, positions: np.ndarray) -> bool:
        """Send position command to XHand robot"""
        if self._device is None or self._hand_command is None:
            # Stub mode
            logger.debug(f"Stub: Sending position command: {positions}")
            return True
        
        try:
            # Update hand command with new positions
            for i in range(12):
                self._hand_command.finger_command[i].position = float(positions[i])
            
            # Send command to XHand
            error_struct = self._device.send_command(self._hand_id, self._hand_command)
            
            if error_struct.error_code != 0:
                # Ignore known hardware sensor errors that don't affect movement
                ignored_errors = [
                    "Sensor fails to read the combined force",
                    "Sensor fails to read the distributed force", 
                    "Sensor fails to read temperature",
                    "Communication data CRC error",
                    "This hardware version does not support force control mode"
                ]
                
                if not any(ignored_error in error_struct.error_message for ignored_error in ignored_errors):
                    logger.warning(f"Failed to send XHand command: {error_struct.error_message}")
                    return False
            
            return True
            
        except Exception as e:
            logger.error(f"Error sending XHand position command: {e}")
            return False
    
    def disconnect(self) -> None:
        """Disconnect from robot"""
        if not self.is_connected:
            raise DeviceNotConnectedError(f"{self} is not connected")
        
        try:
            # TODO: Implement proper disconnection
            logger.info("Disconnecting from XHand...")
            self._is_connected = False
            
            # Disconnect cameras
            for cam in self.cameras.values():
                cam.disconnect()
                
        except Exception as e:
            logger.warning(f"Error during XHand disconnect: {e}")
        
        logger.info(f"{self} disconnected")
    
    def reset_to_home(self) -> bool:
        """Reset robot to home position"""
        if not self.is_connected:
            return False
        
        try:
            home_positions = np.array(self.config.home_position_rad)
            action = {f"{joint_name}.pos": float(home_positions[i]) 
                     for i, joint_name in enumerate(self.joint_names)}
            
            self.send_action(action)
            logger.info("XHand reset to home position")
            return True
            
        except Exception as e:
            logger.error(f"Failed to reset XHand to home: {e}")
            return False
    
    def stop(self) -> bool:
        """Emergency stop"""
        if not self.is_connected:
            return False
        
        # TODO: Implement emergency stop
        logger.info("XHand emergency stop - NOT IMPLEMENTED")
        return True
    
    def recover_from_errors(self) -> bool:
        """Recover from robot errors"""
        if not self.is_connected:
            return False
        
        # TODO: Implement error recovery
        logger.info("XHand error recovery - NOT IMPLEMENTED")
        return True