import logging
import socket
import time
from functools import cached_property
from typing import Any, Optional

import numpy as np

from lerobot.cameras.utils import make_cameras_from_configs
from lerobot.errors import DeviceAlreadyConnectedError, DeviceNotConnectedError

from ..robot import Robot
from ..utils import ensure_safe_goal_position
from .franka_fer_config import FrankaFERConfig

logger = logging.getLogger(__name__)


class FrankaFER(Robot):
    """
    Franka FER robot controlled via direct socket communication to C++ server.
    
    This robot implementation communicates directly with a C++ position server running on a real-time
    computer to control a Franka Emika robot.
    """
    
    config_class = FrankaFERConfig
    name = "franka_fer"
    
    def __init__(self, config: FrankaFERConfig):
        super().__init__(config)
        self.config = config
        self.server_ip = config.server_ip
        self.server_port = config.server_port
        self.socket = None
        self.cameras = make_cameras_from_configs(config.cameras)
        self._is_connected = False
        self._vr_teleoperator = None  # Reference to VR teleoperator for pose reset
        
    @cached_property
    def observation_features(self) -> dict[str, type | tuple]:
        """Define observation feature structure"""
        features = {}
        
        # Joint positions (7 joints)
        for i in range(7):
            features[f"joint_{i}.pos"] = float
            
        # Joint velocities (7 joints)
        for i in range(7):
            features[f"joint_{i}.vel"] = float
            
        # End-effector pose (16-element affine transformation matrix, row-major)
        for i in range(16):
            features[f"ee_pose.{i:02d}"] = float
            
        # Add camera features if any
        for cam_name, cam_config in self.config.cameras.items():
            features[cam_name] = (cam_config.height, cam_config.width, 3)
            
        return features
    
    @cached_property 
    def action_features(self) -> dict[str, type]:
        """Define action feature structure"""
        return {f"joint_{i}.pos": float for i in range(7)}
    
    def _send_command(self, command: str) -> Optional[str]:
        """Send command to server and get response"""
        if not self._is_connected or self.socket is None:
            return None
        
        try:
            self.socket.send((command + "\n").encode())
            response = self.socket.recv(1024).decode().strip()
            return response
        except Exception as e:
            logger.error(f"Communication error: {e}")
            self._is_connected = False
            return None
    
    def _health_check(self) -> bool:
        """Check if server is reachable"""
        try:
            test_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            test_socket.settimeout(2.0)
            result = test_socket.connect_ex((self.server_ip, self.server_port))
            test_socket.close()
            return result == 0
        except Exception as e:
            logger.debug(f"Health check failed: {e}")
            return False
    
    @property
    def is_connected(self) -> bool:
        """Check if robot is connected"""
        return self._is_connected and self.socket is not None
    
    def connect(self, calibrate: bool = True) -> None:
        """Connect to the robot"""
        if self.is_connected:
            raise DeviceAlreadyConnectedError(f"{self} already connected")
        
        # Check server health first
        if not self._health_check():
            raise ConnectionError(f"Cannot reach server at {self.server_ip}:{self.server_port}")
        
        # Connect to server
        try:
            self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.socket.connect((self.server_ip, self.server_port))
            self._is_connected = True
            logger.info(f"Connected to robot server at {self.server_ip}:{self.server_port}")
        except Exception as e:
            logger.error(f"Failed to connect: {e}")
            raise ConnectionError("Failed to connect to Franka robot server")
        
        # Connect cameras if any
        for cam in self.cameras.values():
            cam.connect()
        
        # Configure robot
        self.configure()
        
        logger.info(f"{self} connected")
    
    @property
    def is_calibrated(self) -> bool:
        """Franka robots don't require calibration in this implementation"""
        return True
    
    def calibrate(self) -> None:
        """No-op for Franka robots - they are pre-calibrated"""
        pass
    
    def configure(self) -> None:
        """Configure robot with current settings"""
        if not self.is_connected:
            raise DeviceNotConnectedError(f"{self} is not connected")
        
        # The C++ server applies dynamics factor when it connects to the robot
        # No additional configuration needed here
        pass
    
    def register_vr_teleoperator(self, vr_teleoperator):
        """Register a VR teleoperator for coordinated pose reset"""
        self._vr_teleoperator = vr_teleoperator
        logger.info("VR teleoperator registered with robot for coordinated pose reset")
    
    def _get_joint_positions(self) -> Optional[np.ndarray]:
        """Get current joint positions from server"""
        state = self._get_robot_state()
        if state is not None:
            return state["joint_positions"]
        return None
    
    def _get_robot_state(self) -> Optional[dict]:
        """Get full robot state from server"""
        response = self._send_command("GET_STATE")
        if response and response.startswith("STATE"):
            # Parse: STATE pos0 pos1 ... pos6 vel0 vel1 ... vel6 ee_pose_0 ... ee_pose_15
            parts = response.split()[1:]  # Skip "STATE"
            if len(parts) >= 30:  # 7 positions + 7 velocities + 16 ee_pose elements
                positions = np.array([float(x) for x in parts[:7]])
                velocities = np.array([float(x) for x in parts[7:14]])
                ee_pose = np.array([float(x) for x in parts[14:30]])
                
                return {
                    "joint_positions": positions,
                    "joint_velocities": velocities,
                    "ee_pose": ee_pose
                }
            elif len(parts) >= 14:  # Backwards compatibility - old server without ee_pose
                positions = np.array([float(x) for x in parts[:7]])
                velocities = np.array([float(x) for x in parts[7:14]])
                
                return {
                    "joint_positions": positions,
                    "joint_velocities": velocities,
                    "ee_pose": None
                }
        return None
    
    def get_observation(self) -> dict[str, Any]:
        """Get current robot observation"""
        if not self.is_connected:
            raise DeviceNotConnectedError(f"{self} is not connected")
        
        obs_dict = {}
        
        # Get full robot state including ee_pose
        start = time.perf_counter()
        robot_state = self._get_robot_state()
        if robot_state is not None:
            # Add joint positions
            positions = robot_state["joint_positions"]
            for i, pos in enumerate(positions):
                obs_dict[f"joint_{i}.pos"] = float(pos)
            
            # Add joint velocities if available
            if robot_state["joint_velocities"] is not None:
                velocities = robot_state["joint_velocities"]
                for i, vel in enumerate(velocities):
                    obs_dict[f"joint_{i}.vel"] = float(vel)
            
            # Add end-effector pose if available (as individual float features)
            if robot_state["ee_pose"] is not None:
                ee_pose_flat = robot_state["ee_pose"]
                for i, value in enumerate(ee_pose_flat):
                    obs_dict[f"ee_pose.{i:02d}"] = float(value)
            else:
                # Fallback: identity matrix flattened if ee_pose not available from robot server
                identity_flat = np.eye(4).flatten()
                for i, value in enumerate(identity_flat):
                    obs_dict[f"ee_pose.{i:02d}"] = float(value)
                logger.warning("End-effector pose not available from robot server, using identity matrix")
                
        dt_ms = (time.perf_counter() - start) * 1e3
        logger.debug(f"{self} read robot state: {dt_ms:.1f}ms")
        
        # Capture images from cameras
        for cam_key, cam in self.cameras.items():
            start = time.perf_counter()
            obs_dict[cam_key] = cam.async_read()
            dt_ms = (time.perf_counter() - start) * 1e3
            logger.debug(f"{self} read {cam_key}: {dt_ms:.1f}ms")
        
        return obs_dict
    
    def send_action(self, action: dict[str, Any]) -> dict[str, Any]:
        """Send action to robot"""
        if not self.is_connected:
            raise DeviceNotConnectedError(f"{self} is not connected")
        
        # Extract joint positions from action dict
        joint_positions = []
        for i in range(7):
            key = f"joint_{i}.pos"
            if key not in action:
                raise ValueError(f"Missing joint position for {key}")
            joint_positions.append(action[key])
        
        target_positions = np.array(joint_positions)
        
        # Apply safety limits if configured
        if self.config.max_relative_target is not None:
            current_positions = self._get_joint_positions()
            if current_positions is not None:
                # Create goal_present_pos dict for safety function
                goal_present_pos = {}
                for i in range(7):
                    goal_present_pos[f"joint_{i}"] = (target_positions[i], current_positions[i])
                
                # Apply safety limits
                safe_positions = ensure_safe_goal_position(goal_present_pos, self.config.max_relative_target)
                target_positions = np.array([safe_positions[f"joint_{i}"] for i in range(7)])
        
        # Send command to robot
        cmd = "SET_POSITION " + " ".join(f"{p:.6f}" for p in target_positions)
        response = self._send_command(cmd)
        if response != "OK":
            logger.warning("Failed to send action to robot")
        
        # Return the actual action sent
        return {f"joint_{i}.pos": float(target_positions[i]) for i in range(7)}
    
    def disconnect(self) -> None:
        """Disconnect from robot"""
        if not self.is_connected:
            raise DeviceNotConnectedError(f"{self} is not connected")
        
        # Send disconnect command and close socket
        try:
            self._send_command("DISCONNECT")
            self.socket.close()
        except:
            pass
        
        self.socket = None
        self._is_connected = False
        
        # Disconnect cameras
        for cam in self.cameras.values():
            cam.disconnect()
        
        logger.info(f"{self} disconnected")
    
    def reset_to_home(self) -> bool:
        """Reset robot to home position using MOVE_TO_START"""
        if not self.is_connected:
            return False
        
        home_position = np.array(self.config.home_position)
        
        try:
            # Disconnect current session to allow MOVE_TO_START to execute
            logger.info("Disconnecting to perform safe home movement...")
            self.disconnect()
            
            # Brief pause to ensure clean disconnect
            time.sleep(0.5)
            
            # Reconnect but don't start control loop yet
            logger.info("Reconnecting for home movement...")
            if not self._health_check():
                raise ConnectionError(f"Cannot reach server at {self.server_ip}:{self.server_port}")
            
            self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.socket.connect((self.server_ip, self.server_port))
            self._is_connected = True
            
            # Immediately send MOVE_TO_START before server starts velocity control
            cmd = "MOVE_TO_START " + " ".join(f"{p:.6f}" for p in home_position)
            response = self._send_command(cmd)
            
            # Now wait for the movement to complete and velocity control to start
            time.sleep(3.0)  # Give time for the movement to execute
            
            if response == "OK":
                logger.info("Home position command sent successfully")
                
                # Reconnect cameras after home movement
                for cam in self.cameras.values():
                    cam.connect()
                
                # Reset VR initial pose if VR teleoperator is registered
                if self._vr_teleoperator is not None:
                    try:
                        if hasattr(self._vr_teleoperator, 'reset_initial_pose'):
                            logger.info("Resetting VR initial pose to new robot home position...")
                            if self._vr_teleoperator.reset_initial_pose():
                                logger.info("VR initial pose reset successfully")
                            else:
                                logger.warning("Failed to reset VR initial pose")
                        else:
                            logger.warning("VR teleoperator does not support pose reset")
                    except Exception as e:
                        logger.error(f"Error resetting VR initial pose: {e}")
                
                return True
            else:
                logger.warning(f"Home position command failed: {response}")
                return False
                
        except Exception as e:
            logger.error(f"Failed to reset to home: {e}")
            # Try to reconnect if something went wrong
            try:
                if not self.is_connected:
                    self.connect(calibrate=False)
            except:
                pass
            return False
    
    def stop(self) -> bool:
        """Emergency stop"""
        if not self.is_connected:
            return False
        response = self._send_command("STOP")
        return response == "OK"
    
    def recover_from_errors(self) -> bool:
        """Recover from robot errors"""
        if not self.is_connected:
            return False
        # For now, just try to stop and then check if we can get state
        if self.stop():
            state = self._get_joint_positions()
            return state is not None
        return False