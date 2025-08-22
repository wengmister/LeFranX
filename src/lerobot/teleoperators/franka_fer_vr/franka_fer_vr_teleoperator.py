#!/usr/bin/env python3
"""
Franka FER VR Teleoperator for LeRobot using C++ IK bridge.

This module provides a VR teleoperator for real-time control of Franka FER robots
using hand tracking data. It integrates with LeRobot's teleoperator framework
and includes automatic ADB setup for Meta Quest devices.
"""

import logging
from typing import Dict, Any

from lerobot.teleoperators.teleoperator import Teleoperator
from .config_franka_fer_vr import FrankaFERVRTeleoperatorConfig
from ..vr_router_manager import get_vr_router_manager, VRRouterConfig

logger = logging.getLogger(__name__)


class FrankaFERVRTeleoperator(Teleoperator):
    """
    VR Teleoperator for real-time Franka FER robot control using hand tracking.
    
    This teleoperator receives VR hand pose data via TCP and uses a C++ weighted IK solver
    to compute target joint positions for the Franka FER robot. It's designed to work seamlessly
    with LeRobot's data collection and policy deployment pipeline.
    
    Features:
    - Real-time VR hand tracking via TCP
    - Weighted IK solver optimized for Franka robots
    - Configurable workspace limits and smoothing
    - Support for BiDexHand and other end effectors
    - Automatic adb port forwarding setup for Android VR apps using adb_setup.py
    
    The data flow is:
    1. VR app sends hand pose data via TCP to vr_message_router
    2. VR messages are processed through arm_ik_processor for joint target computation
    3. Joint positions are returned as robot actions
    
    Requires:
    - vr_message_router C++ module built and available (from franka_xhand_teleoperator)
    - VR application sending hand pose data to configured TCP port
    - ADB reverse port forwarding set up for Meta Quest devices (optional, auto-configured)
    
    Usage:
        config = FrankaFERVRTeleoperatorConfig(tcp_port=8000, verbose=True)
        teleop = FrankaFERVRTeleoperator(config)
        
        # In your data collection loop:
        action = teleop.get_action()
    """
    
    # Set class attributes required by Teleoperator
    config_class = FrankaFERVRTeleoperatorConfig
    name = "franka_fer_vr"
    
    def __init__(self, config: FrankaFERVRTeleoperatorConfig):
        """
        Initialize the Franka FER VR teleoperator.
        
        Args:
            config: Configuration object containing TCP port, IK parameters, etc.
        """
        super().__init__(config)
        self.config = config
        
        # Get shared VR router manager
        self.vr_manager = get_vr_router_manager()
        
        # Import arm IK processor
        from .arm_ik_processor import ArmIKProcessor
        
        # Initialize IK processor
        ik_config = {
            'verbose': config.verbose,
            'smoothing_factor': config.smoothing_factor
        }
        self.arm_ik_processor = ArmIKProcessor(ik_config)
        
        # State tracking
        self._robot_reference = None
        self._initialized = False
        self._is_connected = False
        self._last_action = None
        
        logger.info(f"FrankaFERVRTeleoperator initialized with TCP port {config.tcp_port}")
    
    def connect(self, calibrate: bool = True) -> None:  # pylint: disable=unused-argument
        """
        Connect to VR input and setup ADB if configured.
        
        This method:
        1. Sets up ADB reverse port forwarding if enabled
        2. Starts the VR TCP server for receiving hand tracking data
        3. Prepares the teleoperator for action generation
        
        Args:
            calibrate: Unused parameter for compatibility with base class
        """
        if self._is_connected:
            raise RuntimeError("FrankaFERVRTeleoperator is already connected")
        
        # Register with shared VR router manager
        vr_config = VRRouterConfig(
            tcp_port=self.config.tcp_port,
            verbose=self.config.verbose,
            message_timeout_ms=1000.0,  # Increased to handle VR app periodic delays
            setup_adb=self.config.setup_adb
        )
        
        success = self.vr_manager.register_teleoperator(vr_config, "franka_fer_vr")
        if not success:
            raise ConnectionError(f"Failed to register with VR router manager on port {self.config.tcp_port}")
        
        # Wait for robot to be connected to get initial state
        if self._robot_reference and hasattr(self._robot_reference, 'is_connected') and not self._robot_reference.is_connected:
            logger.warning("Robot is not connected yet. IK solver will be initialized when robot connects.")
        else:
            # Try to initialize IK solver if robot is available and connected
            if self._robot_reference:
                self._initialize_ik_solver()
            else:
                logger.info("Robot reference not yet available. IK solver will be initialized on first get_action call.")
        
        self._is_connected = True
        logger.info("FrankaFERVRTeleoperator connected successfully")
    
    def disconnect(self) -> None:
        """
        Disconnect from VR input and cleanup resources.
        
        This method:
        1. Stops the VR TCP server
        2. Cleans up ADB reverse port forwarding if it was set up
        3. Resets internal state
        """
        if not self._is_connected:
            return
        
        # Unregister from shared VR router manager
        self.vr_manager.unregister_teleoperator("franka_fer_vr")
        
        self._is_connected = False
        self._initialized = False
        self._robot_reference = None
        
        logger.info("FrankaFERVRTeleoperator disconnected")
    
    def set_robot(self, robot):
        """
        Set the robot reference for IK processing.
        
        The teleoperator needs access to the robot's current state to:
        1. Get current joint positions for IK optimization
        2. Get end-effector pose for proper workspace mapping
        
        Args:
            robot: Robot instance that implements get_observation() method
        """
        self._robot_reference = robot
        logger.info(f"Robot reference set: {robot.__class__.__name__}")
    
    def get_action(self) -> Dict[str, Any]:
        """
        Get the current action from VR input.
        
        Returns:
            Dictionary with joint position targets: {"joint_0.pos": float, ...}
            
        Raises:
            RuntimeError: If teleoperator is not connected
            ValueError: If invalid data is received from VR
        """
        if not self._is_connected:
            raise RuntimeError("FrankaFERVRTeleoperator is not connected")
        
        # The robot reference should be available from initialization  
        if self._robot_reference is None:
            logger.error("Robot reference not available - VR teleoperator requires robot connection")
            # Return last known action or zero position as fallback
            if self._last_action is not None:
                return self._last_action
            else:
                return {f"joint_{i}.pos": 0.0 for i in range(7)}
        
        # Initialize IK solver if not done yet (delayed initialization)
        if not self._initialized:
            if not self._initialize_ik_solver():
                # Return current robot position if initialization fails
                if self._last_action is not None:
                    return self._last_action
                else:
                    # Emergency fallback
                    return {f"joint_{i}.pos": 0.0 for i in range(7)}
        
        # Get current robot joint positions
        try:
            current_obs = self._robot_reference.get_observation()
            current_joints = [current_obs[f"joint_{i}.pos"] for i in range(7)]
        except Exception as e:
            logger.error(f"Failed to get robot observation: {e}")
            # Return last known action or zero position
            if self._last_action is not None:
                return self._last_action
            else:
                return {f"joint_{i}.pos": 0.0 for i in range(7)}
        
        # Get VR wrist data from shared manager and process through IK
        try:
            # Get VR wrist data from shared manager
            wrist_data, status = self.vr_manager.get_wrist_data()
            
            if not status.get('tcp_connected', False) or wrist_data is None:
                # No valid VR data, return current position to hold
                action = {f"joint_{i}.pos": float(current_joints[i]) for i in range(7)}
                return action
            
            # Process VR wrist data through IK
            arm_action = self.arm_ik_processor.process_wrist_data(
                wrist_data, 
                current_joints
            )
            
            # Convert action format from arm_joint_X.pos to joint_X.pos
            action = {}
            for i in range(7):
                action[f"joint_{i}.pos"] = arm_action[f"arm_joint_{i}.pos"]
            
            # Store as last known good action
            self._last_action = action
            
            return action
            
        except ValueError as e:
            logger.warning(f"Invalid VR data format: {e}")
            # Return current position to hold in place
            return {f"joint_{i}.pos": float(current_joints[i]) for i in range(7)}
        except Exception as e:
            logger.error(f"Unexpected error getting VR action: {e}")
            # Return current position to hold in place
            return {f"joint_{i}.pos": float(current_joints[i]) for i in range(7)}
    
    def get_status(self) -> Dict[str, Any]:
        """
        Get current status of VR teleoperator.
        
        Returns:
            Dictionary containing connection status, initialization state,
            ADB setup status, and VR connection info
        """
        status = {
            "connected": self._is_connected,
            "initialized": self._initialized,
            "vr_connected": False,
            "vr_ready": False
        }
        
        if self._is_connected:
            try:
                vr_status = self.vr_manager.get_status()
                status.update(vr_status)
                status["vr_ready"] = vr_status.get("tcp_connected", False)
            except Exception as e:
                logger.error(f"Failed to get VR status: {e}")
        
        return status
    
    def _initialize_ik_solver(self) -> bool:
        """
        Initialize the IK solver with robot's current state.
        
        This method:
        1. Gets current robot joint positions and end-effector pose
        2. Sets up the IK solver with proper weights and constraints
        3. Configures Q7 limits and workspace parameters
        
        Returns:
            True if initialization succeeded, False otherwise
        """
        if self._robot_reference is None:
            logger.error("No robot reference available for IK initialization")
            return False
        
        try:
            # Get current robot state
            current_obs = self._robot_reference.get_observation()
            current_joints = [current_obs[f"joint_{i}.pos"] for i in range(7)]
            
            # Get initial robot pose (end-effector transformation matrix)
            # Debug: Check what observation keys are available
            logger.info(f"Available observation keys: {list(current_obs.keys())}")
            
            # FrankaFER stores ee_pose as ee_pose.00 through ee_pose.15
            ee_pose_keys = [f"ee_pose.{i:02d}" for i in range(16)]
            if all(key in current_obs for key in ee_pose_keys):
                ee_pose = [current_obs[key] for key in ee_pose_keys]
                logger.info(f"Using robot ee_pose: {ee_pose[:4]} (first row)")
                initial_robot_pose = ee_pose
            elif "ee_pose" in current_obs:
                # Fallback for other robots that might have a single ee_pose key
                ee_pose = current_obs["ee_pose"]
                if len(ee_pose) == 16:  # 4x4 transformation matrix
                    logger.info(f"Using robot ee_pose: {ee_pose[:4]} (first row)")
                    initial_robot_pose = ee_pose
                else:
                    logger.warning("End-effector pose format not recognized, using identity")
                    # Use identity transformation as fallback
                    initial_robot_pose = [1,0,0,0, 0,1,0,0, 0,0,1,0, 0,0,0,1]
            else:
                logger.warning("No end-effector pose in observation, using identity matrix!")
                logger.warning("This will cause IK targets to be relative to [0,0,0] instead of robot position!")
                initial_robot_pose = [1,0,0,0, 0,1,0,0, 0,0,1,0, 0,0,0,1]
            
            # Setup arm IK processor with current robot state
            success = self.arm_ik_processor.setup(
                neutral_pose=current_joints,
                initial_robot_pose=initial_robot_pose,
                manipulability_weight=self.config.manipulability_weight,
                neutral_distance_weight=self.config.neutral_distance_weight,
                current_distance_weight=self.config.current_distance_weight,
                joint_weights=self.config.joint_weights,
                q7_min=self.config.q7_min,
                q7_max=self.config.q7_max
            )
            
            if not success:
                logger.error("Failed to setup arm IK processor")
                return False
            
            self._initialized = True
            logger.info("IK solver initialized successfully")
            return True
            
        except Exception as e:
            logger.error(f"Failed to initialize IK solver: {e}")
            return False
    
    def reset_initial_pose(self) -> bool:
        """Reset the VR initial pose reference to current robot position."""
        if not self._is_connected or not self._initialized:
            logger.warning("VR teleoperator not connected or initialized, cannot reset pose")
            return False
            
        try:
            # Re-initialize the IK solver with current robot position as new reference
            success = self._initialize_ik_solver()
            if success:
                logger.info("VR initial pose reset successfully to current robot position")
            else:
                logger.error("Failed to reset VR initial pose")
            return success
            
        except Exception as e:
            logger.error(f"Failed to reset VR initial pose: {e}")
            return False
    
    # Abstract method implementations
    @property
    def action_features(self) -> dict:
        """Return action features for 7-DOF Franka robot."""
        return {f"joint_{i}.pos": float for i in range(7)}
    
    @property
    def feedback_features(self) -> dict:
        """Return feedback features - VR teleoperator doesn't use feedback."""
        return {}
    
    @property
    def is_connected(self) -> bool:
        """Whether the VR teleoperator is connected."""
        return self._is_connected
    
    @property
    def is_calibrated(self) -> bool:
        """VR teleoperator doesn't require calibration."""
        return True
    
    def calibrate(self) -> None:
        """VR teleoperator doesn't require calibration."""
        pass
    
    def configure(self) -> None:
        """Apply configuration to VR teleoperator."""
        # Configuration is handled in connect()
        pass
    
    def send_feedback(self, feedback: Dict[str, Any]) -> None:  # pylint: disable=unused-argument
        """VR teleoperator doesn't support feedback."""
        pass