#!/usr/bin/env python3
"""
Arm IK Processor - Direct wrapper for WeightedIKSolver
Processes VR wrist data into Franka joint targets using direct IK solving
"""

import logging
import numpy as np
from typing import Dict, Any, Optional, List
import sys
import time
from pathlib import Path

import weighted_ik_bridge

logger = logging.getLogger(__name__)

class ArmIKProcessor:
    """
    Direct interface to WeightedIKSolver for processing VR wrist data.
    Converts VR wrist poses to Franka joint targets using IK.
    """
    
    def __init__(self, config=None):
        """Initialize the arm IK processor"""        
        self.WeightedIKSolver = weighted_ik_bridge.WeightedIKSolver
        
        # Configuration
        self.config = config or {}
        self.verbose = self.config.get('verbose', False)
        self.smoothing_factor = self.config.get('smoothing_factor', 0.7)
        self.movement_scale = self.config.get('movement_scale', 2.0)  # Scale VR movements by 2x
        
        # IK solver instance (will be created in setup)
        self.ik_solver = None
        
        # State tracking
        self.is_initialized = False
        self.initial_robot_pose = None  # 4x4 transformation matrix
        self.initial_vr_pose = None     # Initial VR pose for differential calculation
        self.vr_initialized = False
        self.q7_min = -2.89  # Default Franka Q7 limits
        self.q7_max = 2.89
        self.last_target_joints = None
        
        logger.info("Arm IK processor initialized")
    
    def setup(self, neutral_pose: List[float], initial_robot_pose: List[float], 
              manipulability_weight: float = 1.0, neutral_distance_weight: float = 2.0, 
              current_distance_weight: float = 2.0, joint_weights: Optional[List[float]] = None,
              q7_min: float = -2.89, q7_max: float = 2.89):
        """
        Setup the IK solver with robot state
        
        Args:
            neutral_pose: 7-DOF neutral joint positions
            initial_robot_pose: 16-element transformation matrix (4x4 flattened)
            manipulability_weight: IK weight for manipulability
            neutral_distance_weight: IK weight for neutral pose distance
            current_distance_weight: IK weight for current pose distance  
            joint_weights: 7-element joint weights
            q7_min, q7_max: Q7 joint limits
        """
        if joint_weights is None:
            joint_weights = [2.0, 2.0, 1.0, 1.0, 1.0, 1.0, 1.0]
        
        try:
            # Create WeightedIKSolver instance
            self.ik_solver = self.WeightedIKSolver(
                neutral_pose=neutral_pose,
                weight_manip=manipulability_weight,
                weight_neutral=neutral_distance_weight,
                weight_current=current_distance_weight,
                joint_weights=joint_weights,
                verbose=self.verbose
            )
            
            # Store initial robot pose (4x4 transformation matrix)
            self.initial_robot_pose = np.array(initial_robot_pose).reshape(4, 4)
            
            # Store Q7 limits
            self.q7_min = q7_min
            self.q7_max = q7_max
            
            self.is_initialized = True
            
            if self.verbose:
                logger.info("Arm IK processor setup completed")
                logger.info(f"Q7 limits: [{q7_min:.3f}, {q7_max:.3f}]")
            return True
            
        except Exception as e:
            logger.error(f"Failed to setup arm IK processor: {e}")
            return False
    
    def process_wrist_data(self, wrist_data, current_joints: List[float]) -> Dict[str, float]:
        """
        Process VR wrist data directly into Franka joint targets
        
        Args:
            wrist_data: VRWristData object with position, quaternion, fist_state
            current_joints: Current 7-DOF joint positions
            
        Returns:
            Dict[str, float]: Arm joint actions with 'arm_' prefix
        """
        if not self.is_initialized:
            logger.warning("Arm IK processor not initialized")
            return {f"arm_joint_{i}.pos": float(current_joints[i]) for i in range(7)}
        
        if not wrist_data.valid:
            if self.verbose:
                logger.debug("Invalid wrist data, returning current joints")
            return {f"arm_joint_{i}.pos": float(current_joints[i]) for i in range(7)}
        
        # Check if VR is initialized, if not, handle initialization
        if not self.vr_initialized:
            # Initialize VR (this will be done in _compute_target_pose)
            # For first frame, return current joints to avoid big jump
            try:
                self._compute_target_pose(wrist_data, current_joints)  # This will initialize VR
            except:
                pass  # Ignore any return from initialization
            return {f"arm_joint_{i}.pos": float(current_joints[i]) for i in range(7)}
        
        try:
            # Convert VR wrist pose to target pose relative to initial robot pose
            start_time = time.perf_counter()
            target_position, target_orientation = self._compute_target_pose(wrist_data, current_joints)
            pose_compute_time = time.perf_counter() - start_time
            
            # Debug: Log IK solver inputs
            if self.verbose:
                logger.info(f"IK SOLVER INPUT - target_pos: {target_position}")
                logger.info(f"IK SOLVER INPUT - current_joints: {[f'{j:.3f}' for j in current_joints]}")
            
            # Solve IK using WeightedIKSolver
            ik_start_time = time.perf_counter()
            ik_result = self.ik_solver.solve_q7_optimized(
                target_position=target_position,
                target_orientation=target_orientation,
                current_pose=current_joints,
                q7_min=self.q7_min,
                q7_max=self.q7_max,
                tolerance=1e-6,
                max_iterations=100
            )
            ik_solve_time = time.perf_counter() - ik_start_time
            
            if ik_result.success:
                target_joints = list(ik_result.joint_angles)
                
                # Apply smoothing if we have previous targets
                if self.last_target_joints is not None:
                    target_joints = self._apply_smoothing(target_joints, self.last_target_joints)
                
                self.last_target_joints = target_joints
                
                if self.verbose:
                    total_time = pose_compute_time + ik_solve_time
                    logger.debug(f"IK timing: pose={pose_compute_time*1000:.2f}ms, solve={ik_solve_time*1000:.2f}ms, total={total_time*1000:.2f}ms")
            else:
                # IK failed, return current joints
                target_joints = current_joints
                if self.verbose:
                    logger.debug("IK failed, using current joints")
            
            # Convert to action format with arm prefix
            arm_action = {}
            for i in range(7):
                arm_action[f"arm_joint_{i}.pos"] = float(target_joints[i])
            
            return arm_action
            
        except Exception as e:
            logger.error(f"Error in arm IK processing: {e}")
            # Return current joints as safe fallback
            return {f"arm_joint_{i}.pos": float(current_joints[i]) for i in range(7)}
    
    def _compute_target_pose(self, wrist_data, current_joints):
        """
        Convert VR wrist data to target pose using differential approach
        
        Args:
            wrist_data: VRWristData with position [x, y, z] and quaternion [x, y, z, w]
            
        Returns:
            tuple: (target_position, target_orientation)
                - target_position: [x, y, z] 
                - target_orientation: 9-element rotation matrix (3x3 flattened)
        """
        # Transform VR coordinates to robot frame (like original)
        # VR: +x=right, +y=up, +z=forward → Robot: +x=forward, +y=left, +z=up
        robot_position = np.array([
            wrist_data.position[2],   # Robot X = VR Z (forward)
            -wrist_data.position[0],  # Robot Y = -VR X (left = -right)
            wrist_data.position[1]    # Robot Z = VR Y (up)
        ])
        
        # Transform quaternion to robot frame
        robot_quaternion = self._transform_vr_quaternion_to_robot(wrist_data.quaternion)
        
        # Initialize VR pose if first time (store TRANSFORMED coordinates like original)
        if not self.vr_initialized:
            self.initial_vr_pose = {
                'position': robot_position,    # Store robot-frame position
                'quaternion': robot_quaternion # Store robot-frame quaternion [x, y, z, w]
            }
            self.vr_initialized = True
            logger.info(f"VR INITIALIZED at robot position: {robot_position}")
            # Return zero delta for initialization frame
            target_position = self.initial_robot_pose.flatten()[12:15]  # Current robot position [12,13,14]
            target_rotation_matrix = self.initial_robot_pose[:3, :3]  # Current robot orientation
            return target_position.tolist(), target_rotation_matrix.flatten().tolist()
        
        # Debug: Confirm we're past initialization
        if self.verbose:
            logger.debug(f"VR processing (initialized): current={robot_position}, initial={self.initial_vr_pose['position']}")
        
        # Calculate delta in robot frame (like original line 239-243)
        robot_pos_delta = robot_position - self.initial_vr_pose['position']
        
        # Debug: Print delta calculation
        if self.verbose:
            logger.info(f"VR current: {robot_position}")
            logger.info(f"VR initial: {self.initial_vr_pose['position']}")
            logger.info(f"Robot delta: {robot_pos_delta} (norm: {np.linalg.norm(robot_pos_delta):.4f}m)")
        
        # Apply workspace limits (75cm max offset)
        max_offset = 0.75
        if np.linalg.norm(robot_pos_delta) > max_offset:
            robot_pos_delta = robot_pos_delta / np.linalg.norm(robot_pos_delta) * max_offset
        
        # Calculate target position: initial robot + robot delta
        # Note: FrankaFER stores pose in row-major format as 16-element array
        # The 4x4 transformation matrix in row-major has translation at indices [12,13,14]
        initial_robot_translation = self.initial_robot_pose.flatten()[12:15]  # Extract translation [12,13,14]
        target_position = initial_robot_translation + robot_pos_delta
        
        # Debug: Always log the final delta being applied
        delta_norm = np.linalg.norm(robot_pos_delta)
        if self.verbose:
            logger.info(f"FINAL robot delta applied: {robot_pos_delta} (norm: {delta_norm:.4f}m)")
        
        # Calculate orientation delta (in robot frame, like original)
        initial_robot_quat_xyzw = self.initial_vr_pose['quaternion']  # [x, y, z, w] in robot frame
        current_robot_quat_xyzw = robot_quaternion                    # [x, y, z, w] in robot frame
        
        # Convert to [w, x, y, z] for easier quaternion math
        initial_quat_wxyz = np.array([initial_robot_quat_xyzw[3], initial_robot_quat_xyzw[0], initial_robot_quat_xyzw[1], initial_robot_quat_xyzw[2]])
        current_quat_wxyz = np.array([current_robot_quat_xyzw[3], current_robot_quat_xyzw[0], current_robot_quat_xyzw[1], current_robot_quat_xyzw[2]])
        
        # Calculate quaternion delta in robot frame: current * initial^-1 (like original)
        robot_quat_delta_wxyz = self._quaternion_multiply(current_quat_wxyz, self._quaternion_inverse(initial_quat_wxyz))
        
        # Get initial robot orientation as quaternion
        initial_robot_rot_matrix = self.initial_robot_pose[:3, :3]
        initial_robot_quat = self._rotation_matrix_to_quaternion(initial_robot_rot_matrix)
        
        # Apply delta: robot_delta * initial_robot_orientation (like original)
        target_quat = self._quaternion_multiply(robot_quat_delta_wxyz, initial_robot_quat)
        target_quat = target_quat / np.linalg.norm(target_quat)  # Normalize
        
        # Convert target quaternion back to rotation matrix
        target_rotation_matrix = self._quaternion_to_rotation_matrix(target_quat)
        
        return target_position.tolist(), target_rotation_matrix.flatten().tolist()
    
    def _transform_vr_quaternion_to_robot(self, vr_quaternion):
        """
        Transform VR quaternion to robot frame following original logic
        
        Args:
            vr_quaternion: [x, y, z, w] in VR frame
            
        Returns:
            np.array: [x, y, z, w] in robot frame
        """
        # Convert VR quaternion to rotation matrix
        qx, qy, qz, qw = vr_quaternion
        vr_matrix = np.array([
            [1 - 2*(qy*qy + qz*qz), 2*(qx*qy - qz*qw), 2*(qx*qz + qy*qw)],
            [2*(qx*qy + qz*qw), 1 - 2*(qx*qx + qz*qz), 2*(qy*qz - qx*qw)],
            [2*(qx*qz - qy*qw), 2*(qy*qz + qx*qw), 1 - 2*(qx*qx + qy*qy)]
        ])
        
        # Define coordinate transformation matrix from VR to robot
        # VR: [right, up, forward] → Robot: [forward, left, up]  
        transform_matrix = np.array([
            [0,  0,  1],   # Robot X = VR Z (forward)
            [-1, 0,  0],   # Robot Y = -VR X (left = -right)  
            [0,  1,  0]    # Robot Z = VR Y (up)
        ])
        
        # Apply transformation: R_robot = T * R_vr * T^-1
        robot_matrix = transform_matrix @ vr_matrix @ transform_matrix.T
        
        # Convert back to quaternion
        robot_quat_wxyz = self._rotation_matrix_to_quaternion(robot_matrix)
        
        # Return in [x, y, z, w] format
        return np.array([robot_quat_wxyz[1], robot_quat_wxyz[2], robot_quat_wxyz[3], robot_quat_wxyz[0]])
    
    def _quaternion_multiply(self, q1, q2):
        """Multiply two quaternions [w, x, y, z]"""
        w1, x1, y1, z1 = q1
        w2, x2, y2, z2 = q2
        return np.array([
            w1*w2 - x1*x2 - y1*y2 - z1*z2,
            w1*x2 + x1*w2 + y1*z2 - z1*y2,
            w1*y2 - x1*z2 + y1*w2 + z1*x2,
            w1*z2 + x1*y2 - y1*x2 + z1*w2
        ])
    
    def _quaternion_inverse(self, q):
        """Inverse of quaternion [w, x, y, z]"""
        w, x, y, z = q
        norm_sq = w*w + x*x + y*y + z*z
        return np.array([w, -x, -y, -z]) / norm_sq
    
    def _rotation_matrix_to_quaternion(self, R):
        """Convert 3x3 rotation matrix to quaternion [w, x, y, z]"""
        trace = np.trace(R)
        if trace > 0:
            s = np.sqrt(trace + 1.0) * 2
            w = 0.25 * s
            x = (R[2, 1] - R[1, 2]) / s
            y = (R[0, 2] - R[2, 0]) / s
            z = (R[1, 0] - R[0, 1]) / s
        else:
            if R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
                s = np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2]) * 2
                w = (R[2, 1] - R[1, 2]) / s
                x = 0.25 * s
                y = (R[0, 1] + R[1, 0]) / s
                z = (R[0, 2] + R[2, 0]) / s
            elif R[1, 1] > R[2, 2]:
                s = np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2]) * 2
                w = (R[0, 2] - R[2, 0]) / s
                x = (R[0, 1] + R[1, 0]) / s
                y = 0.25 * s
                z = (R[1, 2] + R[2, 1]) / s
            else:
                s = np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1]) * 2
                w = (R[1, 0] - R[0, 1]) / s
                x = (R[0, 2] + R[2, 0]) / s
                y = (R[1, 2] + R[2, 1]) / s
                z = 0.25 * s
        return np.array([w, x, y, z])
    
    def _quaternion_to_rotation_matrix(self, q):
        """Convert quaternion [w, x, y, z] to 3x3 rotation matrix"""
        w, x, y, z = q
        return np.array([
            [1 - 2*(y*y + z*z), 2*(x*y - z*w), 2*(x*z + y*w)],
            [2*(x*y + z*w), 1 - 2*(x*x + z*z), 2*(y*z - x*w)],
            [2*(x*z - y*w), 2*(y*z + x*w), 1 - 2*(x*x + y*y)]
        ])
    
    def _apply_smoothing(self, current: List[float], previous: List[float]) -> List[float]:
        """Apply exponential smoothing to joint targets"""
        alpha = 1.0 - self.smoothing_factor  # Convert to smoothing weight
        return [alpha * c + (1 - alpha) * p for c, p in zip(current, previous)]
    
    def set_q7_limits(self, q7_min: float, q7_max: float):
        """Update Q7 joint limits"""
        self.q7_min = q7_min
        self.q7_max = q7_max
        if self.verbose:
            logger.info(f"Q7 limits updated to [{q7_min:.3f}, {q7_max:.3f}]")
    
    def is_ready(self) -> bool:
        """Check if processor is ready"""
        return self.is_initialized and self.ik_solver is not None
    
    def get_status(self) -> Dict[str, Any]:
        """Get processor status"""
        return {
            "initialized": self.is_initialized,
            "ik_ready": self.ik_solver is not None,
            "has_initial_pose": self.initial_robot_pose is not None,
            "q7_limits": [self.q7_min, self.q7_max]
        }