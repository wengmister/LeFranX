"""
VR Hand Detector Adapter - Drop-in replacement for VRHandDetector using VRMessageRouter

This adapter wraps the new VRMessageRouter C++ implementation to provide the same interface
as the original VRHandDetector, enabling seamless integration with existing retargeting code.
"""

import logging
from typing import Optional, Tuple, Any
import numpy as np

import vr_message_router


logger = logging.getLogger(__name__)


OPERATOR2MANO_RIGHT = np.array(
    [
        [0, 0, -1],
        [-1, 0, 0],
        [0, 1, 0],
    ]
)


def adaptive_retargeting_xhand(landmarks):
    """
    Apply adaptive pinky retargeting specifically for XHand robot.
    Compensates for human-to-robot finger length differences by using
    adaptive scaling based on finger extension state.
    
    Args:
        landmarks: np.ndarray of shape (21, 3) with hand landmarks
        
    Returns:
        np.ndarray of shape (21, 3) with retargeted landmarks
    """
    landmarks = landmarks.copy()
    
    pinky_mcp = 17   # PINKY_MCP (base)
    pinky_pip = 18   # PINKY_PIP
    pinky_dip = 19   # PINKY_DIP  
    pinky_tip = 20   # PINKY_TIP
    
    # Adaptive scaling based on finger curl state
    # Calculate finger extension (distance from MCP to TIP)
    pinky_extension = np.linalg.norm(landmarks[pinky_tip] - landmarks[pinky_mcp])
    
    # Scale more when extended (for reaching), less when curled (for fist-making)
    # Tuned specifically for XHand robot kinematics
    max_extension = 0.10
    min_extension = 0.03
    
    # Normalize extension ratio (0.0 = fully curled, 1.0 = fully extended)
    extension_ratio = np.clip((pinky_extension - min_extension) / (max_extension - min_extension), 0.0, 1.0)
    
    # Adaptive scaling: more scaling when extended, less when curled
    base_scale = 1.2   # Minimum scaling for curled positions
    max_scale = 2.2    # Maximum scaling for extended positions
    
    adaptive_scale = base_scale + (max_scale - base_scale) * extension_ratio
    
    # Apply same adaptive scaling to all segments
    mcp_to_pip_scale = adaptive_scale
    pip_to_dip_scale = adaptive_scale  
    dip_to_tip_scale = adaptive_scale
    
    # Apply progressive scaling along kinematic chain
    # Start from MCP (base remains unchanged) and extend each segment
    
    # Extend MCP->PIP segment
    mcp_to_pip_vector = landmarks[pinky_pip] - landmarks[pinky_mcp]
    landmarks[pinky_pip] = landmarks[pinky_mcp] + mcp_to_pip_vector * mcp_to_pip_scale
    
    # Extend PIP->DIP segment (using new PIP position)
    pip_to_dip_vector = landmarks[pinky_dip] - landmarks[pinky_pip]  
    landmarks[pinky_dip] = landmarks[pinky_pip] + pip_to_dip_vector * pip_to_dip_scale
    
    # Extend DIP->TIP segment (using new DIP position)
    dip_to_tip_vector = landmarks[pinky_tip] - landmarks[pinky_dip]
    landmarks[pinky_tip] = landmarks[pinky_dip] + dip_to_tip_vector * dip_to_tip_scale
    
    return landmarks


class VRHandDetectorAdapter:
    """
    Adapter that wraps VRMessageRouter to provide VRHandDetector-compatible interface.
    
    This allows existing hand retargeting code to work with the new VR message routing
    pipeline without modification.
    """
    
    def __init__(self, hand_type: str = "Right", robot_name: str = "xhand", 
                 tcp_port: int = 8000, verbose: bool = False, router: vr_message_router.VRMessageRouter = None):
        """
        Initialize the VR hand detector adapter.
        
        This adapter wraps the VRMessageRouter C++ implementation to provide the same interface
        as the original VRHandDetector, enabling seamless integration with existing retargeting code.
        
        Args:
            hand_type: Hand type ("Right" or "Left") - currently only Right supported
            robot_name: Robot name used for robot-specific retargeting (e.g., "xhand")
            tcp_port: TCP port for VR message router communication
            verbose: Enable verbose logging for debugging
            router: Pre-configured VRMessageRouter instance to use
            
        Raises:
            ImportError: If vr_message_router module is not available
            RuntimeError: If TCP server fails to start
        """
        self.hand_type = hand_type
        self.robot_name = robot_name
        self.tcp_port = tcp_port
        self.verbose = verbose
        self.operator2mano = OPERATOR2MANO_RIGHT
        self.router = router
        
        # Handle router initialization
        if self.router is None:
            # No router provided - adapter will work with external VR manager
            # This is the new mode for shared VR router architecture
            self.router_available = False
            logger.info("VRHandDetectorAdapter initialized for shared VR manager mode")
        else:
            # Legacy mode: manage own router (deprecated)
            try:           
                # Start TCP server
                if not self.router.start_tcp_server():
                    logger.error("Failed to start VR message router TCP server - port may be in use")
                    self.router_available = False
                    raise RuntimeError(f"TCP server failed to bind to port {tcp_port}")
                else:
                    logger.info(f"VR message router started on port {tcp_port}")
                    self.router_available = True
                    
            except ImportError as e:
                logger.error(f"VRMessageRouter not available: {e}. Please build the C++ module.")
                self.router = None
                self.router_available = False
                raise ImportError("vr_message_router module not found - please build the C++ extension")
            except OSError as e:
                logger.error(f"Network error starting VR router: {e}")
                self.router = None
                self.router_available = False
                raise OSError(f"Failed to bind TCP socket to port {tcp_port}: {e}")
            except Exception as e:
                logger.error(f"Unexpected error initializing VR message router: {e}")
                self.router = None
                self.router_available = False
                raise RuntimeError(f"VR message router initialization failed: {e}")
    
    def detect(self) -> Tuple[Any, Optional[np.ndarray], Optional[np.ndarray], Any]:
        """
        Detect hand pose from VR input and apply coordinate transformations.
        
        This method retrieves hand landmarks from the VR message router, applies the same
        coordinate transformations as the original MediaPipe-based VRHandDetector, and
        returns data in the format expected by dex-retargeting.
        
        The transformation pipeline includes:
        1. Scale landmarks to match MediaPipe coordinate range (1.05x)
        2. Flip X-coordinate for right hand to correct mirroring
        3. Center landmarks at wrist (make wrist origin)
        4. Estimate hand orientation using SVD on key points
        5. Transform to MANO coordinate system
        6. Apply robot-specific adaptive retargeting if needed
        
        Returns:
            tuple: (None, joint_pos, keypoint_2d, None) matching VRHandDetector interface
                - joint_pos: numpy array of shape (21, 3) with transformed 3D landmark positions
                  in MANO coordinate system, ready for dex-retargeting
                - keypoint_2d: None (not used by retargeting)
                - Other elements: None (compatibility with original interface)
                
        Raises:
            None: Errors are caught and logged, returns (None, None, None, None) on failure
        """
        if not self.router_available or self.router is None:
            # In shared VR manager mode, this method should not be called directly
            # Use process_landmarks_data() instead with data from VRRouterManager
            if self.verbose:
                logger.debug("detect() called in shared VR manager mode - use process_landmarks_data() instead")
            return None, None, None, None
        
        try:
            # Get messages from router
            messages = self.router.get_messages()
            
            # Check if we have valid landmarks data
            if not messages.landmarks_valid or not messages.landmarks_data.valid:
                if self.verbose:
                    logger.debug("No valid landmarks data available")
                return None, None, None, None
            
            # Convert landmarks to numpy array
            landmarks = messages.landmarks_data.landmarks
            if len(landmarks) == 0:
                if self.verbose:
                    logger.debug("Empty landmarks data")
                return 0, None, None, None
            
            # Convert to numpy array (N, 3) where N should be 21 for hand landmarks
            joint_pos = np.array(landmarks, dtype=np.float32)
            
            if joint_pos.shape[0] != 21:
                logger.warning(f"Invalid landmark count: expected 21, got {joint_pos.shape[0]}. "
                             f"Check VR app hand tracking output format.")
                return None, None, None, None
            
            if joint_pos.shape[1] != 3:
                logger.warning(f"Invalid landmark dimensions: expected 3D coordinates, got shape {joint_pos.shape}. "
                             f"Each landmark should have [x, y, z] coordinates.")
                return None, None, None, None
            
            if self.verbose:
                logger.debug(f"Detected hand landmarks: {joint_pos.shape}")

            # Use consolidated processing method
            joint_pos = self._process_landmarks_internal(joint_pos)
            if joint_pos is None:
                return None, None, None, None
             
            # Return in VRHandDetector format: (None, joint_pos, keypoint_2d, None)
            # keypoint_2d is set to None since retargeting only uses joint_pos
            return None, joint_pos, None, None
            
        except ValueError as e:
            logger.error(f"Data conversion error in VR hand detection: {e}. "
                        f"Check landmark data format from VR app.")
            return None, None, None, None
        except np.linalg.LinAlgError as e:
            logger.error(f"Linear algebra error in hand orientation estimation: {e}. "
                        f"Hand pose may be degenerate or invalid.")
            return None, None, None, None
        except Exception as e:
            logger.error(f"Unexpected error in VR hand detection: {e}")
            return None, None, None, None
    
    def get_wrist_data(self) -> Optional[dict]:
        """
        Get wrist pose data (bonus feature not in original VRHandDetector).
        
        Returns:
            dict with position, quaternion, and fist_state, or None if not available
        """
        if not self.router_available or self.router is None:
            return None
        
        try:
            messages = self.router.get_messages()
            
            if not messages.wrist_valid or not messages.wrist_data.valid:
                return None
            
            wrist_data = messages.wrist_data
            return {
                'position': np.array(wrist_data.position),
                'quaternion': np.array(wrist_data.quaternion),  # [x, y, z, w]
                'fist_state': wrist_data.fist_state
            }
            
        except Exception as e:
            logger.error(f"Error getting wrist data: {e}")
            return None
    
    def get_status(self) -> dict:
        """
        Get router status information.
        
        Returns:
            dict with connection and data validity status
        """
        if not self.router_available or self.router is None:
            return {
                'router_available': False,
                'tcp_connected': False,
                'running': False,
                'wrist_valid': False,
                'landmarks_valid': False
            }
        
        try:
            status = self.router.get_status()
            status['router_available'] = True
            return status
        except Exception as e:
            logger.error(f"Error getting router status: {e}")
            return {'router_available': False}
        
    @staticmethod
    def estimate_frame_from_hand_points(keypoint_3d_array: np.ndarray) -> np.ndarray:
        """
        Compute the 3D coordinate frame (orientation only) from detected hand landmarks.
        
        This method estimates the hand's coordinate frame using Singular Value Decomposition (SVD)
        on key hand points (wrist, index MCP, middle MCP) and applies Gram-Schmidt orthonormalization
        to create a consistent coordinate system aligned with MANO conventions.
        
        Algorithm:
        1. Extract wrist, index MCP, and middle MCP points
        2. Compute initial X vector from wrist to middle MCP
        3. Fit a plane through the three points using SVD
        4. Orthonormalize X and Z vectors using Gram-Schmidt process
        5. Ensure consistent handedness based on index-to-middle direction
        
        Args:
            keypoint_3d_array: Hand landmarks array of shape (21, 3) with landmarks
                              centered at wrist (wrist at origin)
                              
        Returns:
            np.ndarray: 3x3 rotation matrix representing the hand's coordinate frame
                       in MANO convention, where columns are [x_axis, y_axis, z_axis]
                       
        Raises:
            AssertionError: If input array is not shape (21, 3)
        """
        assert keypoint_3d_array.shape == (21, 3)
        points = keypoint_3d_array[[0, 5, 9], :]

        # Compute vector from palm to the first joint of middle finger
        x_vector = points[0] - points[2]

        # Normal fitting with SVD
        points = points - np.mean(points, axis=0, keepdims=True)
        _, _, v = np.linalg.svd(points)

        normal = v[2, :]

        # Gram–Schmidt Orthonormalize
        x = x_vector - np.sum(x_vector * normal) * normal
        x = x / np.linalg.norm(x)
        z = np.cross(x, normal)

        # We assume that the vector from pinky to index is similar the z axis in MANO convention
        if np.sum(z * (points[1] - points[2])) < 0:
            normal *= -1
            z *= -1
        frame = np.stack([x, normal, z], axis=1)
        return frame
    
    def process_landmarks_data(self, landmarks_data) -> Optional[np.ndarray]:
        """
        Process landmarks data from external source (like VR router manager).
        
        Args:
            landmarks_data: Landmarks data object with .landmarks attribute
            
        Returns:
            Processed joint positions as numpy array or None if processing failed
        """
        try:
            if landmarks_data is None or not hasattr(landmarks_data, 'landmarks'):
                return None
                
            landmarks = landmarks_data.landmarks
            if len(landmarks) == 0:
                return None
            
            # Convert to numpy array (N, 3) where N should be 21 for hand landmarks
            joint_pos = np.array(landmarks, dtype=np.float32)
            
            # DEBUG: Print raw VR input
            if self.verbose:
                import time
                timestamp = time.time()
                print(f"  RAW VR LANDMARKS INPUT at {timestamp:.3f}:")
                print(f"  Shape: {joint_pos.shape}")
                print(f"  First 3 landmarks: {joint_pos[:3]}")
                print(f"  Wrist (landmark 0): {joint_pos[0]}")
                print(f"  Index tip (landmark 8): {joint_pos[8] if len(joint_pos) > 8 else 'N/A'}")
                
                # Check for suspicious landmark patterns that might indicate VR data corruption
                wrist_pos = joint_pos[0]
                if np.allclose(wrist_pos, [0, 0, 0], atol=1e-6):
                    logger.warning(f"VR LANDMARKS: Wrist at origin at {timestamp:.3f} - possible data corruption")
                
                # Check if all landmarks are the same (another corruption indicator)
                if len(np.unique(joint_pos.reshape(-1, 3), axis=0)) < 5:
                    logger.warning(f"VR LANDMARKS: Too few unique positions at {timestamp:.3f} - possible data corruption")
            
            # Apply the same processing logic as in detect() method
            # (Copy the landmark processing logic from detect() method)
            
            if joint_pos.shape[0] != 21:
                if self.verbose:
                    logger.warning(f"Expected 21 landmarks, got {joint_pos.shape[0]}")
                return None
            
            # Use consolidated processing method (same as detect())
            return self._process_landmarks_internal(joint_pos)
            
        except Exception as e:
            if self.verbose:
                logger.error(f"Error processing external landmarks data: {e}")
            return None
    
    def _process_landmarks_internal(self, joint_pos: np.ndarray) -> Optional[np.ndarray]:
        """
        Consolidated landmark processing method used by both detect() and process_landmarks_data().
        
        This eliminates code duplication and ensures identical processing in both legacy and new modes.
        """
        try:
            # Apply same transformations as original VRHandDetector
            keypoint_3d_array = joint_pos.copy()
            
            # Scale to match MediaPipe coordinate range (from original)
            keypoint_3d_array *= 1.05
            
            # Convert coordinate system for right hand (from original)
            if self.hand_type == "Right":
                keypoint_3d_array[:, 0] = -keypoint_3d_array[:, 0]
            
            # Make wrist the origin (same as MediaPipe processing)
            keypoint_3d_array = keypoint_3d_array - keypoint_3d_array[0:1, :]

            # Estimate hand orientation using the shifted array
            wrist_rot = self.estimate_frame_from_hand_points(keypoint_3d_array)

            # Transform to MANO coordinate system
            joint_pos = keypoint_3d_array @ wrist_rot @ self.operator2mano
            
            # Apply robot-specific adaptive retargeting
            if "xhand" in self.robot_name.lower():
                joint_pos = adaptive_retargeting_xhand(joint_pos)
            
            return joint_pos
            
        except Exception as e:
            if self.verbose:
                logger.error(f"Error in consolidated landmark processing: {e}")
            return None
    
    def __del__(self):
        """Clean up resources."""
        if hasattr(self, 'router') and self.router is not None:
            try:
                self.router.stop()
            except:
                pass