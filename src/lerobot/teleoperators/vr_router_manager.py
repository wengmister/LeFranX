#!/usr/bin/env python3
"""
VR Router Manager - Shared VR message router for multiple teleoperators.

This manager ensures that only one VR message router instance exists and can be
shared between multiple teleoperators (arm + hand control) from the same VR source.
"""

import logging
import threading
from typing import Dict, Any, Optional, Tuple
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class VRRouterConfig:
    """Configuration for shared VR router."""
    tcp_port: int = 8000
    verbose: bool = False
    message_timeout_ms: float = 1000.0  # Increased from 200ms to handle VR app periodic delays
    setup_adb: bool = True


class VRRouterManager:
    """
    Singleton manager for shared VR message router.
    
    This class ensures that only one VR TCP server runs and multiple teleoperators
    can access VR data from the same source. It handles:
    - Single VR message router instance
    - Reference counting for proper lifecycle management
    - ADB setup (only once)
    - Data distribution to multiple consumers
    """
    
    _instance: Optional['VRRouterManager'] = None
    _lock = threading.Lock()
    
    def __new__(cls) -> 'VRRouterManager':
        """Singleton pattern implementation."""
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._initialized = False
        return cls._instance
    
    def __init__(self):
        """Initialize the VR router manager (only once)."""
        if self._initialized:
            return
            
        self._initialized = True
        self._vr_router = None
        self._config = None
        self._reference_count = 0
        self._is_started = False
        self._adb_setup_done = False
        self._adb_setup_available = False
        
        # Try to import ADB setup utilities
        try:
            from .adb_setup import setup_adb_reverse, cleanup_adb_reverse
            self.setup_adb_reverse = setup_adb_reverse
            self.cleanup_adb_reverse = cleanup_adb_reverse
            self._adb_setup_available = True
            logger.info("VRRouterManager: ADB setup utilities loaded")
        except ImportError as e:
            logger.warning(f"VRRouterManager: ADB setup not available: {e}")
            self._adb_setup_available = False
        
        logger.info("VRRouterManager initialized (singleton)")
    
    def register_teleoperator(self, config: VRRouterConfig, teleop_name: str) -> bool:
        """
        Register a teleoperator to use the shared VR router.
        
        Args:
            config: VR router configuration
            teleop_name: Name of the teleoperator for logging
            
        Returns:
            True if registration successful, False otherwise
        """
        with self._lock:
            self._reference_count += 1
            logger.info(f"VRRouterManager: Registered {teleop_name} (ref count: {self._reference_count})")
            
            # If this is the first teleoperator, initialize the router
            if self._reference_count == 1:
                return self._initialize_router(config)
            else:
                # Validate config compatibility with existing router
                if self._config and self._config.tcp_port != config.tcp_port:
                    logger.error(f"VRRouterManager: TCP port mismatch! Existing: {self._config.tcp_port}, New: {config.tcp_port}")
                    self._reference_count -= 1
                    return False
                
                logger.info(f"VRRouterManager: {teleop_name} using existing router")
                return self._is_started
    
    def unregister_teleoperator(self, teleop_name: str) -> None:
        """
        Unregister a teleoperator from the shared VR router.
        
        Args:
            teleop_name: Name of the teleoperator for logging
        """
        with self._lock:
            if self._reference_count > 0:
                self._reference_count -= 1
                logger.info(f"VRRouterManager: Unregistered {teleop_name} (ref count: {self._reference_count})")
                
                # If no more teleoperators, shutdown the router
                if self._reference_count == 0:
                    self._shutdown_router()
    
    def _initialize_router(self, config: VRRouterConfig) -> bool:
        """Initialize the VR router with given configuration."""
        try:
            # Import VR message router
            import vr_message_router
            
            # Store config
            self._config = config
            
            # Setup ADB if requested and available
            if config.setup_adb and self._adb_setup_available and not self._adb_setup_done:
                try:
                    success = self.setup_adb_reverse(config.tcp_port)
                    if success:
                        self._adb_setup_done = True
                        logger.info(f"VRRouterManager: ADB reverse setup successful for port {config.tcp_port}")
                    else:
                        logger.warning("VRRouterManager: ADB reverse setup failed")
                except Exception as e:
                    logger.warning(f"VRRouterManager: Error during ADB setup: {e}")
            
            # Create and start VR router
            router_config = vr_message_router.VRRouterConfig()
            router_config.tcp_port = config.tcp_port
            router_config.verbose = config.verbose
            router_config.message_timeout_ms = config.message_timeout_ms
            
            self._vr_router = vr_message_router.VRMessageRouter(router_config)
            
            if self._vr_router.start_tcp_server():
                self._is_started = True
                logger.info(f"VRRouterManager: VR TCP server started on port {config.tcp_port}")
                return True
            else:
                logger.error("VRRouterManager: Failed to start VR TCP server")
                return False
                
        except ImportError as e:
            logger.error(f"VRRouterManager: Failed to import vr_message_router: {e}")
            return False
        except Exception as e:
            logger.error(f"VRRouterManager: Error initializing VR router: {e}")
            return False
    
    def _shutdown_router(self) -> None:
        """Shutdown the VR router and cleanup resources."""
        if self._vr_router:
            try:
                self._vr_router.stop()
                logger.info("VRRouterManager: VR router stopped")
            except Exception as e:
                logger.warning(f"VRRouterManager: Error stopping VR router: {e}")
            
            self._vr_router = None
            self._is_started = False
        
        # Cleanup ADB if we set it up
        if self._adb_setup_done and self._adb_setup_available and self._config:
            try:
                self.cleanup_adb_reverse(self._config.tcp_port)
                logger.info("VRRouterManager: ADB reverse cleanup completed")
            except Exception as e:
                logger.warning(f"VRRouterManager: Error during ADB cleanup: {e}")
            
            self._adb_setup_done = False
        
        self._config = None
    
    def get_vr_data(self) -> Tuple[Optional[Any], Optional[Any], Dict[str, Any]]:
        """
        Get VR data for both arm and hand control.
        
        Returns:
            Tuple of (wrist_data, landmarks_data, status)
            - wrist_data: For arm control (None if not available)
            - landmarks_data: For hand control (None if not available)  
            - status: VR connection status dictionary
        """
        if not self._is_started or not self._vr_router:
            return None, None, {"tcp_connected": False, "error": "VR router not started"}
        
        try:
            messages = self._vr_router.get_messages()
            status = self._vr_router.get_status()
            
            wrist_data = messages.wrist_data if messages.wrist_valid else None
            landmarks_data = messages.landmarks_data if messages.landmarks_valid else None
            
            return wrist_data, landmarks_data, status
            
        except Exception as e:
            logger.error(f"VRRouterManager: Error getting VR data: {e}")
            return None, None, {"tcp_connected": False, "error": str(e)}
    
    def get_wrist_data(self) -> Tuple[Optional[Any], Dict[str, Any]]:
        """
        Get VR wrist data for arm control.
        
        Returns:
            Tuple of (wrist_data, status)
        """
        wrist_data, _, status = self.get_vr_data()
        return wrist_data, status
    
    def get_landmarks_data(self) -> Tuple[Optional[Any], Dict[str, Any]]:
        """
        Get VR landmarks data for hand control.
        
        Returns:
            Tuple of (landmarks_data, status)
        """
        _, landmarks_data, status = self.get_vr_data()
        return landmarks_data, status
    
    def get_status(self) -> Dict[str, Any]:
        """
        Get VR router status.
        
        Returns:
            Dictionary with status information
        """
        status = {
            "manager_initialized": self._initialized,
            "router_started": self._is_started,
            "reference_count": self._reference_count,
            "adb_setup": self._adb_setup_done,
            "tcp_port": self._config.tcp_port if self._config else None
        }
        
        if self._is_started and self._vr_router:
            try:
                vr_status = self._vr_router.get_status()
                status.update(vr_status)
            except Exception as e:
                status["vr_error"] = str(e)
        
        return status
    
    @property
    def is_started(self) -> bool:
        """Whether the VR router is started and ready."""
        return self._is_started
    
    @property
    def reference_count(self) -> int:
        """Number of teleoperators currently using this router."""
        return self._reference_count


# Convenience function to get the singleton instance
def get_vr_router_manager() -> VRRouterManager:
    """Get the singleton VR router manager instance."""
    return VRRouterManager()