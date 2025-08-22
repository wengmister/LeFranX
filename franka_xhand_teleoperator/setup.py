from setuptools import setup
import subprocess
import sys
import os

from pybind11.setup_helpers import Pybind11Extension, build_ext
import pybind11

# Check if Eigen3 is available
def find_eigen():
    """Find Eigen3 installation"""
    try:
        # Try pkg-config first
        result = subprocess.run(['pkg-config', '--cflags', 'eigen3'], 
                              capture_output=True, text=True)
        if result.returncode == 0:
            include_dir = result.stdout.strip().replace('-I', '')
            return [include_dir]
    except:
        pass
    
    # Common Eigen installation paths
    common_paths = [
        '/usr/include/eigen3',
        '/usr/local/include/eigen3',
        '/opt/homebrew/include/eigen3',  # macOS Homebrew
        '/usr/include'  # Sometimes Eigen is directly in /usr/include
    ]
    
    for path in common_paths:
        if os.path.exists(os.path.join(path, 'Eigen')):
            return [path]
    
    print("Warning: Eigen3 not found in common locations")
    return []

# Common include directories
common_include_dirs = [
    pybind11.get_include(),
    "include/",
    *find_eigen()
]

# Common source files
common_sources = [
    "src/weighted_ik.cpp", 
    "src/geofik.cpp"
]

# Define the extension modules
ext_modules = [
    # Weighted IK Bridge module
    Pybind11Extension(
        "weighted_ik_bridge",
        [
            "src/weighted_ik_bridge.cpp",
            *common_sources
        ],
        include_dirs=common_include_dirs,
        language='c++'
    ),
    
    # VR Message Router module
    Pybind11Extension(
        "vr_message_router",
        [
            "src/vr_message_router.cpp"
        ],
        include_dirs=common_include_dirs,
        language='c++'
    ),
]

setup(
    name="franka_xhand_teleoperator",
    version="0.1.0",
    description="Franka XHand Teleoperator with VR support for LeRobot",
    author="LeRobot Team",
    ext_modules=ext_modules,
    cmdclass={"build_ext": build_ext},
    zip_safe=False,
    python_requires=">=3.7",
    install_requires=[
        "pybind11>=2.6.0",
        "numpy",
    ]
)