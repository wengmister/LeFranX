#!/bin/bash

# Build script for Franka Server

# Check for optional libfranka path argument
FRANKA_PATH=""
if [ $# -ge 1 ]; then
    FRANKA_PATH="-DFRANKA_INSTALL_PATH=$1"
    echo "Using custom libfranka path: $1"
fi

echo "Building Franka Server..."

# Create build directory if it doesn't exist
mkdir -p build
cd build

# Configure with CMake
cmake .. -DCMAKE_BUILD_TYPE=Release $FRANKA_PATH

# Build
make -j$(nproc)

echo "Build complete!"
echo ""
echo "Executables:"
echo "  ./build/franka_server <robot_ip>      - Franka control server"