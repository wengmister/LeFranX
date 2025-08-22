#!/bin/bash

# Build script for Franka Velocity Server

echo "Building Franka Velocity Server..."

# Create build directory if it doesn't exist
mkdir -p build
cd build

# Configure with CMake
cmake .. -DCMAKE_BUILD_TYPE=Release

# Build
make -j$(nproc)

echo "Build complete!"
echo ""
echo "Executables:"
echo "  ./build/velocity_server <robot_ip>      - Velocity control server"