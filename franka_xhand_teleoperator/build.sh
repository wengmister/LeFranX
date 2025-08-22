#!/bin/bash
# Build script for franka_xhand_teleoperator extensions

echo "🔨 Building franka_xhand_teleoperator extensions..."

# Clean previous builds
rm -rf build/
rm -f src/*.so

# Create build directory
mkdir -p build
cd build

# Configure with CMake
cmake ..

# Build all targets
make -j$(nproc)

cd ..

# Check if builds succeeded
if [ -f build/vr_message_router*.so ] && [ -f build/weighted_ik_bridge*.so ]; then
    echo "✅ All extensions built successfully!"
    echo "📁 Generated files:"
    ls -la src/*.so
else
    echo "❌ Build failed!"
    exit 1
fi