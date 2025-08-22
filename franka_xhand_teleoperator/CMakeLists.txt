# DEPRECATED: Use setup.py instead for pip-installable package
# This CMakeLists.txt is kept for reference but setup.py is now the preferred build method
cmake_minimum_required(VERSION 3.12)
project(franka_xhand_teleoperator)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Find required packages
find_package(pybind11 REQUIRED)
find_package(Eigen3 REQUIRED)

# Add the include directories
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include)
include_directories(${EIGEN3_INCLUDE_DIR})

# Common source files
set(COMMON_SOURCES
    src/weighted_ik.cpp
    src/geofik.cpp
)

# VR Message Router module
pybind11_add_module(vr_message_router 
    src/vr_message_router.cpp
)

target_compile_features(vr_message_router PRIVATE cxx_std_17)
set_target_properties(vr_message_router PROPERTIES
    CXX_VISIBILITY_PRESET "hidden"
    VISIBILITY_INLINES_HIDDEN ON
    LIBRARY_OUTPUT_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/build"
)

# Weighted IK Bridge module
pybind11_add_module(weighted_ik_bridge 
    src/weighted_ik_bridge.cpp
    ${COMMON_SOURCES}
)

target_compile_features(weighted_ik_bridge PRIVATE cxx_std_17)
target_link_libraries(weighted_ik_bridge PRIVATE Eigen3::Eigen)
set_target_properties(weighted_ik_bridge PROPERTIES
    CXX_VISIBILITY_PRESET "hidden"
    VISIBILITY_INLINES_HIDDEN ON
    LIBRARY_OUTPUT_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/build"
)

# VR IK Bridge module removed - using vr_message_router instead