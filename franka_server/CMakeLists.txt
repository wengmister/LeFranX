cmake_minimum_required(VERSION 3.10)
project(franka_server)

# Set C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Set optimization level to Release
set(CMAKE_BUILD_TYPE Release)
set(CMAKE_CXX_FLAGS_RELEASE "-O3 -DNDEBUG")

# Set CMAKE_PREFIX_PATH to find your libfranka installation
# You can override this by passing -DFRANKA_INSTALL_PATH=/path/to/franka/install
set(FRANKA_INSTALL_PATH "$ENV{HOME}/ws/franka/install" CACHE PATH "Path to libfranka installation directory")
set(CMAKE_PREFIX_PATH "${CMAKE_PREFIX_PATH};${FRANKA_INSTALL_PATH}")

# Find required packages
find_package(Franka REQUIRED)
find_package(Threads REQUIRED)
find_package(Eigen3 REQUIRED)
find_package(ruckig REQUIRED)

# Franka Server executable
add_executable(franka_server 
    src/franka_server.cpp
    src/examples_common.cpp
)

# Link libraries
target_link_libraries(franka_server
    Franka::Franka
    Threads::Threads
    ruckig::ruckig
    ${EIGEN3_LIBRARIES}
)

# Include directories
target_include_directories(franka_server PRIVATE 
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${EIGEN3_INCLUDE_DIR}
)

# Optional: Set runtime path for shared libraries
set_target_properties(franka_server PROPERTIES
    INSTALL_RPATH_USE_LINK_PATH TRUE
)