// Franka Velocity Control Server with Ruckig Smoothing
// Receives velocity commands via network and executes them smoothly using libfranka + ruckig
#include <cmath>
#include <iostream>
#include <thread>
#include <atomic>
#include <mutex>
#include <array>
#include <chrono>
#include <sys/socket.h>
#include <netinet/in.h>
#include <arpa/inet.h>
#include <unistd.h>
#include <fcntl.h>
#include <fstream>
#include <sstream>
#include <vector>
#include <filesystem>
#include <iomanip>
#include <string>
#include <cstring>
#include <csignal>

#include <franka/exception.h>
#include <franka/robot.h>
#include <Eigen/Dense>
#include <ruckig/ruckig.hpp>

#include "examples_common.h"

struct PositionCommand {
    std::array<double, 7> joint_positions;
    std::chrono::steady_clock::time_point timestamp;
    bool is_valid = false;
};

struct RobotState {
    std::array<double, 7> joint_positions;
    std::array<double, 7> joint_velocities;
    std::array<double, 16> ee_pose;
    std::chrono::steady_clock::time_point timestamp;
};

class FrankaPositionServer {
private:
    // Network communication
    int tcp_server_socket_;
    int client_socket_;
    std::atomic<bool> running_{true};
    std::atomic<bool> client_connected_{false};
    std::atomic<bool> ready_for_clients_{true}; // Controls when network thread accepts connections
    const int PORT = 5000;
    
    // Position command state
    PositionCommand current_position_cmd_;
    std::mutex position_mutex_;
    std::chrono::steady_clock::time_point last_command_time_;
    static constexpr double COMMAND_TIMEOUT_SEC = 0.5; // Stop if no commands for 500ms
    
    // Pending start position state
    std::array<double, 7> pending_start_position_;
    std::atomic<bool> has_pending_start_{false};
    
    // Robot state
    RobotState latest_robot_state_;
    std::mutex state_mutex_;
    
    // Ruckig trajectory smoothing
    std::unique_ptr<ruckig::Ruckig<7>> trajectory_generator_;
    ruckig::InputParameter<7> ruckig_input_;
    ruckig::OutputParameter<7> ruckig_output_;
    bool ruckig_initialized_ = false;
    
    // Robot configuration
    franka::Robot* robot_;
    std::atomic<bool> robot_connected_{false};
    std::string robot_ip_;
    
    // Safety limits for velocity control
    static constexpr std::array<double, 7> MAX_JOINT_VELOCITY = {1.5, 1.5, 1.8, 1.8, 2.0, 2.0, 2.0};
    static constexpr std::array<double, 7> MAX_JOINT_ACCELERATION = {4.0, 4.0, 4.0, 4.0, 6.0, 6.0, 6.0};
    static constexpr std::array<double, 7> MAX_JOINT_JERK = {8.0, 8.0, 8.0, 8.0, 12.0, 12.0, 12.0};
    static constexpr double CONTROL_CYCLE_TIME = 0.001; // 1 kHz
    
public:
    FrankaPositionServer() : robot_(nullptr) {
        setupNetworking();
        
        // Initialize ruckig
        trajectory_generator_ = std::make_unique<ruckig::Ruckig<7>>();
        trajectory_generator_->delta_time = CONTROL_CYCLE_TIME;
        
        // Set up velocity limits for ruckig
        for (size_t i = 0; i < 7; ++i) {
            ruckig_input_.max_velocity[i] = MAX_JOINT_VELOCITY[i];
            ruckig_input_.max_acceleration[i] = MAX_JOINT_ACCELERATION[i];
            ruckig_input_.max_jerk[i] = MAX_JOINT_JERK[i];
            ruckig_input_.target_acceleration[i] = 0.0;
        }
        
        std::cout << "Franka Velocity Server initialized" << std::endl;
    }
    
    ~FrankaPositionServer() {
        running_ = false;
        if (client_socket_ >= 0) close(client_socket_);
        if (tcp_server_socket_ >= 0) close(tcp_server_socket_);
    }
    
private:
    void setupNetworking() {
        tcp_server_socket_ = socket(AF_INET, SOCK_STREAM, 0);
        if (tcp_server_socket_ < 0) {
            throw std::runtime_error("Failed to create TCP socket");
        }
        
        // Allow socket reuse
        int opt = 1;
        setsockopt(tcp_server_socket_, SOL_SOCKET, SO_REUSEADDR, &opt, sizeof(opt));
        
        struct sockaddr_in server_addr;
        server_addr.sin_family = AF_INET;
        server_addr.sin_addr.s_addr = INADDR_ANY;
        server_addr.sin_port = htons(PORT);
        
        if (bind(tcp_server_socket_, (struct sockaddr*)&server_addr, sizeof(server_addr)) < 0) {
            throw std::runtime_error("Failed to bind TCP socket");
        }
        
        if (listen(tcp_server_socket_, 1) < 0) {
            throw std::runtime_error("Failed to listen on TCP socket");
        }
        
        std::cout << "TCP server listening on port " << PORT << std::endl;
    }
    
    void networkThread() {
        while (running_) {
            // Wait until main thread is ready for new clients
            while (running_ && !ready_for_clients_) {
                std::this_thread::sleep_for(std::chrono::milliseconds(50));
            }
            
            if (!running_) break;
            
            std::cout << "Waiting for client connection..." << std::endl;
            
            struct sockaddr_in client_addr;
            socklen_t client_len = sizeof(client_addr);
            
            client_socket_ = accept(tcp_server_socket_, (struct sockaddr*)&client_addr, &client_len);
            if (client_socket_ < 0) {
                if (running_) {
                    std::cerr << "Failed to accept client connection" << std::endl;
                }
                continue;
            }
            
            client_connected_ = true;
            std::cout << "Client connected from " << inet_ntoa(client_addr.sin_addr) << std::endl;
            
            // Handle client communication
            handleClient();
            
            close(client_socket_);
            client_connected_ = false;
            std::cout << "Client disconnected" << std::endl;
        }
    }
    
    void handleClient() {
        char buffer[1024];
        
        while (running_ && client_connected_) {
            ssize_t bytes_received = recv(client_socket_, buffer, sizeof(buffer) - 1, 0);
            
            if (bytes_received <= 0) {
                // Client disconnected or error - immediately stop robot
                std::cout << "Client disconnected - stopping robot motion" << std::endl;
                
                // Send current position as stop command (hold position)
                PositionCommand stop_cmd;
                {
                    std::lock_guard<std::mutex> lock(state_mutex_);
                    for (int i = 0; i < 7; i++) {
                        stop_cmd.joint_positions[i] = latest_robot_state_.joint_positions[i];
                    }
                }
                stop_cmd.timestamp = std::chrono::steady_clock::now();
                stop_cmd.is_valid = true;
                
                {
                    std::lock_guard<std::mutex> lock(position_mutex_);
                    current_position_cmd_ = stop_cmd;
                    last_command_time_ = stop_cmd.timestamp;
                }
                
                client_connected_ = false;
                break;
            }
            
            buffer[bytes_received] = '\0';
            processMessage(std::string(buffer));
        }
    }
    
    void processMessage(const std::string& message) {
        std::istringstream iss(message);
        std::string command;
        iss >> command;
        
        if (command == "GET_STATE") {
            sendRobotState();
        }
        else if (command == "SET_POSITION") {
            // Format: SET_POSITION p0 p1 p2 p3 p4 p5 p6
            PositionCommand cmd;
            bool valid = true;
            
            for (int i = 0; i < 7; i++) {
                if (!(iss >> cmd.joint_positions[i])) {
                    valid = false;
                    break;
                }
                // Basic range check (rough Franka limits)
                if (std::abs(cmd.joint_positions[i]) > 3.0) {
                    std::cerr << "Warning: Joint " << i << " position " << cmd.joint_positions[i] << " may be out of range" << std::endl;
                }
            }
            
            if (valid) {
                cmd.timestamp = std::chrono::steady_clock::now();
                cmd.is_valid = true;
                
                {
                    std::lock_guard<std::mutex> lock(position_mutex_);
                    current_position_cmd_ = cmd;
                    last_command_time_ = cmd.timestamp;
                }
                
                sendResponse("OK");
            } else {
                sendResponse("ERROR Invalid position command format");
            }
        }
        else if (command == "MOVE_TO_START") {
            // Format: MOVE_TO_START p0 p1 p2 p3 p4 p5 p6
            // This command moves slowly to the starting position using libfranka's position control
            std::array<double, 7> start_positions;
            bool valid = true;
            
            for (int i = 0; i < 7; i++) {
                if (!(iss >> start_positions[i])) {
                    valid = false;
                    break;
                }
            }
            
            if (valid && robot_connected_) {
                std::cout << "MOVE_TO_START command received - will execute before starting velocity control" << std::endl;
                
                // Store the start position to execute before velocity control begins
                {
                    std::lock_guard<std::mutex> lock(position_mutex_);
                    PositionCommand start_cmd;
                    start_cmd.joint_positions = start_positions;
                    start_cmd.timestamp = std::chrono::steady_clock::now();
                    start_cmd.is_valid = true;
                    
                    // Use a special flag to indicate this is a move-to-start command
                    pending_start_position_ = start_positions;
                    has_pending_start_ = true;
                }
                
                sendResponse("OK");
            } else {
                sendResponse("ERROR Invalid start position command or robot not connected");
            }
        }
        else if (command == "STOP") {
            // Stop all motion - hold current position
            PositionCommand stop_cmd;
            {
                std::lock_guard<std::mutex> lock(state_mutex_);
                for (int i = 0; i < 7; i++) {
                    stop_cmd.joint_positions[i] = latest_robot_state_.joint_positions[i];
                }
            }
            stop_cmd.timestamp = std::chrono::steady_clock::now();
            stop_cmd.is_valid = true;
            
            {
                std::lock_guard<std::mutex> lock(position_mutex_);
                current_position_cmd_ = stop_cmd;
                last_command_time_ = stop_cmd.timestamp;
            }
            
            sendResponse("OK");
        }
        else if (command == "DISCONNECT") {
            client_connected_ = false;
            sendResponse("OK");
        }
        else {
            sendResponse("ERROR Unknown command");
        }
    }
    
    void sendRobotState() {
        std::lock_guard<std::mutex> lock(state_mutex_);
        
        std::ostringstream response;
        response << "STATE ";
        
        // Joint positions
        for (int i = 0; i < 7; i++) {
            response << std::fixed << std::setprecision(6) << latest_robot_state_.joint_positions[i];
            if (i < 6) response << " ";
        }
        response << " ";
        
        // Joint velocities
        for (int i = 0; i < 7; i++) {
            response << std::fixed << std::setprecision(6) << latest_robot_state_.joint_velocities[i];
            if (i < 6) response << " ";
        }
        response << " ";
        
        // End-effector pose (4x4 transformation matrix)
        for (int i = 0; i < 16; i++) {
            response << std::fixed << std::setprecision(6) << latest_robot_state_.ee_pose[i];
            if (i < 15) response << " ";
        }
        
        sendResponse(response.str());
    }
    
    void sendResponse(const std::string& response) {
        if (client_connected_ && client_socket_ >= 0) {
            std::string msg = response + "\n";
            send(client_socket_, msg.c_str(), msg.length(), 0);
        }
    }
    
    std::array<double, 7> getCurrentPositionCommand() {
        std::lock_guard<std::mutex> lock(position_mutex_);
        
        // If no valid command ever received, return current robot position
        if (!current_position_cmd_.is_valid) {
            std::lock_guard<std::mutex> state_lock(state_mutex_);
            return latest_robot_state_.joint_positions;
        }
        
        auto now = std::chrono::steady_clock::now();
        double time_since_last_cmd = std::chrono::duration<double>(now - last_command_time_).count();
        
        // If no recent commands, return current position to hold position
        if (time_since_last_cmd > COMMAND_TIMEOUT_SEC) {
            std::lock_guard<std::mutex> state_lock(state_mutex_);
            return latest_robot_state_.joint_positions;
        }
        
        return current_position_cmd_.joint_positions;
    }
    
public:
    bool connectRobot(const std::string& robot_ip, double dynamics_factor = 0.3) {
        try {
            robot_ = new franka::Robot(robot_ip);
            
            // Set default collision behavior directly
            robot_->setCollisionBehavior(
                  {{250.0, 250.0, 250.0, 250.0, 200.0, 200.0, 150.0}}, {{250.0, 250.0, 250.0, 250.0, 200.0, 200.0, 150.0}},
                  {{250.0, 250.0, 250.0, 250.0, 200.0, 200.0, 150.0}}, {{250.0, 250.0, 250.0, 250.0, 200.0, 200.0, 150.0}},
                  {{150.0, 150.0, 150.0, 150.0, 150.0, 150.0}}, {{150.0, 150.0, 150.0, 150.0, 150.0, 150.0}},
                  {{150.0, 150.0, 150.0, 150.0, 150.0, 150.0}}, {{150.0, 150.0, 150.0, 150.0, 150.0, 150.0}});
            
            // Set joint impedance for velocity control (moderate stiffness)
            robot_->setJointImpedance({{3000, 3000, 3000, 2500, 2500, 2000, 2000}});
            
            // Initialize robot state by reading current state
            franka::RobotState initial_state = robot_->readOnce();
            {
                std::lock_guard<std::mutex> lock(state_mutex_);
                for (int i = 0; i < 7; i++) {
                    latest_robot_state_.joint_positions[i] = initial_state.q[i];
                    latest_robot_state_.joint_velocities[i] = initial_state.dq[i];
                }
                for (int i = 0; i < 16; i++) {
                    latest_robot_state_.ee_pose[i] = initial_state.O_T_EE[i];
                }
                latest_robot_state_.timestamp = std::chrono::steady_clock::now();
            }
            
            robot_connected_ = true;
            std::cout << "Connected to robot at " << robot_ip << std::endl;
            std::cout << "Initial robot position read successfully" << std::endl;
            return true;
            
        } catch (const franka::Exception& e) {
            std::cerr << "Failed to connect to robot: " << e.what() << std::endl;
            return false;
        }
    }
    
    void startVelocityControl() {
        if (!robot_connected_) {
            throw std::runtime_error("Robot not connected");
        }
        
        auto control_callback = [this](const franka::RobotState& robot_state, 
                                     franka::Duration period) -> franka::JointVelocities {
            
            // Update stored robot state
            {
                std::lock_guard<std::mutex> lock(state_mutex_);
                for (int i = 0; i < 7; i++) {
                    latest_robot_state_.joint_positions[i] = robot_state.q[i];
                    latest_robot_state_.joint_velocities[i] = robot_state.dq[i];
                }
                for (int i = 0; i < 16; i++) {
                    latest_robot_state_.ee_pose[i] = robot_state.O_T_EE[i];
                }
                latest_robot_state_.timestamp = std::chrono::steady_clock::now();
            }
            
            // Initialize ruckig on first call - exactly like reference implementation
            if (!ruckig_initialized_) {
                for (int i = 0; i < 7; i++) {
                    ruckig_input_.current_position[i] = robot_state.q[i];
                    ruckig_input_.current_velocity[i] = 0.0; // Start with zero velocity command
                    ruckig_input_.current_acceleration[i] = 0.0; // Start with zero acceleration
                    ruckig_input_.target_position[i] = robot_state.q[i]; // Start with current position as target
                    ruckig_input_.target_velocity[i] = 0.0; // Start with zero target velocity
                }
                ruckig_initialized_ = true;
                std::cout << "Ruckig velocity control initialized!" << std::endl;
                std::cout << "Starting with zero velocity commands to smoothly take over control" << std::endl;
            } else {
                // Update current state for Ruckig using previous Ruckig output for continuity
                for (int i = 0; i < 7; i++) {
                    ruckig_input_.current_position[i] = robot_state.q[i];
                    ruckig_input_.current_velocity[i] = ruckig_output_.new_velocity[i]; // Use our own velocity command for continuity
                    ruckig_input_.current_acceleration[i] = ruckig_output_.new_acceleration[i]; // Use Ruckig's acceleration
                }
            }
            
            // Get position commands from client
            std::array<double, 7> target_positions = getCurrentPositionCommand();
            
            // Set ruckig target positions directly from client commands
            for (int i = 0; i < 7; i++) {
                ruckig_input_.target_position[i] = target_positions[i];
                ruckig_input_.target_velocity[i] = 0.0; // Always target zero velocity for smooth stops
            }
            
            // Always run Ruckig to generate smooth velocity commands
            ruckig::Result ruckig_result = trajectory_generator_->update(ruckig_input_, ruckig_output_);
            
            std::array<double, 7> target_joint_velocities;
            
            if (ruckig_result == ruckig::Result::Working || ruckig_result == ruckig::Result::Finished) {
                // Use Ruckig's smooth velocity output
                for (int i = 0; i < 7; i++) {
                    target_joint_velocities[i] = ruckig_output_.new_velocity[i];
                }
            } else {
                // Emergency fallback: zero velocity to stop smoothly
                for (int i = 0; i < 7; i++) {
                    target_joint_velocities[i] = 0.0;
                }
                std::cout << "Ruckig error, using zero velocity for safety" << std::endl;
            }
            
            // Debug output for first few cycles
            static int debug_counter = 0;
            debug_counter++;
            if (debug_counter <= 10 || debug_counter % 100 == 0) {
                std::cout << "Target pos: ";
                for (int i = 0; i < 7; i++) {
                    std::cout << std::fixed << std::setprecision(4) << target_positions[i] << " ";
                }
                std::cout << " | Ruckig vel: ";
                for (int i = 0; i < 7; i++) {
                    std::cout << std::fixed << std::setprecision(4) << target_joint_velocities[i] << " ";
                }
                std::cout << std::endl;
            }
            
            if (!running_ || !client_connected_) {
                // Smooth stop: continue outputting decreasing velocities instead of abrupt stop
                static bool stopping = false;
                static int stop_counter = 0;
                
                if (!stopping) {
                    std::cout << "Stopping control loop - robot will decelerate smoothly" << std::endl;
                    stopping = true;
                    stop_counter = 0;
                    
                    // Set target position to current position to stop smoothly
                    for (int i = 0; i < 7; i++) {
                        ruckig_input_.target_position[i] = robot_state.q[i];
                        ruckig_input_.target_velocity[i] = 0.0;
                    }
                }
                
                stop_counter++;
                
                // Continue running ruckig for smooth deceleration
                ruckig::Result ruckig_result = trajectory_generator_->update(ruckig_input_, ruckig_output_);
                
                if (ruckig_result == ruckig::Result::Working || ruckig_result == ruckig::Result::Finished) {
                    for (int i = 0; i < 7; i++) {
                        target_joint_velocities[i] = ruckig_output_.new_velocity[i];
                    }
                } else {
                    for (int i = 0; i < 7; i++) {
                        target_joint_velocities[i] = 0.0;
                    }
                }
                
                // Check if robot has stopped (very low velocities)
                double max_vel = 0.0;
                for (int i = 0; i < 7; i++) {
                    max_vel = std::max(max_vel, std::abs(target_joint_velocities[i]));
                }
                
                // Stop when velocities are very small or after timeout
                if (max_vel < 0.001 || stop_counter > 1000) { // 1 second timeout at 1kHz
                    std::cout << "Robot stopped smoothly" << std::endl;
                    stopping = false;
                    return franka::MotionFinished(franka::JointVelocities({0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0}));
                }
                
                return franka::JointVelocities(target_joint_velocities);
            }
            
            return franka::JointVelocities(target_joint_velocities);
        };
        
        try {
            std::cout << "Starting velocity control loop..." << std::endl;
            std::cout << "WARNING: Robot will stop when client disconnects or server exits" << std::endl;
            
            robot_->control(control_callback);
            
        } catch (const franka::ControlException& e) {
            std::cerr << "Control exception: " << e.what() << std::endl;
        }
        
        // After control loop ends, explicitly stop the robot
        try {
            std::cout << "Control loop ended - stopping robot" << std::endl;
            robot_->stop();
        } catch (const franka::Exception& e) {
            std::cerr << "Error stopping robot: " << e.what() << std::endl;
        }
    }
    
    void run(const std::string& robot_ip) {
        // Store robot IP for reconnections
        robot_ip_ = robot_ip;
        
        // Connect to robot
        if (!connectRobot(robot_ip)) {
            throw std::runtime_error("Failed to connect to robot");
        }
        
        // Start network thread
        std::thread network_thread(&FrankaPositionServer::networkThread, this);
        
        std::cout << "Server ready. Waiting for client connection..." << std::endl;
        
        while (running_) {
            // Wait for client to connect before starting control
            std::cout << "Waiting for client connection before starting control..." << std::endl;
            while (running_ && !client_connected_) {
                std::this_thread::sleep_for(std::chrono::milliseconds(100));
            }
            
            if (!running_) break;
            
            std::cout << "Client connection detected, proceeding..." << std::endl;
            
            std::cout << "Client connected. Waiting for commands..." << std::endl;
            
            // Wait for potential MOVE_TO_START command or first position command
            // Give client time to send initial commands before starting control
            std::this_thread::sleep_for(std::chrono::milliseconds(500));
            
            // Check if we need to move to start position first
            if (has_pending_start_) {
                std::cout << "Moving to start position first..." << std::endl;
                try {
                    MotionGenerator motion_generator(0.5, pending_start_position_);
                    robot_->control(motion_generator);
                    std::cout << "Reached start position" << std::endl;
                    has_pending_start_ = false;
                } catch (const franka::Exception& e) {
                    std::cerr << "Failed to move to start position: " << e.what() << std::endl;
                    // Continue anyway - don't break the session
                }
            }
            
            std::cout << "Starting velocity control..." << std::endl;
            std::cout << "Robot will remain stationary until velocity commands are received." << std::endl;
            
            try {
                // Reset ruckig state for new session
                ruckig_initialized_ = false;
                
                // Start velocity control (blocking)
                startVelocityControl();
                
            } catch (const std::exception& e) {
                std::cerr << "Control error: " << e.what() << std::endl;
            }
            
            // Prevent network thread from accepting new connections during reconnection
            ready_for_clients_ = false;
            
            // After control ends, we need to reconnect to robot for next session
            std::cout << "Reconnecting to robot for next session..." << std::endl;
            
            // Disconnect and reconnect robot
            if (robot_) {
                delete robot_;
                robot_ = nullptr;
                robot_connected_ = false;
            }
            
            // Wait a moment for robot to be ready
            std::this_thread::sleep_for(std::chrono::milliseconds(100));
            
            // Reconnect
            if (!connectRobot(robot_ip_)) {
                std::cerr << "Failed to reconnect to robot" << std::endl;
                running_ = false;
                break;
            }
            
            // Ensure client_connected_ is false before waiting for next client
            client_connected_ = false;
            
            // Now ready to accept new clients
            ready_for_clients_ = true;
        }
        
        running_ = false;
        if (network_thread.joinable()) {
            network_thread.join();
        }
        
        if (robot_) {
            delete robot_;
            robot_ = nullptr;
        }
    }
    
    void shutdown() {
        running_ = false;
    }
};

int main(int argc, char** argv) {
    if (argc != 2) {
        std::cerr << "Usage: " << argv[0] << " <robot-hostname>" << std::endl;
        return -1;
    }
    
    try {
        FrankaPositionServer server;
        
        // Signal handler for graceful shutdown
        std::signal(SIGINT, [](int) {
            std::cout << "\nShutting down server..." << std::endl;
            exit(0);
        });
        
        server.run(argv[1]);
        
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return -1;
    }
    
    return 0;
}