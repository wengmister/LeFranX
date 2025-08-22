// VR Message Router - Clean TCP/ADB handler for dual arm+hand VR teleoperation
// Separates message routing from processing logic

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>

#include <memory>
#include <thread>
#include <atomic>
#include <mutex>
#include <chrono>
#include <iostream>
#include <algorithm>
#include <sstream>
#include <regex>
#include <sys/socket.h>
#include <netinet/in.h>
#include <arpa/inet.h>
#include <unistd.h>

namespace py = pybind11;

struct VRWristData {
    std::array<double, 3> position;
    std::array<double, 4> quaternion;  // x, y, z, w
    std::string fist_state;
    bool valid = false;
    std::chrono::steady_clock::time_point timestamp;
};

struct VRLandmarks {
    std::vector<std::array<double, 3>> landmarks;  // 21 landmarks x [x, y, z]
    bool valid = false;
    std::chrono::steady_clock::time_point timestamp;
};

struct VRMessages {
    VRWristData wrist_data;
    VRLandmarks landmarks_data;
    bool wrist_valid = false;
    bool landmarks_valid = false;
};

struct VRRouterConfig {
    int tcp_port = 8000;
    bool verbose = false;
    double message_timeout_ms = 100.0;  // Timeout for message validity
};

class VRMessageRouter {
private:
    // TCP connection
    int tcp_socket_ = -1;
    int client_socket_ = -1;
    std::atomic<bool> running_{false};
    std::thread tcp_thread_;
    
    // Message state
    VRMessages current_messages_;
    std::mutex messages_mutex_;
    
    // Configuration
    VRRouterConfig config_;
    
    // Message parsing patterns
    std::regex wrist_pattern_;
    std::regex landmarks_pattern_;

public:
    VRMessageRouter(const VRRouterConfig& config = VRRouterConfig()) 
        : config_(config) {
        
        // Initialize message parsing patterns
        // Wrist: "Right wrist:, x, y, z, qx, qy, qz, qw, leftFist: state"
        wrist_pattern_ = std::regex(R"(Right wrist:,\s*([-\d\.]+),\s*([-\d\.]+),\s*([-\d\.]+),\s*([-\d\.]+),\s*([-\d\.]+),\s*([-\d\.]+),\s*([-\d\.]+),\s*leftFist:\s*(\w+))");
        
        // Landmarks: "Right landmarks: x1,y1,z1,x2,y2,z2,..."
        landmarks_pattern_ = std::regex(R"(Right landmarks:\s*(.*))");
        
        if (config_.verbose) {
            std::cout << "VRMessageRouter initialized with TCP port " << config_.tcp_port << std::endl;
        }
    }
    
    ~VRMessageRouter() {
        stop();
    }
    
    bool start_tcp_server() {
        if (running_) {
            std::cerr << "TCP server already running" << std::endl;
            return false;
        }
        
        // Setup TCP socket
        tcp_socket_ = socket(AF_INET, SOCK_STREAM, 0);
        if (tcp_socket_ < 0) {
            std::cerr << "Failed to create TCP socket" << std::endl;
            return false;
        }
        
        // Allow socket reuse
        int opt = 1;
        if (setsockopt(tcp_socket_, SOL_SOCKET, SO_REUSEADDR, &opt, sizeof(opt)) < 0) {
            std::cerr << "Failed to set socket options" << std::endl;
            close(tcp_socket_);
            return false;
        }
        
        // Bind to port
        struct sockaddr_in address;
        address.sin_family = AF_INET;
        address.sin_addr.s_addr = INADDR_ANY;
        address.sin_port = htons(config_.tcp_port);
        
        if (bind(tcp_socket_, (struct sockaddr*)&address, sizeof(address)) < 0) {
            std::cerr << "Failed to bind TCP socket to port " << config_.tcp_port << std::endl;
            close(tcp_socket_);
            return false;
        }
        
        // Listen for connections
        if (listen(tcp_socket_, 1) < 0) {
            std::cerr << "Failed to listen on TCP socket" << std::endl;
            close(tcp_socket_);
            return false;
        }
        
        // Start receiver thread
        running_ = true;
        tcp_thread_ = std::thread(&VRMessageRouter::tcp_receiver_thread, this);
        
        if (config_.verbose) {
            std::cout << "VR TCP server started on port " << config_.tcp_port << std::endl;
        }
        
        return true;
    }
    
    void stop() {
        if (!running_) {
            return;
        }
        
        running_ = false;
        
        // Close sockets
        if (client_socket_ >= 0) {
            close(client_socket_);
            client_socket_ = -1;
        }
        
        if (tcp_socket_ >= 0) {
            close(tcp_socket_);
            tcp_socket_ = -1;
        }
        
        // Wait for thread to finish
        if (tcp_thread_.joinable()) {
            tcp_thread_.join();
        }
        
        if (config_.verbose) {
            std::cout << "VR TCP server stopped" << std::endl;
        }
    }
    
    VRMessages get_messages() {
        std::lock_guard<std::mutex> lock(messages_mutex_);
        
        // Check message timeouts
        auto now = std::chrono::steady_clock::now();
        auto timeout_duration = std::chrono::milliseconds(static_cast<int>(config_.message_timeout_ms));
        
        VRMessages result = current_messages_;
        
        // Mark as invalid if too old
        if (current_messages_.wrist_data.valid && 
            (now - current_messages_.wrist_data.timestamp) > timeout_duration) {
            result.wrist_valid = false;
        }
        
        if (current_messages_.landmarks_data.valid &&
            (now - current_messages_.landmarks_data.timestamp) > timeout_duration) {
            result.landmarks_valid = false;
        }
        
        return result;
    }
    
    py::dict get_status() {
        std::lock_guard<std::mutex> lock(messages_mutex_);
        
        py::dict status;
        status["tcp_connected"] = (client_socket_ >= 0);
        status["running"] = running_.load();
        status["wrist_valid"] = current_messages_.wrist_valid;
        status["landmarks_valid"] = current_messages_.landmarks_valid;
        
        return status;
    }

private:
    void tcp_receiver_thread() {
        if (config_.verbose) {
            std::cout << "VR TCP receiver thread started" << std::endl;
        }
        
        while (running_) {
            try {
                // Accept connection
                if (client_socket_ < 0) {
                    struct sockaddr_in client_address;
                    socklen_t client_len = sizeof(client_address);
                    
                    client_socket_ = accept(tcp_socket_, (struct sockaddr*)&client_address, &client_len);
                    if (client_socket_ < 0) {
                        if (running_) {
                            std::cerr << "Failed to accept TCP connection" << std::endl;
                        }
                        std::this_thread::sleep_for(std::chrono::milliseconds(100));
                        continue;
                    }
                    
                    if (config_.verbose) {
                        std::cout << "VR client connected from " << inet_ntoa(client_address.sin_addr) << std::endl;
                    }
                }
                
                // Receive data
                char buffer[4096];
                int bytes_received = recv(client_socket_, buffer, sizeof(buffer) - 1, 0);
                
                if (bytes_received <= 0) {
                    if (config_.verbose) {
                        std::cout << "VR client disconnected" << std::endl;
                    }
                    close(client_socket_);
                    client_socket_ = -1;
                    continue;
                }
                
                buffer[bytes_received] = '\0';
                parse_vr_messages(std::string(buffer));
                
            } catch (const std::exception& e) {
                std::cerr << "Error in VR TCP receiver: " << e.what() << std::endl;
                if (client_socket_ >= 0) {
                    close(client_socket_);
                    client_socket_ = -1;
                }
                std::this_thread::sleep_for(std::chrono::milliseconds(100));
            }
        }
        
        if (config_.verbose) {
            std::cout << "VR TCP receiver thread stopped" << std::endl;
        }
    }
    
    void parse_vr_messages(const std::string& message) {
        auto now = std::chrono::steady_clock::now();
        std::lock_guard<std::mutex> lock(messages_mutex_);
        
        // Parse wrist data
        std::smatch wrist_match;
        if (std::regex_search(message, wrist_match, wrist_pattern_)) {
            try {
                VRWristData wrist_data;
                wrist_data.position[0] = std::stod(wrist_match[1]);
                wrist_data.position[1] = std::stod(wrist_match[2]);
                wrist_data.position[2] = std::stod(wrist_match[3]);
                wrist_data.quaternion[0] = std::stod(wrist_match[4]);
                wrist_data.quaternion[1] = std::stod(wrist_match[5]);
                wrist_data.quaternion[2] = std::stod(wrist_match[6]);
                wrist_data.quaternion[3] = std::stod(wrist_match[7]);
                wrist_data.fist_state = wrist_match[8];
                wrist_data.valid = true;
                wrist_data.timestamp = now;
                
                current_messages_.wrist_data = wrist_data;
                current_messages_.wrist_valid = true;
                
                if (config_.verbose) {
                    std::cout << "Parsed wrist: pos=(" << wrist_data.position[0] << "," 
                              << wrist_data.position[1] << "," << wrist_data.position[2] 
                              << ") fist=" << wrist_data.fist_state << std::endl;
                }
                
            } catch (const std::exception& e) {
                std::cerr << "Error parsing wrist data: " << e.what() << std::endl;
            }
        }
        
        // Parse landmarks data
        std::smatch landmarks_match;
        if (std::regex_search(message, landmarks_match, landmarks_pattern_)) {
            try {
                VRLandmarks landmarks_data;
                std::string landmarks_str = landmarks_match[1];
                
                // Parse comma-separated landmarks: x1,y1,z1,x2,y2,z2,...
                std::istringstream iss(landmarks_str);
                std::string token;
                std::vector<double> values;
                
                while (std::getline(iss, token, ',')) {
                    // Trim whitespace
                    token.erase(0, token.find_first_not_of(" \t\r\n"));
                    token.erase(token.find_last_not_of(" \t\r\n") + 1);
                    
                    if (!token.empty()) {
                        values.push_back(std::stod(token));
                    }
                }
                
                // Convert to landmarks (groups of 3)
                landmarks_data.landmarks.clear();
                for (size_t i = 0; i + 2 < values.size(); i += 3) {
                    std::array<double, 3> landmark = {values[i], values[i+1], values[i+2]};
                    landmarks_data.landmarks.push_back(landmark);
                }
                
                landmarks_data.valid = true;
                landmarks_data.timestamp = now;
                
                current_messages_.landmarks_data = landmarks_data;
                current_messages_.landmarks_valid = true;
                
                if (config_.verbose) {
                    std::cout << "Parsed landmarks: " << landmarks_data.landmarks.size() 
                              << " points" << std::endl;
                }
                
            } catch (const std::exception& e) {
                std::cerr << "Error parsing landmarks data: " << e.what() << std::endl;
            }
        }
    }
};

// Python bindings
PYBIND11_MODULE(vr_message_router, m) {
    m.doc() = "VR Message Router - TCP/ADB handler for dual arm+hand VR teleoperation";
    
    py::class_<VRRouterConfig>(m, "VRRouterConfig")
        .def(py::init<>())
        .def_readwrite("tcp_port", &VRRouterConfig::tcp_port)
        .def_readwrite("verbose", &VRRouterConfig::verbose)
        .def_readwrite("message_timeout_ms", &VRRouterConfig::message_timeout_ms);
    
    py::class_<VRWristData>(m, "VRWristData")
        .def(py::init<>())
        .def_readwrite("position", &VRWristData::position)
        .def_readwrite("quaternion", &VRWristData::quaternion)
        .def_readwrite("fist_state", &VRWristData::fist_state)
        .def_readwrite("valid", &VRWristData::valid);
    
    py::class_<VRLandmarks>(m, "VRLandmarks")
        .def(py::init<>())
        .def_readwrite("landmarks", &VRLandmarks::landmarks)
        .def_readwrite("valid", &VRLandmarks::valid);
    
    py::class_<VRMessages>(m, "VRMessages")
        .def(py::init<>())
        .def_readwrite("wrist_data", &VRMessages::wrist_data)
        .def_readwrite("landmarks_data", &VRMessages::landmarks_data)
        .def_readwrite("wrist_valid", &VRMessages::wrist_valid)
        .def_readwrite("landmarks_valid", &VRMessages::landmarks_valid);
    
    py::class_<VRMessageRouter>(m, "VRMessageRouter")
        .def(py::init<const VRRouterConfig&>(), py::arg("config") = VRRouterConfig())
        .def("start_tcp_server", &VRMessageRouter::start_tcp_server)
        .def("stop", &VRMessageRouter::stop)
        .def("get_messages", &VRMessageRouter::get_messages)
        .def("get_status", &VRMessageRouter::get_status);
}