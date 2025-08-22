// weighted_ik.h - Weighted IK solver stub
#pragma once

#include <array>

struct WeightedIKResult {
    bool success;
    std::array<double, 7> joint_angles;
};

class WeightedIKSolver {
public:
    WeightedIKSolver(const std::array<double, 7>& neutral_pose,
                     double manipulability_weight,
                     double neutral_distance_weight,
                     double current_distance_weight,
                     const std::array<double, 7>& joint_weights,
                     bool verbose);
    
    WeightedIKResult solve_q7_optimized(
        const std::array<double, 3>& target_pos,
        const std::array<double, 9>& target_rot,
        const std::array<double, 7>& current_joints,
        double q7_start, double q7_end,
        double tolerance, int max_iterations);

private:
    std::array<double, 7> neutral_pose_;
    std::array<double, 7> joint_weights_;
    bool verbose_;
};