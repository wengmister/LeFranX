// Lightweight IK Bridge - Direct WeightedIKSolver bindings
// Clean interface without TCP/VR dependencies

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>

// Include the weighted IK solver
#include "../include/weighted_ik.h"

namespace py = pybind11;

PYBIND11_MODULE(weighted_ik_bridge, m) {
    m.doc() = "Lightweight WeightedIK Bridge - Direct IK solver interface";
    
    // Bind WeightedIKResult struct
    py::class_<WeightedIKResult>(m, "WeightedIKResult")
        .def(py::init<>())
        .def_readwrite("success", &WeightedIKResult::success)
        .def_readwrite("joint_angles", &WeightedIKResult::joint_angles)
        .def_readwrite("q7_optimal", &WeightedIKResult::q7_optimal)
        .def_readwrite("score", &WeightedIKResult::score)
        .def_readwrite("manipulability", &WeightedIKResult::manipulability)
        .def_readwrite("neutral_distance", &WeightedIKResult::neutral_distance)
        .def_readwrite("current_distance", &WeightedIKResult::current_distance)
        .def_readwrite("solution_index", &WeightedIKResult::solution_index)
        .def_readwrite("total_solutions_found", &WeightedIKResult::total_solutions_found)
        .def_readwrite("valid_solutions_count", &WeightedIKResult::valid_solutions_count)
        .def_readwrite("q7_values_tested", &WeightedIKResult::q7_values_tested)
        .def_readwrite("optimization_iterations", &WeightedIKResult::optimization_iterations)
        .def_readwrite("duration_microseconds", &WeightedIKResult::duration_microseconds);
    
    // Bind WeightedIKSolver class
    py::class_<WeightedIKSolver>(m, "WeightedIKSolver")
        .def(py::init<const std::array<double, 7>&, double, double, double, 
                     const std::array<double, 7>&, bool>(),
             py::arg("neutral_pose"),
             py::arg("weight_manip"),
             py::arg("weight_neutral"), 
             py::arg("weight_current"),
             py::arg("joint_weights") = std::array<double, 7>{{1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0}},
             py::arg("verbose") = true)
        
        .def("solve_q7_optimized", &WeightedIKSolver::solve_q7_optimized,
             py::arg("target_position"),
             py::arg("target_orientation"),
             py::arg("current_pose"),
             py::arg("q7_min"),
             py::arg("q7_max"),
             py::arg("tolerance") = 1e-6,
             py::arg("max_iterations") = 100)
        
        .def("update_weights", &WeightedIKSolver::update_weights,
             py::arg("weight_manip"),
             py::arg("weight_neutral"),
             py::arg("weight_current"))
        
        .def("update_joint_weights", &WeightedIKSolver::update_joint_weights,
             py::arg("joint_weights"))
        
        .def("update_neutral_pose", &WeightedIKSolver::update_neutral_pose,
             py::arg("neutral_pose"))
        
        .def("get_neutral_pose", &WeightedIKSolver::get_neutral_pose)
        
        .def("set_verbose", &WeightedIKSolver::set_verbose,
             py::arg("verbose"));
}