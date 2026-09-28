#include "motor_parameters.hpp"
#include "simulation_types.hpp"
#include "generate_data.hpp"
#include "types.hpp"

#include <pybind11/pybind11.h>
#include <pybind11/native_enum.h>
#include <pybind11/stl.h>

namespace py = pybind11;

PYBIND11_MODULE(cpp_py_interface, m) {

    py::class_<Current_State>(m, "Current_State")
        .def_readonly("curr_i", &Current_State::curr_i)
        .def_readonly("curr_omega", &Current_State::curr_omega)
        .def_readonly("curr_T", &Current_State::curr_T);

    py::class_<Input>(m, "Input")   
        .def_readonly("V", &Input::V)
        .def_readonly("load_torque", &Input::load_torque);

    py::class_<New_State>(m, "New_State")
        .def_readonly("new_i", &New_State::new_i)
        .def_readonly("new_omega", &New_State::new_omega)
        .def_readonly("new_T", &New_State::new_T);
    
    py::class_<Transition>(m, "Transition")
        .def_readonly("current_state", &Transition::current_state)
        .def_readonly("input", &Transition::input)
        .def_readonly("new_state", &Transition::new_state);

    m.def("generate_datasets", &generate_datasets, py::return_value_policy::move);

    py::native_enum<electric_fields>(m, "electric_fields", "enum.Enum")
        .value("R_0", electric_fields::R_0)
        .value("T_0", electric_fields::T_0)
        .value("L_a", electric_fields::L_a)
        .value("K_e", electric_fields::K_e)
        .value("alpha", electric_fields::alpha)
        .finalize();

    py::native_enum<mechanical_fields>(m, "mechanical_fields", "enum.Enum")
        .value("K_t", mechanical_fields::K_t)
        .value("D", mechanical_fields::D)
        .value("J", mechanical_fields::J)
        .finalize();

    py::native_enum<thermal_fields>(m, "thermal_fields", "enum.Enum")
        .value("T_ambient", thermal_fields::T_ambient)
        .value("R_th", thermal_fields::R_th)
        .value("C_th", thermal_fields::C_th)
        .finalize();

    py::class_<Request>(m, "Request", py::dynamic_attr())
        .def(py::init<>())
        .def_readwrite("electric_map", &Request::electric_map)
        .def_readwrite("mechanical_map", &Request::mechanical_map)
        .def_readwrite("thermal_map", &Request::thermal_map);

    m.def("mutate_electric_fields", &mutate_electric_fields);
    m.def("mutate_mechanical_fields", &mutate_mechanical_fields);
    m.def("mutate_thermal_fields", &mutate_thermal_fields);
}