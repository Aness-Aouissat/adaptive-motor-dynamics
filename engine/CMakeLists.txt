set(PYBIND11_FINDPYTHON ON)
find_package(pybind11 CONFIG REQUIRED)

pybind11_add_module(
    cpp_py_interface 
    
    bindings/cpp_py_interface.cpp

    motor/motor_dynamics.cpp
    motor/motor_parameters.cpp

    simulation/generate_data.cpp
    simulation/integral_step.cpp
    )
target_include_directories(cpp_py_interface PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/motor ${CMAKE_CURRENT_SOURCE_DIR}/simulation)

install(TARGETS cpp_py_interface DESTINATION .)

