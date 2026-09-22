#pragma once
#include "types.hpp"

double calculate_ROT_current(
    double i, 
    double V, 
    double omega, 
    double T,
    const Electric_Param& electric_param
);
double calculate_ROT_omega(
    double i, 
    double omega, 
    double load_torque, 
    const Mechanical_Param& mechanical_param
);
double calculate_ROT_temp(
    double i,
    double T,
    const Thermal_Param& thermal_param,
    const Electric_Param& electric_param
);