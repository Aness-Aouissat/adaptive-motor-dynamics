#include <cmath>
#include "types.hpp"

double calculate_ROT_current(
    double i, 
    double V, 
    double omega, 
    double T,
    const Electric_Param& electric_param
) {

    double inverse_inductance = (1 / electric_param.L_a);
    double back_emf = (omega * electric_param.K_e);
    double resistive_voltage_drop = (i * (electric_param.R_0 * (1 + electric_param.alpha * (T - electric_param.T_0))));

    double derivative = inverse_inductance * (V - resistive_voltage_drop - back_emf);

    return derivative;
}

double calculate_ROT_omega(
    double i, 
    double omega, 
    double load_torque, 
    const Mechanical_Param& mechanical_param
) {

    double inverse_inertia = (1 / mechanical_param.J);
    double motor_torque = (i * mechanical_param.K_t);
    double damping_torque = (omega * mechanical_param.D);

    double derivative = inverse_inertia * (motor_torque - load_torque - damping_torque);

    return derivative;
}

double calculate_ROT_temp(
    double i,
    double T,
    const Thermal_Param& thermal_param,
    const Electric_Param& electric_param
) {

    double inverse_thermal_capacitance = (1 / thermal_param.C_th); 
    double heat_produced = (std::pow(i, 2) * (electric_param.R_0 * (1 + electric_param.alpha * (T - electric_param.T_0))));
    double heat_lost = ((T- thermal_param.T_ambient) / thermal_param.R_th);

    double derivative = inverse_thermal_capacitance * (heat_produced - heat_lost);

    return derivative;
}