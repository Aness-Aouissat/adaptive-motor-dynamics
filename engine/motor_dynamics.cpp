#include "interface.hpp"

double calculate_ROT_Current(double i, double V, double omega, const Electric_Param& electric_param) {

    double derivative;
    double inverse_inductance = (1 / electric_param.L_a);
    double back_emf = (omega * electric_param.K_e);
    double resistive_voltage_drop = (i * electric_param.R_a);

    derivative = inverse_inductance * (V - resistive_voltage_drop - back_emf);

    return derivative;
}

double calculate_ROT_Omega(double i, double omega, double load_torque, const Mechanical_Param& mechanical_param) {

    double derivative;
    double inverse_inertia = (1 / mechanical_param.J);
    double motor_torque = (i * mechanical_param.K_t);
    double damping_torque = (omega * mechanical_param.D);

    derivative = inverse_inertia * (motor_torque - load_torque - damping_torque);

    return derivative;
}

int main() {
    return 0;
}