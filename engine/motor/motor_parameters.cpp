#include "types.hpp"
#include "motor_parameters.hpp"

static Electric_Param electric_param{1, 2, 3, 4, 5}; //TBD, filler for now
static Mechanical_Param mechanical_param{1, 2, 3}; //TBD, filler for now
static Thermal_Param thermal_param{1, 2, 3}; //TBD, filler for now

const Electric_Param& get_electric_params() {
    return electric_param;
}

const Mechanical_Param& get_mechanical_params() {
    return mechanical_param;
}

const Thermal_Param& get_thermal_params() {
    return thermal_param;
}

void mutate_electric_fields(const Request& request) {

    for(const auto& [param, delta] : request.electric_map) {
        switch(param) {
            case (electric_fields::R_0):
                electric_param.R_0 += delta;
                break;
            case (electric_fields::T_0):
                electric_param.T_0 += delta;
                break;
            case (electric_fields::L_a):
                electric_param.L_a += delta;
                break;
            case (electric_fields::K_e):
                electric_param.K_e += delta;
                break;
            case (electric_fields::alpha):
                electric_param.alpha += delta;
                break;
            default:
                break;
        }
    }
}   

void mutate_mechanical_fields(const Request& request) {

    for(const auto& [param, delta] : request.mechanical_map) {
        switch(param) {
            case (mechanical_fields::K_t):
                mechanical_param.K_t += delta;
                break;
            case (mechanical_fields::D):
                mechanical_param.D += delta;
                break;
            case (mechanical_fields::J):
                mechanical_param.J += delta;
                break;
            default:
                break;
        }
    }
}

void mutate_thermal_fields(const Request& request) {

    for(const auto& [param, delta] : request.thermal_map) {
        switch(param) {
            case (thermal_fields::T_ambient):
                thermal_param.T_ambient += delta;
                break;
            case (thermal_fields::R_th):
                thermal_param.R_th += delta;
                break;
            case (thermal_fields::C_th):
                thermal_param.C_th += delta;
                break;
            default:
                break;
        }
    }
}