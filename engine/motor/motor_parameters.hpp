#pragma once 
#include <unordered_map>
#include "types.hpp"

enum class electric_fields {
    R_0,
    T_0,
    L_a,
    K_e,
    alpha,
};

enum class mechanical_fields {
    K_t,
    D,
    J
};

enum class thermal_fields {
    T_ambient,
    R_th,
    C_th
};

typedef struct{
    std::unordered_map<electric_fields, double> electric_map;
    std::unordered_map<mechanical_fields, double> mechanical_map;
    std::unordered_map<thermal_fields, double> thermal_map;
} Request;

void mutate_electric_fields(const Request&);
void mutate_mechanical_fields(const Request&);
void mutate_thermal_fields(const Request&);

const Electric_Param& get_electric_params();
const Mechanical_Param& get_mechanical_params();
const Thermal_Param& get_thermal_params();