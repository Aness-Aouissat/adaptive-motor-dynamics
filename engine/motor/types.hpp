#pragma once

struct Electric_Param {
    double R_0;
    double T_0;
    double L_a;
    double K_e;
    double alpha;
};

struct Mechanical_Param {
    double K_t;
    double D;
    double J;
} ;

struct Thermal_Param {
    double T_ambient;
    double R_th;
    double C_th;
};