#pragma once

typedef struct {
    double R_0;
    double T_0;
    double L_a;
    double K_e;
    double alpha;
} Electric_Param;

typedef struct {
    double K_t;
    double D;
    double J;
} Mechanical_Param;

typedef struct{
    double T_ambient;
    double R_th;
    double C_th;
} Thermal_Param;