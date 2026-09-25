#pragma once

struct Current_State {
    double curr_i;
    double curr_omega;
    double curr_T;
};

struct New_State {
    double new_i;
    double new_omega;
    double new_T;
};

struct Input {
    double V;
    double load_torque;
};

struct Transition {
    Current_State current_state;
    Input input;
    New_State new_state;
};

double next_state(double, double, double);