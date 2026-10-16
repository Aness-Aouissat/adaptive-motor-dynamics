#pragma once
#include "simulation_types.hpp"

std::vector<std::vector<Transition>> generate_datasets(int, int, int);
Transition generate_transition(Current_State&, Input);