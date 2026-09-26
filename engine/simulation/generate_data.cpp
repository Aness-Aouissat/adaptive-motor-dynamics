#include "motor/motor_dynamics.hpp"
#include "motor/motor_parameters.hpp"
#include "simulation_types.hpp"
#include <vector>
#include <random>

static std::vector<Transition> dataset_1, dataset_2, dataset_3, dataset_4;

static Current_State current_state;

Transition generate_transition( 
    Current_State& current_state, 
    Input input
) {
    const Electric_Param& electric_params = get_electric_params();
    double current_rate_of_change = calculate_ROT_current(
        current_state.curr_i, 
        input.V, 
        current_state.curr_omega, 
        current_state.curr_T, 
        electric_params
    );
    
    const Mechanical_Param& mechanical_params = get_mechanical_params();
    double omega_rate_of_change = calculate_ROT_omega(
        current_state.curr_i,
        current_state.curr_omega,
        input.load_torque,
        mechanical_params
    );
    
    const Thermal_Param& thermal_params = get_thermal_params();
    double temp_rate_of_change = calculate_ROT_temp(
        current_state.curr_i,
        current_state.curr_T,
        thermal_params,
        electric_params
    ); 

    New_State new_state;

    new_state.new_i = next_state(0.001, current_rate_of_change, current_state.curr_i);
    new_state.new_omega = next_state(0.001, omega_rate_of_change, current_state.curr_omega);
    new_state.new_T = next_state(0.001, temp_rate_of_change, current_state.curr_T);

    Transition transition;

    transition.current_state = current_state;
    transition.input = input;
    transition.new_state = new_state;

    current_state.curr_i = new_state.new_i;
    current_state.curr_omega = new_state.new_omega;
    current_state.curr_T = new_state.new_T;

    return transition;
}

std::vector<Transition> generate_dataset(
    int vector_num,
    int samples
) {
    std::random_device random;
    std::mt19937 generator(random());
    std::uniform_real_distribution<double> V_range(9, 13); //arbitrary values, will define later
    std::uniform_real_distribution<double> load_torque_range(15, 20); //arbitrary values, will define later
    
    int i = 0;

    switch(vector_num){
        case 1: 
            i = 0;
            current_state = {0, 0, 21}; // arbitrary values, will define later
            while(i < samples) {
                double random_V = V_range(generator);
                double random_load_torque = load_torque_range(generator);
                Input input = {random_V, random_load_torque};
                Transition transition = generate_transition(current_state, input);
                dataset_1.push_back(transition);
                i++;
            };
            return dataset_1;
        case 2: 
            i = 0;
            current_state = {0, 0, 21}; // arbitrary values, will define later
            while(i < samples) {
                double random_V = V_range(generator);
                double random_load_torque = load_torque_range(generator);
                Input input = {random_V, random_load_torque};
                Transition transition = generate_transition(current_state, input);
                dataset_2.push_back(transition);
                i++;
            };
            return dataset_2;
        case 3:
            i = 0;
            current_state = {0, 0, 21}; // arbitrary values, will define later
            while(i < samples) {
                double random_V = V_range(generator);
                double random_load_torque = load_torque_range(generator);
                Input input = {random_V, random_load_torque};
                Transition transition = generate_transition(current_state, input);
                dataset_3.push_back(transition);
                i++;
            };
            return dataset_3;
        case 4:
            i = 0;
            current_state = {0, 0, 21}; // arbitrary values, will define later
            while(i < samples) {
                double random_V = V_range(generator);
                double random_load_torque = load_torque_range(generator);
                Input input = {random_V, random_load_torque};
                Transition transition = generate_transition(current_state, input);
                dataset_4.push_back(transition);
                i++;
            };
            return dataset_4;
        default:
            std::vector<Transition> empty;
            return empty;
    }
}