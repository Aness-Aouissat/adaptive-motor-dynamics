double next_state(double time_delta, double ROT, double curr_state) {

    return curr_state + (time_delta * ROT);
}