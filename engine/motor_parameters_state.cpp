#include "interface.hpp"

static Electric_Param electric_param{1, 2, 3}; //TBD, filler for now
static Mechanical_Param mechanical_param{1, 2, 3}; //TBD, filler for now

const Electric_Param& get_electric_params() {
    return electric_param;
}

const Mechanical_Param& get_mechanical_params() {
    return mechanical_param;
}

//static void mutate_electric_fields() {
//
//}   

//static void mutate_mechanical_fields() {
//
//}