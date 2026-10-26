import cpp_py_interface as cpi

def populate_electric_dict(electric_dict, rng):

    electric_ranges = {
        cpi.electric_fields.R_0: (-0.7, 2.0),
        cpi.electric_fields.T_0: (0, 0),
        cpi.electric_fields.L_a: (-0.005, 0.015),
        cpi.electric_fields.K_e: (-0.05, 0.12),
        cpi.electric_fields.alpha: (-0.0025, 0.005)
    }

    for field, value_range in electric_ranges.items():
        electric_dict[field] = rng.uniform(value_range[0], value_range[1])
    
def populate_mechanical_dict(mechanical_dict, rng):

    mechanical_ranges = {
        cpi.mechanical_fields.K_t: (-0.05, 0.12),
        cpi.mechanical_fields.D: (-0.0015, 0.005),
        cpi.mechanical_fields.J: (-0.0014, 0.003)
    }

    for field, value_range in mechanical_ranges.items():
        mechanical_dict[field] = rng.uniform(value_range[0], value_range[1])

def populate_thermal_dict(thermal_dict, rng):

    thermal_ranges = {
        cpi.thermal_fields.T_ambient: (-12.0, 25.0),
        cpi.thermal_fields.R_th: (-1.0, 3.0),
        cpi.thermal_fields.C_th: (-90.0, 180.0)
    }

    for field, value_range in thermal_ranges.items():
        thermal_dict[field] = rng.uniform(value_range[0], value_range[1])

def populate_request(rng):

    electric_dict = dict()
    populate_electric_dict(electric_dict, rng)

    mechanical_dict = dict()
    populate_mechanical_dict(mechanical_dict, rng)

    thermal_dict = dict()
    populate_thermal_dict(thermal_dict, rng)

    return cpi.Request(electric_dict, mechanical_dict, thermal_dict)

def mutate(rng):

    request = populate_request(rng)

    cpi.mutate_electric_fields(request)
    cpi.mutate_mechanical_fields(request)
    cpi.mutate_thermal_fields(request)