import cpp_py_interface as cpi

type converted_transition = tuple[tuple[float, float, float], tuple[float, float], tuple[float, float, float]]
type converted_dataset = list[converted_transition]
type converted_datasets = list[converted_dataset]

def convert_multiple_datasets(datasets: list[list[cpi.Transition]]) -> converted_datasets:

    converted_datasets = []

    for i in range(len(datasets)):
        dataset = datasets[i]
        converted_dataset = convert_dataset(dataset) 
        converted_datasets.append(converted_dataset)

    return converted_datasets

def convert_dataset(transition_dataset: list[cpi.Transition]) -> converted_dataset:

    converted_dataset = []

    for i in range(len(transition_dataset)):
        transition = transition_dataset[i]
        converted_transition = convert_transition(transition)
        converted_dataset.append(converted_transition)

    return converted_dataset

def convert_transition(transition: cpi.Transition) -> converted_transition:

    converted_current_state = (transition.current_state.curr_i, transition.current_state.curr_omega, transition.current_state.curr_T)
    converted_input = (transition.input.V, transition.input.load_torque)
    converted_new_state = (transition.new_state.new_i, transition.new_state.new_omega, transition.new_state.new_T)

    converted_transition = (converted_current_state, converted_input, converted_new_state)

    return converted_transition