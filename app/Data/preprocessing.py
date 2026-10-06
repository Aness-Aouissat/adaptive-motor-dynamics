import transition_conversion as tc
import numpy as np

def to_np_arrays(dataset: tc.converted_dataset) -> tuple[np.ndarray, np.ndarray, np.ndarray]:

    curr_state_array = np.empty((0, 3))
    input_array = np.empty((0, 2))
    new_state_array = np.empty((0, 3))

    for transition in dataset: 

        curr_state_element = np.array(transition[0]).reshape(1, 3)
        curr_state_array = np.append(curr_state_array, curr_state_element, axis = 0)

        input_element = np.array(transition[1]).reshape(1, 2)
        input_array = np.append(input_array, input_element, axis = 0)

        new_state_element = np.array(transition[2]).reshape(1, 3)
        new_state_array = np.append(new_state_array, new_state_element, axis = 0)

    return (curr_state_array, input_array, new_state_array)
    
def generate_model_inputs(dataset: tuple[np.ndarray, np.ndarray, np.ndarray]) -> np.ndarray:

    curr_state_array = dataset[0]
    input_array = dataset[1]

    model_inputs = np.concatenate((curr_state_array, input_array), axis = 1)

    return model_inputs

def generate_model_targets(dataset: tuple[np.ndarray, np.ndarray, np.ndarray]) -> np.ndarray:

    curr_state_array = dataset[0]
    new_state_array = dataset[2]

    model_targets = new_state_array - curr_state_array

    return model_targets

# def standardize or normalize():
#    None