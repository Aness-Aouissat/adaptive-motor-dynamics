from Data.preprocessing import to_np_arrays, generate_model_inputs, generate_model_targets
from Data.transition_conversion import convert_transition
from Models.linear import LinearRegression
from Models.polynomial import to_polynomial
from Models.mlp import MultiLayerPerceptron, from_np_to_tensor, from_tensor_to_np
from Live.session_state import SessionState
from Live.loaded_model import LoadedModel
from Training.model_type import Model

import cpp_py_interface as cpi
import numpy as np

def prepare_transition(CurrState: cpi.Current_State, Input: cpi.Input):

    transition = cpi.generate_transition(CurrState, Input) 
    converted_transition = convert_transition(transition) 

    storage_array = [converted_transition]
    np_transition = to_np_arrays(storage_array)
    model_input = generate_model_inputs(np_transition)
    model_target = generate_model_targets(np_transition)

    return (model_input, model_target)

def inference(model_input: np.ndarray, loaded_model: LoadedModel) -> np.ndarray:

    match loaded_model.model_type:
        case Model.LINEAR:
            if loaded_model.x_scaler != None and loaded_model.y_scaler != None:
                scaled_model_input = loaded_model.x_scaler.transform(model_input)
                prediction = loaded_model.loaded_model.predict(scaled_model_input)
                return loaded_model.y_scaler.inverse_transform(prediction)
            else:
                return loaded_model.loaded_model.predict(model_input)
            
        case Model.POLYNOMIAL:
            if loaded_model.x_scaler != None and loaded_model.y_scaler != None:
                scaled_model_input = loaded_model.x_scaler.transform(model_input)
                scaled_polynomial_model_input = to_polynomial(scaled_model_input)
                prediction = loaded_model.loaded_model.predict(scaled_polynomial_model_input)
                return loaded_model.y_scaler.inverse_transform(prediction)
            else:
                polynomial_model_input = to_polynomial(model_input)
                return loaded_model.loaded_model.predict(polynomial_model_input)
            
        case Model.MLP:
            if loaded_model.x_scaler is not None and loaded_model.y_scaler is not None:
                scaled_model_input = loaded_model.x_scaler.transform(model_input)
                tensor_input = from_np_to_tensor(scaled_model_input)
                tensor_prediction = loaded_model.loaded_model(tensor_input)
                prediction = from_tensor_to_np(tensor_prediction)
                return loaded_model.y_scaler.inverse_transform(prediction)
            else:
                tensor_input = from_np_to_tensor(model_input)
                tensor_prediction = loaded_model.loaded_model(tensor_input)
                return from_tensor_to_np(tensor_prediction)

def step(session_state: SessionState, loaded_model: LoadedModel): 

    if session_state.step_count % 100 == 0:
        session_state.current_input = cpi.Input(session_state.rng.uniform(5, 13), session_state.rng.uniform(0.05, 0.35))

    model_input, model_target = prepare_transition(session_state.current_state, session_state.current_input)

    prediction = inference(model_input, loaded_model)
    residual_error = model_target - prediction
    
    session_state.step_count += 1

    return (model_input, model_target, prediction, residual_error)