from Training.config import HIDDEN_LAYERS
from Training.model_type import Model
from Live.mutation import mutate
from Live.config import PYTORCH_SEED, LEARNING_RATE, BATCH_SIZE, SEED, EPOCHS_PER_BATCH
from Live.loaded_model import LoadedModel
from Live.session_runner import step
from Live.session_state import SessionState
from Models.mlp import from_np_to_tensor, from_tensor_to_np
from Models import mlp
from pathlib import Path
from random import Random

import cpp_py_interface as cpi
import numpy as np
import joblib
import torch

curr_abs_path = Path(__file__).resolve()

def load_mlp_model():
    model_path = curr_abs_path.parent.parent / "Saved" / "models" / "MLP_Weights.pth"
    model_weights = torch.load(model_path, weights_only = True)
    loaded_model = mlp.MultiLayerPerceptron(5, HIDDEN_LAYERS, 3)
    loaded_model.load_state_dict(model_weights)
    loaded_model.eval()

    return loaded_model

def load_scalers():
    scalers_path = curr_abs_path.parent.parent / "Saved" / "scalers"
    x_scaler = joblib.load(scalers_path / "x_scaler.joblib")
    y_scaler = joblib.load(scalers_path / "y_scaler.joblib")

    return x_scaler, y_scaler

def mlp_validation_loop(num_of_samples, learning_rate, batch_size, steps_per_batch, seed = SEED):
    cpi.reset_mutations()
    torch.manual_seed(PYTORCH_SEED)

    x_scaler, y_scaler = load_scalers()

    session_state = SessionState(Model.MLP, cpi.Current_State(0, 0, 21.0), seed)

    simulation_model = LoadedModel(Model.MLP)
    simulation_model.load_model()

    loaded_model = load_mlp_model()
    optimizer = torch.optim.Adam(loaded_model.parameters(), lr = learning_rate)

    for i in range(100):
        step(session_state, simulation_model)

    mutate(Random(seed))

    x_batch = []
    y_batch = []
    errors = []

    for i in range(num_of_samples):
        model_input, model_target, prediction, residual_error = step(session_state, simulation_model)

        scaled_x = x_scaler.transform(np.asarray(model_input).reshape(1, 5))
        scaled_y = y_scaler.transform(np.asarray(model_target).reshape(1, 3))

        loaded_model.eval()

        with torch.no_grad():
            predicted = loaded_model(from_np_to_tensor(scaled_x))

        prediction = from_tensor_to_np(predicted)
        errors.append(np.abs(scaled_y - prediction).reshape(3))

        x_batch.append(scaled_x.reshape(5))
        y_batch.append(scaled_y.reshape(3))

        if len(x_batch) == batch_size:
            mlp_train_loop(np.asarray(x_batch), np.asarray(y_batch), loaded_model, optimizer, steps_per_batch)
            x_batch.clear()
            y_batch.clear()

    cpi.reset_mutations()

    return np.asarray(errors)

def mlp_test_loop(num_of_samples, test_seed):
    return mlp_validation_loop(num_of_samples, LEARNING_RATE, BATCH_SIZE, EPOCHS_PER_BATCH, test_seed)

def mlp_train_loop(x_train, y_train, loaded_model, optimizer, epochs = 1):
    loaded_model.train()
    tensor_x_data = mlp.from_np_to_tensor(x_train)
    tensor_y_data = mlp.from_np_to_tensor(y_train)

    for epoch in range(epochs):

        optimizer.zero_grad()
        curr_prediction = loaded_model(tensor_x_data)
        loss = ((curr_prediction - tensor_y_data)**2).mean()
        
        loss.backward()
        optimizer.step()
        
    return loaded_model