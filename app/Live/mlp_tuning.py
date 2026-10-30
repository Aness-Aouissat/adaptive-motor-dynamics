from Training.config import HIDDEN_LAYERS
from Data.orchestration import validate_pipeline, train_pipeline
from Data.preprocessing import Scaler
from Data.splitting import Strategy
from Live.mutation import mutate
from Live.config import PYTORCH_SEED, LEARNING_RATE, EPOCHS
from Models.mlp import from_np_to_tensor, from_tensor_to_np
from Models import mlp
from sklearn.metrics import r2_score
from pathlib import Path
from random import Random

import cpp_py_interface as cpi
import joblib
import torch

curr_abs_path = Path(__file__).resolve()
seed = 0

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

x_scaler, y_scaler = load_scalers()

def mlp_validation_loop(num_of_trajectories, num_of_samples, learning_rate, epochs):

    cpi.reset_mutations()
    mutate(Random(seed))

    split_data = validate_pipeline(num_of_trajectories, num_of_samples, seed = seed, validation_strategy = Strategy.HOLDOUT, scale_technique = Scaler.NONE)
    cpi.reset_mutations()

    x_train, y_train = split_data[0], split_data[1]
    x_validate, y_validate = split_data[2], split_data[3]

    scaled_x_train = x_scaler.transform(x_train)
    scaled_x_validate = x_scaler.transform(x_validate)
    scaled_y_train = y_scaler.transform(y_train)
    scaled_y_validate = y_scaler.transform(y_validate)

    loaded_model = load_mlp_model()

    torch.manual_seed(PYTORCH_SEED)
    trained_model = mlp_train_loop(scaled_x_train, scaled_y_train, loaded_model, learning_rate, epochs)

    trained_model.eval()

    tensor_x_validate_data = from_np_to_tensor(scaled_x_validate)
    with torch.no_grad():
        predicted = trained_model(tensor_x_validate_data)
        
    np_predicted_array = from_tensor_to_np(predicted)
                            
    return r2_score(scaled_y_validate, np_predicted_array)

def mlp_test_loop(num_of_trajectories, num_of_samples):

    cpi.reset_mutations()
    mutate(Random(seed))
    
    split_data = train_pipeline(num_of_trajectories, num_of_samples, seed = seed, scale_technique = Scaler.NONE)
    cpi.reset_mutations()

    x_train, y_train = split_data[0], split_data[1]
    x_test, y_test = split_data[2], split_data[3]

    scaled_x_train = x_scaler.transform(x_train)
    scaled_x_test = x_scaler.transform(x_test)
    scaled_y_train = y_scaler.transform(y_train)
    scaled_y_test = y_scaler.transform(y_test)

    loaded_model = load_mlp_model()
    
    torch.manual_seed(PYTORCH_SEED)
    trained_model = mlp_train_loop(scaled_x_train, scaled_y_train, loaded_model, LEARNING_RATE, EPOCHS)

    trained_model.eval()
    
    tensor_x_test_data = from_np_to_tensor(scaled_x_test)
    with torch.no_grad():
        predicted = trained_model(tensor_x_test_data)
            
    np_predicted_array = from_tensor_to_np(predicted)
                                
    return r2_score(scaled_y_test, np_predicted_array)

def mlp_train_loop(x_train, y_train, loaded_model, learning_rate, epochs):

    loaded_model.train()
    optimizer = torch.optim.Adam(loaded_model.parameters(), lr = learning_rate)

    tensor_x_data = mlp.from_np_to_tensor(x_train)
    tensor_y_data = mlp.from_np_to_tensor(y_train)

    for epoch in range(epochs):

        optimizer.zero_grad()
        curr_prediction = loaded_model(tensor_x_data)
        loss = ((curr_prediction - tensor_y_data)**2).mean()
        
        loss.backward()
        optimizer.step()
        
    return loaded_model