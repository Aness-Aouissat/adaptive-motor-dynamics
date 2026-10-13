import Data.orchestration as o
import Models.linear as l
import Models.polynomial as p
import Training.mlp_training as mlp
import Training.config as config
import torch
import joblib

from Models.mlp import from_np_to_tensor, from_tensor_to_np
from Data.preprocessing import StandardScaler, MinMaxScaler
from Training.model_type import Model
from sklearn.metrics import r2_score
from pathlib import Path

split_data = o.train_pipeline(20, 5000, 0, scale_technique = config.SCALE_TECHNIQUE) 

def test(trained_model, model_type: Model, x_test, y_test):

    match model_type:
        case Model.LINEAR:
            return trained_model.score(x_test, y_test)
        
        case Model.POLYNOMIAL:
            polynomial_x_test = p.to_polynomial(x_test)
            return trained_model.score(polynomial_x_test, y_test)
        
        case Model.MLP:
            tensor_x_test_data = from_np_to_tensor(x_test)
            predicted = trained_model(tensor_x_test_data)   
            np_predicted_array = from_tensor_to_np(predicted) 
            return r2_score(y_test, np_predicted_array)
        
        case _:
            return 

def store(trained_model, model: Model, x_scaler: StandardScaler | MinMaxScaler, y_scaler: StandardScaler | MinMaxScaler):

    curr_abs_path = Path(__file__).resolve()

    match model:
        case Model.LINEAR:
            model_path = curr_abs_path.parent.parent / "artifacts" / "models" / "Linear_Model.joblib"
            joblib.dump(trained_model, model_path)
        case Model.POLYNOMIAL:
            model_path = curr_abs_path.parent.parent / "artifacts" / "models" / "Polynomial_Model.joblib"
            joblib.dump(trained_model, model_path)
        case Model.MLP:
            model_path = curr_abs_path.parent.parent / "artifacts" / "models" / "MLP_Weights.pth"
            torch.save(trained_model.state_dict(), model_path)

    scalers_path = curr_abs_path.parent.parent / "artifacts" / "scalers"
    joblib.dump(x_scaler, scalers_path / "x_scaler.joblib")
    joblib.dump(y_scaler, scalers_path / "y_scaler.joblib")

def final_train_pipeline(split_data, model: Model):

    x_train, y_train = split_data[0], split_data[1]
    x_test, y_test = split_data[2], split_data[3]
    x_scaler, y_scaler = split_data[4], split_data[5]

    match model:
        case Model.LINEAR:
            trained_model = l.model.fit(x_train, y_train)

            print(test(trained_model, Model.LINEAR, x_test, y_test))
            decision = input("Store final Linear model? (y/n): ")
            
            if decision.lower() == "y":
                store(trained_model, Model.LINEAR, x_scaler, y_scaler)
            else:
                print("Model rejected. No files were saved.")
        
        case Model.POLYNOMIAL:
            polynomial_x_train = p.to_polynomial(x_train)
            trained_model = p.model.fit(polynomial_x_train, y_train) 
             
            print(test(trained_model, Model.POLYNOMIAL, x_test, y_test))
            decision = input("Store final Polynomial model? (y/n): ")

            if decision.lower() == "y":
                store(trained_model, Model.POLYNOMIAL, x_scaler, y_scaler)
            else:
                print("Model rejected. No files were saved.")

        case Model.MLP:
            torch.manual_seed(config.PYTORCH_SEED)
            trained_model = mlp.train_mlp(x_train, y_train)

            print(test(trained_model, Model.MLP, x_test, y_test))
            decision = input("Store final MLP model? (y/n): ")
            
            if decision.lower() == "y":
                store(trained_model, Model.MLP, x_scaler, y_scaler)
            else:
                print("Model rejected. No files were saved.")

        case _:
            return