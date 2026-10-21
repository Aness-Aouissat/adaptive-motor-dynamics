from Training.config import SCALE_TECHNIQUE, HIDDEN_LAYERS
from Training.model_type import Model
from Data.preprocessing import Scaler
from Models import mlp
from pathlib import Path

import joblib
import torch

class LoadedModel:
    def __init__(
        self,
        model_type: Model
    ):
        self.model_type = model_type 
        self.loaded_model = None
        self.x_scaler = None
        self.y_scaler = None

    def load_model(self):

        curr_abs_path = Path(__file__).resolve()
        scalers_path = curr_abs_path.parent.parent / "Saved" / "scalers"

        if SCALE_TECHNIQUE != Scaler.NONE:
            self.x_scaler = joblib.load(scalers_path / "x_scaler.joblib")
            self.y_scaler = joblib.load(scalers_path / "y_scaler.joblib")
        
        match self.model_type:
            case Model.LINEAR:
                model_path = curr_abs_path.parent.parent / "Saved" / "models" / "Linear_Model.joblib"
                self.loaded_model = joblib.load(model_path)
            case Model.POLYNOMIAL:
                model_path = curr_abs_path.parent.parent / "Saved" / "models" / "Polynomial_Model.joblib"
                self.loaded_model = joblib.load(model_path)
            case Model.MLP:
                model_path = curr_abs_path.parent.parent / "Saved" / "models" / "MLP_Weights.pth"
                model_weights = torch.load(model_path, weights_only = True)
                loaded_model = mlp.MultiLayerPerceptron(5, HIDDEN_LAYERS, 3)
                loaded_model.load_state_dict(model_weights)
                self.loaded_model = loaded_model.eval()