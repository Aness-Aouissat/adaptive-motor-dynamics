from Data.orchestration import train_pipeline
from Data.preprocessing import Scaler
from Live.session_state import SessionState, SessionPhase
from Live.loaded_model import LoadedModel
from Live.session_runner import inference
from Training.model_type import Model
from collections import deque

import numpy as np

class DriftManager:
    def __init__(
        self,
        session_state: SessionState,
        loaded_model: LoadedModel
    ):
        self.session_state = session_state
        self.loaded_model = loaded_model
        self.sliding_window = deque()
        self.reference_mean = None
        self.reference_K = None
        self.threshold = None
        self.current_CUSUM = np.array([0, 0, 0])

    def get_initial_CUSUM_reference_values(self):
        reference_data = train_pipeline(4, 750, 1, scale_technique = Scaler.NONE)
        x_test, y_test = reference_data[2], reference_data[3]

        predictions = []
        for i in range(len(x_test)):
            prediction = inference(x_test[i].reshape(1, -1), self.loaded_model)
            predictions.append(prediction)

        np_predictions = np.squeeze(np.array(predictions), axis = 1)
        residuals = y_test - np_predictions

        std_dev = np.std(np.abs(residuals), axis = 0)
        
        self.reference_K = std_dev / 2
        self.reference_mean = np.mean(np.abs(residuals), axis = 0)
        if self.loaded_model.model_type == Model.LINEAR:
            self.threshold = np.array([3.19229320e-01, 8.99822336e-13, 1.01346355e-01])
        elif self.loaded_model.model_type == Model.POLYNOMIAL:
            self.threshold = np.array([3.97069983e-13, 8.74652836e-13, 1.46010025e-14])
        elif self.loaded_model.model_type == Model.MLP:
            self.threshold = np.array([0.41661246, 0.28349883, 0.00623767])

    def CUSUM(self):
        current_residual = self.sliding_window[len(self.sliding_window) - 1]
        abs_current_residual = np.abs(current_residual)
        self.current_CUSUM = np.maximum(0, self.current_CUSUM + abs_current_residual - self.reference_mean - self.reference_K)

    def drift_detector(self):
        if self.session_state.drift_status != SessionPhase.ADAPTING:
            self.CUSUM()
            if (self.current_CUSUM > self.threshold).any():
                return True

    def update_drift_status(self):
        if self.drift_detector() == True and self.session_state.drift_status != SessionPhase.ADAPTING:
            self.session_state.drift_status = SessionPhase.ADAPTING

    def maintain_sliding_window(self):
        if len(self.sliding_window) > 1000:
            self.sliding_window.popleft()
        else:
            pass