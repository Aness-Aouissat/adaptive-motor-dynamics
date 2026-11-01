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

        self.cusum_window = deque(maxlen = 100)
        self.cusum_reference_mean = None
        self.cusum_reference_K = None
        self.cusum_threshold = None
        self.current_CUSUM = np.array([0, 0, 0])

        self.page_hinkley_lambda_threshold = np.array([0.7339500779, 1e-9, 1e-9])
        self.page_hinkley_delta = 0.5
        self.reset_page_hinkley()

    def update_page_hinkley(self):
        residual = np.asarray(self.sliding_window[-1]).reshape(3)

        scale = np.maximum(self.cusum_reference_mean, np.array([0.08, 0.08, 0.0008]))
        value = np.log(1 + np.abs(residual) / scale)

        self.page_hinkley_count += 1
        self.page_hinkley_running_mean = self.page_hinkley_running_mean + (value - self.page_hinkley_running_mean) / self.page_hinkley_count

        self.page_hinkley_running_sum += value - self.page_hinkley_running_mean - self.page_hinkley_delta
        self.page_hinkley_running_min_sum = np.minimum(self.page_hinkley_running_min_sum, self.page_hinkley_running_sum)

        ph_statistic = self.page_hinkley_running_sum - self.page_hinkley_running_min_sum

        return (ph_statistic > self.page_hinkley_lambda_threshold).any()

    def reset_page_hinkley(self):
        self.page_hinkley_count = 0
        self.page_hinkley_running_mean = np.zeros(3)
        self.page_hinkley_running_sum = np.zeros(3)
        self.page_hinkley_running_min_sum = np.zeros(3)

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
        
        self.cusum_reference_K = std_dev / 2
        self.cusum_reference_mean = np.mean(np.abs(residuals), axis = 0)
        if self.loaded_model.model_type == Model.LINEAR:
            self.cusum_threshold = np.array([3.19229320e-01, 8.99822336e-13, 1.01346355e-01])
        elif self.loaded_model.model_type == Model.POLYNOMIAL:
            self.cusum_threshold = np.array([3.97069983e-13, 8.74652836e-13, 1.46010025e-14])
        elif self.loaded_model.model_type == Model.MLP:
            self.cusum_threshold = np.array([0.41661246, 0.28349883, 0.00623767])

    def CUSUM(self):
        current_residual = self.sliding_window[len(self.sliding_window) - 1]
        abs_current_residual = np.abs(current_residual)
        self.CUSUM_calculation(abs_current_residual)

    def CUSUM_calculation(self, abs_current_residual):
        residual = abs_current_residual.reshape(3)
        excess = residual - self.cusum_reference_mean - self.cusum_reference_K

        self.cusum_window.append(excess)

        running_sum = np.zeros(3)
        maximum_CUSUM = np.zeros(3)
        for value in list(self.cusum_window)[::-1]:
            running_sum += value
            maximum_CUSUM = np.maximum(maximum_CUSUM, running_sum)

        self.current_CUSUM = maximum_CUSUM

    def cusum_drift_detector(self):
        if self.session_state.drift_status != SessionPhase.ADAPTING:
            self.CUSUM()
            if (self.current_CUSUM > self.cusum_threshold).any():
                return True

    def update_drift_status(self):
        if self.session_state.drift_status == SessionPhase.ADAPTING:
            return

        if self.loaded_model.model_type == Model.POLYNOMIAL and self.session_state.drift_status == SessionPhase.STABLE:
            is_drift = self.update_page_hinkley()
        else:
            is_drift = self.cusum_drift_detector()

        if is_drift:
            self.session_state.drift_status = SessionPhase.ADAPTING

    def maintain_sliding_window(self):
        if len(self.sliding_window) > 1000:
            self.sliding_window.popleft()
        else:
            pass