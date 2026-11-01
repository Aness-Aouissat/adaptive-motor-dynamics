from Live.mlp_tuning import mlp_train_loop
from Live.config import LEARNING_RATE, LEARNING_BATCH_SIZE, EVALUATION_BATCH_SIZE, EPOCHS_PER_BATCH
from Live.session_state import SessionPhase
from Live.session_runner import inference
from Live.drift import DriftManager
from Models.polynomial import to_polynomial
from Training.model_type import Model
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from collections import deque

import numpy as np
import copy
import torch

class OnlineAdapter:
    def __init__(
        self,
        drift_manager: DriftManager
    ):
        self.drift_manager = drift_manager
        self.replay_buffer = deque(maxlen=1000)
        self.replay_rng = np.random.default_rng(42)
        self.reset()

    def maintain_adaption(self, model_input, model_target):
        if self.currently_evaluating == True:
            self.append_evaluation_data(model_input, model_target)
        elif self.currently_learning == False:
            self.append_learning_data(model_input, model_target)
        else:
            match self.drift_manager.loaded_model.model_type:
                case Model.MLP:
                    self.adapt_MLP()
                case _:
                    self.adapt_linear_and_polynomial()
            self.currently_learning = False
   
    def append_learning_data(self, model_input, model_target):
        self.replay_buffer.append((np.asarray(model_input).reshape(-1).copy(), np.asarray(model_target).reshape(-1).copy()))
        if self.learning_batch_count < LEARNING_BATCH_SIZE:
            self.x_learning_data[self.learning_batch_count] = self.drift_manager.loaded_model.x_scaler.transform(model_input).reshape(-1)
            self.y_learning_data[self.learning_batch_count] = self.drift_manager.loaded_model.y_scaler.transform(model_target).reshape(-1)
            self.learning_batch_count += 1

            if self.learning_batch_count == LEARNING_BATCH_SIZE:
                self.currently_learning = True
                self.learning_batch_count = 0

    def append_evaluation_data(self, model_input, model_target):
        if self.evaluation_batch_count == EVALUATION_BATCH_SIZE:
            self.evaluate_model()
            self.currently_evaluating = False
        elif self.evaluation_batch_count < EVALUATION_BATCH_SIZE:
            self.x_evaluating_data[self.evaluation_batch_count] = model_input.reshape(-1)
            self.y_evaluating_data[self.evaluation_batch_count] = model_target.reshape(-1)
            self.evaluation_batch_count += 1

    def adapt_MLP(self):
        indices = self.replay_rng.choice(len(self.replay_buffer), size=200, replace = False)

        training_inputs = []
        training_targets = []
        for index in indices:
            model_input, model_target = self.replay_buffer[int(index)]  
            training_inputs.append(model_input)
            training_targets.append(model_target)

        training_inputs = np.stack(training_inputs)
        training_targets = np.stack(training_targets)

        training_inputs = self.drift_manager.loaded_model.x_scaler.transform(training_inputs)
        training_targets = self.drift_manager.loaded_model.y_scaler.transform(training_targets)

        updated_model = mlp_train_loop(training_inputs, training_targets, self.training_model, self.optimizer, EPOCHS_PER_BATCH)
        self.drift_manager.loaded_model.loaded_model = copy.deepcopy(updated_model).eval()

        self.currently_evaluating = True

    def adapt_linear_and_polynomial(self):
        x_data = []
        y_data = []

        for model_input, model_target in self.replay_buffer:
            x_data.append(model_input)
            y_data.append(model_target)

        x_data = np.stack(x_data)
        y_data = np.stack(y_data)

        x_data = self.drift_manager.loaded_model.x_scaler.transform(x_data)
        y_data = self.drift_manager.loaded_model.y_scaler.transform(y_data)

        if self.drift_manager.loaded_model.model_type == Model.POLYNOMIAL:
            x_data = to_polynomial(x_data)

        updated_model = LinearRegression()
        updated_model.fit(x_data, y_data)

        self.drift_manager.loaded_model.loaded_model = updated_model
        self.currently_evaluating = True

    def evaluate_model(self):
        pre_drift_MAE = self.drift_manager.reference_mean

        prediction = inference(self.x_evaluating_data, self.drift_manager.loaded_model)
        residuals = self.y_evaluating_data - prediction

        score = r2_score(self.y_evaluating_data, prediction)

        post_adaption_MAE = np.mean(np.abs(residuals), axis = 0)

        mae_limits = 3 * pre_drift_MAE
        if self.drift_manager.loaded_model.model_type != Model.MLP:
            target_std = np.std(self.y_evaluating_data, axis=0)
            mae_limits = np.maximum(mae_limits, 0.01 * target_std)
        
        if all((post_adaption_MAE <= mae_limits)) and score >= 0.90:
            self.get_new_CUSUM_reference_values(residuals)
            self.evaluation_batch_count = 0
        else:
            self.evaluation_batch_count = 0
            self.currently_evaluating = False

    def get_new_CUSUM_reference_values(self, residuals):
        std_dev = np.std(np.abs(residuals), axis = 0)

        self.drift_manager.reference_K = std_dev / 2
        self.drift_manager.reference_mean = np.mean(np.abs(residuals), axis = 0)
        self.drift_manager.cusum_window.clear()
        self.drift_manager.current_CUSUM = np.zeros(3)
        match self.drift_manager.loaded_model.model_type:
            case Model.LINEAR:
                self.drift_manager.threshold = np.array([0.2216, 1e-12, 0.04])
            case Model.POLYNOMIAL:
                self.drift_manager.threshold = np.array([1, 2, 0.001526])
            case Model.MLP:
                self.drift_manager.threshold = np.array([26.58, 3.03, 0.0184])

        if self.drift_manager.loaded_model.model_type == Model.LINEAR:
            self.drift_manager.threshold[1] = max(self.drift_manager.threshold[1], 1e-7)

        self.reset()
        self.drift_manager.session_state.stable_step_counter = 0
        self.drift_manager.session_state.drift_status = SessionPhase.STABLE

    def reset(self):
        self.replay_buffer.clear()
        self.x_learning_data = np.empty((LEARNING_BATCH_SIZE, 5))
        self.y_learning_data = np.empty((LEARNING_BATCH_SIZE, 3))
        self.learning_batch_count = 0
        if self.drift_manager.loaded_model.model_type == Model.MLP:
            self.training_model = copy.deepcopy(self.drift_manager.loaded_model.loaded_model)
            self.optimizer = torch.optim.Adam(self.training_model.parameters(), lr = LEARNING_RATE)
        self.x_evaluating_data = np.empty((EVALUATION_BATCH_SIZE, 5))
        self.y_evaluating_data = np.empty((EVALUATION_BATCH_SIZE, 3))
        self.evaluation_batch_count = 0
        self.currently_learning = False
        self.currently_evaluating = False