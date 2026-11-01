from Live.mlp_tuning import mlp_train_loop
from Live.config import LEARNING_RATE, LEARNING_BATCH_SIZE, EVALUATION_BATCH_SIZE, EPOCHS_PER_BATCH
from Live.session_state import SessionPhase
from Live.session_runner import inference
from Live.drift import DriftManager
from Models.polynomial import to_polynomial
from Training.model_type import Model
from sklearn.linear_model import SGDRegressor, LinearRegression
from sklearn.metrics import r2_score

import numpy as np
import copy
import torch

class OnlineAdapter:
    def __init__(
        self,
        drift_manager: DriftManager
    ):
        self.drift_manager = drift_manager
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
        updated_model = mlp_train_loop(self.x_learning_data, self.y_learning_data, self.training_model, self.optimizer, EPOCHS_PER_BATCH)
        self.drift_manager.loaded_model.loaded_model = copy.deepcopy(updated_model)

        self.currently_evaluating = True

    def adapt_linear_and_polynomial(self):
        coef_init = self.drift_manager.loaded_model.loaded_model.coef_.copy()
        intercept_init = self.drift_manager.loaded_model.loaded_model.intercept_.copy()

        if self.drift_manager.loaded_model.model_type == Model.POLYNOMIAL:
            polynomial_x_data = to_polynomial(self.x_learning_data)

        if self.initialized == False:
            for i, regressor in enumerate(self.sgd_regressors):
                if self.drift_manager.loaded_model.model_type == Model.POLYNOMIAL:
                    regressor.fit(polynomial_x_data, self.y_learning_data[:, i], coef_init = coef_init[i], intercept_init = np.asarray([intercept_init[i]]))
                else:
                    regressor.fit(self.x_learning_data, self.y_learning_data[:, i], coef_init = coef_init[i], intercept_init = np.asarray([intercept_init[i]]))
            self.initialized = True
        else:
            for i, regressor in enumerate(self.sgd_regressors):
                if self.drift_manager.loaded_model.model_type == Model.POLYNOMIAL:
                    regressor.partial_fit(polynomial_x_data, self.y_learning_data[:, i].reshape(-1))
                else:
                    regressor.partial_fit(self.x_learning_data, self.y_learning_data[:, i].reshape(-1))

        coef_init = np.stack([self.sgd_regressors[0].coef_, self.sgd_regressors[1].coef_, self.sgd_regressors[2].coef_])
        intercept_init = np.stack([self.sgd_regressors[0].intercept_[0], self.sgd_regressors[1].intercept_[0], self.sgd_regressors[2].intercept_[0]])

        temp_model = LinearRegression()
        temp_model.coef_, temp_model.intercept_ = coef_init, intercept_init

        self.drift_manager.loaded_model.loaded_model = temp_model

        self.currently_evaluating = True

    def evaluate_model(self):
        pre_drift_MAE = self.drift_manager.reference_mean

        prediction = inference(self.x_evaluating_data, self.drift_manager.loaded_model)
        residuals = self.y_evaluating_data - prediction

        score = r2_score(self.y_evaluating_data, prediction)

        post_adaption_MAE = np.mean(np.abs(residuals), axis = 0)

        if all((post_adaption_MAE <= (1.5 * pre_drift_MAE))) and score >= 0.90:
            self.get_new_CUSUM_reference_values(residuals)
            self.evaluation_batch_count = 0
        else:
            self.evaluation_batch_count = 0
            self.currently_evaluating = False

    def get_new_CUSUM_reference_values(self, residuals):
        std_dev = np.std(np.abs(residuals), axis = 0)

        self.drift_manager.reference_K = std_dev / 2
        self.drift_manager.reference_mean = np.mean(np.abs(residuals), axis = 0)
        self.drift_manager.current_CUSUM = np.array([0, 0, 0])

        candidate_thresholds = []

        for residual in residuals:
            abs_current_residual = np.abs(residual)
            self.drift_manager.CUSUM_calculation(abs_current_residual)
            candidate_thresholds.append(self.drift_manager.current_CUSUM)

        self.drift_manager.current_CUSUM = np.array([0, 0, 0])

        candidate_thresholds = np.asarray(candidate_thresholds)
        self.drift_manager.threshold = 2 * np.max(candidate_thresholds, axis = 0)

        self.reset()
        self.drift_manager.session_state.drift_status = SessionPhase.STABLE

    def reset(self):
        self.x_learning_data = np.empty((LEARNING_BATCH_SIZE, 5))
        self.y_learning_data = np.empty((LEARNING_BATCH_SIZE, 3))
        self.learning_batch_count = 0
        if self.drift_manager.loaded_model.model_type == Model.MLP:
            self.training_model = copy.deepcopy(self.drift_manager.loaded_model.loaded_model)
            self.optimizer = torch.optim.Adam(self.training_model.parameters(), lr = LEARNING_RATE)
        else:
            self.sgd_regressors = [SGDRegressor(loss="squared_error", penalty=None, learning_rate="constant", eta0=1e-4, max_iter=1, tol=None, shuffle=False) for i in range(3)]
            self.initialized = False
        self.x_evaluating_data = np.empty((EVALUATION_BATCH_SIZE, 5))
        self.y_evaluating_data = np.empty((EVALUATION_BATCH_SIZE, 3))
        self.evaluation_batch_count = 0
        self.currently_learning = False
        self.currently_evaluating = False