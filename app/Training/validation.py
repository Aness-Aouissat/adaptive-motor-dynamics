import Data.orchestration as o
import Data.preprocessing as pp
import Models.linear as l
import Models.polynomial as p
import Training.mlp_training as mlp
import Training.config as config
import torch

from Models.mlp import from_np_to_tensor, from_tensor_to_np
from sklearn.metrics import r2_score
from enum import Enum

split_data = o.validate_pipeline(20, 5000, 0, scale_technique = pp.Scaler.STANDARDIZE) 

class Model(Enum):
    LINEAR = 0
    POLYNOMIAL = 1
    MLP = 2

def validate(split_data, model: Model):

    if type(split_data) == list:
        scores = []

        for fold in split_data:
            x_train, y_train = fold[0], fold[1]
            x_validate, y_validate = fold[2], fold[3]

            match model:
                case Model.LINEAR:
                    trained_model = l.model.fit(x_train, y_train)
                    predicted = trained_model.predict(x_validate)

                    scores.append(trained_model.score(x_validate, y_validate))

                case Model.POLYNOMIAL:
                    polynomial_x_train = p.to_polynomial(x_train)
                    trained_model = p.model.fit(polynomial_x_train, y_train)
                    polynomial_x_validate = p.to_polynomial(x_validate)
                    scores.append(trained_model.score(polynomial_x_validate, y_validate))

                case Model.MLP:
                    torch.manual_seed(config.SEED)
                    trained_model = mlp.train_mlp(x_train, y_train)
                    tensor_x_validate_data = from_np_to_tensor(x_validate)
                    predicted = trained_model(tensor_x_validate_data)

                    np_predicted_array = from_tensor_to_np(predicted)
                    
                    scores.append(r2_score(y_validate, np_predicted_array))

                case _:
                    return

        return scores
    
    else:
        x_train, y_train = split_data[0], split_data[1]
        x_validate, y_validate = split_data[2], split_data[3]

        match model:
            case Model.LINEAR:
                trained_model = l.model.fit(x_train, y_train)

                return trained_model.score(x_validate, y_validate)

            case Model.POLYNOMIAL:
                polynomial_x_train = p.to_polynomial(x_train)
                trained_model = p.model.fit(polynomial_x_train, y_train)
                polynomial_x_validate = p.to_polynomial(x_validate)

                return trained_model.score(polynomial_x_validate, y_validate)

            case Model.MLP:
                torch.manual_seed(config.SEED)
                trained_model = mlp.train_mlp(x_train, y_train)
                tensor_x_validate_data = from_np_to_tensor(x_validate)
                predicted = trained_model(tensor_x_validate_data)

                np_predicted_array = from_tensor_to_np(predicted)

                return r2_score(y_validate, np_predicted_array)

            case _:
                return