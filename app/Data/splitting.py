from sklearn.model_selection import train_test_split, KFold
from enum import Enum
import transition_conversion as tc
import numpy as np

def test_split(trajectories: list[tuple[np.ndarray, np.ndarray]], test_size_ratio: float, seed: int) -> tuple[list[tuple[np.ndarray, np.ndarray]], list[tuple[np.ndarray, np.ndarray]]]:

    x_y_train, x_y_test = train_test_split(trajectories, test_size = test_size_ratio, random_state = seed)
    return (x_y_train, x_y_test)

class Strategy(Enum):
    HOLDOUT = 0
    GROUP_K_FOLD = 1

def validation_split(train_data: list[tuple[np.ndarray, np.ndarray]], validation_size_ratio: float, seed: int, splits: int, strategy: Strategy):

    match strategy:
        case Strategy.GROUP_K_FOLD:
            folds = []
            kf = KFold(n_splits = splits, shuffle = True, random_state = seed)
            for fold, (train_index, test_index) in enumerate(kf.split(train_data)):
                folds.append((train_index, test_index))
            return folds
        case _:
            return train_test_split(train_data, test_size = validation_size_ratio, random_state = seed)