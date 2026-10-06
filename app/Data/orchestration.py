import preprocessing as pp
import splitting as s
import transition_conversion as tc
import cpp_py_interface as cpi
import numpy as np

def prepare_data(
        num_trajectories: int, 
        num_samples: int, 
        seed: int = 0,
):
    datasets = cpi.generate_datasets(num_trajectories, num_samples, seed) 
    
    converted_datasets = tc.convert_multiple_datasets(datasets)
    
    trajectories = []
    
    for dataset in converted_datasets:
        np_dataset = pp.to_np_arrays(dataset)
        model_inputs = pp.generate_model_inputs(np_dataset)
        model_targets = pp.generate_model_targets(np_dataset)
        trajectories.append((model_inputs, model_targets))

    return trajectories

def validate_pipeline(
        num_trajectories: int, 
        num_samples: int, 
        seed: int = 0, 
        test_size_ratio: float = 0.3, 
        validation_strategy: s.Strategy = s.Strategy.GROUP_K_FOLD, 
        validation_size_ratio: float = 0.3, 
        k_fold_splits: int  = 3,
        scale_technique: pp.Scaler = pp.Scaler.NONE,
        ):

    trajectories = prepare_data(num_trajectories, num_samples, seed)

    test_train_split_data = s.test_split(trajectories, test_size_ratio, seed) 
    train_data, test_data = test_train_split_data[0], test_train_split_data[1] 

    if validation_strategy == s.Strategy.GROUP_K_FOLD:
        folds = s.validation_split(train_data, validation_size_ratio, seed, k_fold_splits, validation_strategy) 
        scaled_folds = []
        for fold in folds:
            train_data, validate_data = fold[0], fold[1]

            x_train = np.empty((0, 5))
            y_train = np.empty((0, 3))
            for pair in train_data:
                x_train = np.concatenate((x_train, pair[0]), axis = 0)
                y_train = np.concatenate((y_train, pair[1]), axis = 0)

            x_validate = np.empty((0, 5))
            y_validate = np.empty((0, 3))
            for pair in validate_data:
                x_validate = np.concatenate((x_validate, pair[0]), axis = 0)
                y_validate = np.concatenate((y_validate, pair[1]), axis = 0)

            match scale_technique:
                case pp.Scaler.NORMALIZE:
                    x_scaler = pp.normalize(x_train)
                    y_scaler = pp.normalize(y_train)
                    
                    x_train = x_scaler.transform(x_train)
                    y_train = y_scaler.transform(y_train)
                    x_validate = x_scaler.transform(x_validate)
                    y_validate = y_scaler.transform(y_validate)

                    scaled_folds.append((x_train, y_train, x_validate, y_validate, x_scaler, y_scaler))
                case pp.Scaler.STANDARDIZE:                        
                    x_scaler = pp.standardize(x_train)
                    y_scaler = pp.standardize(y_train)
                    
                    x_train = x_scaler.transform(x_train)
                    y_train = y_scaler.transform(y_train)
                    x_validate = x_scaler.transform(x_validate)
                    y_validate = y_scaler.transform(y_validate)

                    scaled_folds.append((x_train, y_train, x_validate, y_validate, x_scaler, y_scaler))
                case _:
                    scaled_folds.append((x_train, y_train, x_validate, y_validate))

        return scaled_folds
    
    else:
        validation_split_data = s.validation_split(train_data, validation_size_ratio, seed, k_fold_splits, validation_strategy)
        train_data, validate_data = validation_split_data[0], validation_split_data[1]

        x_train = np.empty((0, 5))
        y_train = np.empty((0, 3))
        for pair in train_data:
            x_train = np.concatenate((x_train, pair[0]), axis = 0)
            y_train = np.concatenate((y_train, pair[1]), axis = 0)

        x_validate = np.empty((0, 5))
        y_validate = np.empty((0, 3))
        for pair in validate_data:
            x_validate = np.concatenate((x_validate, pair[0]), axis = 0)
            y_validate = np.concatenate((y_validate, pair[1]), axis = 0)
        
        match scale_technique:
            case pp.Scaler.NORMALIZE:
                x_scaler = pp.normalize(x_train)
                y_scaler = pp.normalize(y_train)

                x_train = x_scaler.transform(x_train)
                y_train = y_scaler.transform(y_train)
                x_validate = x_scaler.transform(x_validate)
                y_validate = y_scaler.transform(y_validate)

                return (x_train, y_train, x_validate, y_validate, x_scaler, y_scaler)
            case pp.Scaler.STANDARDIZE:
                x_scaler = pp.standardize(x_train)
                y_scaler = pp.standardize(y_train)                
                
                x_train = x_scaler.transform(x_train)
                y_train = y_scaler.transform(y_train)
                x_validate = x_scaler.transform(x_validate)
                y_validate = y_scaler.transform(y_validate)

                return (x_train, y_train, x_validate, y_validate, x_scaler, y_scaler)
            case _:
                return (x_train, y_train, x_validate, y_validate)

def train_pipeline(
        num_trajectories: int, 
        num_samples: int, 
        seed: int = 0, 
        test_size_ratio: float = 0.3, 
        scale_technique: pp.Scaler = pp.Scaler.NONE,
):
    trajectories = prepare_data(num_trajectories, num_samples, seed)

    test_train_split_data = s.test_split(trajectories, test_size_ratio, seed) 
    train_data, test_data = test_train_split_data[0], test_train_split_data[1] 

    x_train = np.empty((0, 5))
    y_train = np.empty((0, 3))
    for pair in train_data:
        x_train = np.concatenate((x_train, pair[0]), axis = 0)
        y_train = np.concatenate((y_train, pair[1]), axis = 0)

    x_test = np.empty((0, 5))
    y_test = np.empty((0, 3))
    for pair in test_data:
        x_test = np.concatenate((x_test, pair[0]), axis = 0)
        y_test = np.concatenate((y_test, pair[1]), axis = 0)
            
    match scale_technique:
        case pp.Scaler.NORMALIZE:
            x_scaler = pp.normalize(x_train)
            y_scaler = pp.normalize(y_train)

            x_train = x_scaler.transform(x_train)
            y_train = y_scaler.transform(y_train)
            x_test = x_scaler.transform(x_test)
            y_test = y_scaler.transform(y_test)

            return (x_train, y_train, x_test, y_test, x_scaler, y_scaler)
        case pp.Scaler.STANDARDIZE:
            x_scaler = pp.standardize(x_train)
            y_scaler = pp.standardize(y_train)

            x_train = x_scaler.transform(x_train)
            y_train = y_scaler.transform(y_train)
            x_test = x_scaler.transform(x_test)
            y_test = y_scaler.transform(y_test)

            return (x_train, y_train, x_test, y_test, x_scaler, y_scaler)
        case _:
            return (x_train, y_train, x_test, y_test)     