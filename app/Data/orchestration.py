import preprocessing as pp
import splitting as s
import transition_conversion as tc
import cpp_py_interface as cpi

def data_pipeline(
        num_trajectories: int, 
        num_samples: int, 
        seed: int = 0, 
        test_size_ratio: float = 0.3, 
        validation_strategy: s.Strategy = s.Strategy.GROUP_K_FOLD, 
        validation_size_ratio: float = 0.3, 
        k_fold_splits: int  = 3
        ):
    datasets = cpi.generate_datasets(num_trajectories, num_samples, seed) 

    converted_datasets = tc.convert_multiple_datasets(datasets)

    trajectories = []

    for dataset in converted_datasets:
        np_dataset = pp.to_np_arrays(dataset)
        model_inputs = pp.generate_model_inputs(np_dataset)
        model_targets = pp.generate_model_targets(np_dataset)
        trajectories.append((model_inputs, model_targets))

    test_train_split_data = s.test_split(trajectories, test_size_ratio, seed) 
    train_data, test_data = test_train_split_data[0], test_train_split_data[1]

    if validation_strategy == s.Strategy.GROUP_K_FOLD:
        folds = s.validation_split(train_data, validation_size_ratio, seed, k_fold_splits, validation_strategy)
    else:
        validation_split_data = s.validation_split(train_data, validation_size_ratio, seed, k_fold_splits, validation_strategy)
        x_y_train, x_y_validate = validation_split_data[0], validation_split_data[1]