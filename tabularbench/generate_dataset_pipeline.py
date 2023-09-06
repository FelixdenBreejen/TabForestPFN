import os
import numpy as np
from tabularbench.utils.keyword_to_function_conversion import convert_keyword_to_function
from sklearn.model_selection import train_test_split

from pathlib import Path
import pickle

#There are three steps to generate a dataset:
# 1) Generate x
# 2) Generate y
# 3) Transform x
# For each of this step, the user can provide the name of the method used (or a list of method for step 3),
# and a dictionary (or a list of dictionaries for step 3) with the different parameters


def generate_data(config, rng):
    method = convert_keyword_to_function[config["data__method_name"]]
    data_config = {}
    for key in config.keys():
        if key.startswith("data__") and key != "data__method_name":
            data_config[key[len("data__"):]] = config[key]
    data = method(**data_config, rng=rng)
    return data

def generate_target(x, config, rng):
    method = convert_keyword_to_function[config["target__method_name"]]
    target_config = {}
    for key in config.keys():
        if key.startswith("target__") and key != "target__method_name":
            target_config[key[len("target__"):]] = config[key]
    data = method(x, **target_config, rng=rng)
    return data

def transform_data(x_train, x_val, x_test, y_train, y_val, y_test, config, rng, categorical_indicator=None):
    i = 0
    print("transforming data...")
    while True:
        if f"transform__{i}__method_name" in config.keys():
            print("transform", i)
            method = convert_keyword_to_function[config[f"transform__{i}__method_name"]]
            if categorical_indicator is None:
                apply_on = "all"
            else:
                apply_on = config[f"transform__{i}__apply_on"]
            target_config = {}
            for key in config.keys():
                if key.startswith(f"transform__{i}__") and key != f"transform__{i}__method_name" and key != f"transform__{i}__apply_on":
                    target_config[key[len(f"transform__{i}__"):]] = config[key]
            if apply_on == "all":
                x_train, x_val, x_test, y_train, y_val, y_test = method(x_train, x_val, x_test, y_train, y_val, y_test, **target_config, rng=rng)
            elif apply_on == "numerical":
                if not np.all(categorical_indicator):
                    x_train[:, ~categorical_indicator], x_val[:, ~categorical_indicator], x_test[:, ~categorical_indicator], y_train, y_val, y_test = method(x_train[:, ~categorical_indicator], x_val[:, ~categorical_indicator], x_test[:, ~categorical_indicator], y_train, y_val, y_test, **target_config, rng=rng)
            elif apply_on == "categorical":
                if np.any(categorical_indicator):
                    x_train[:, categorical_indicator], x_val[:, categorical_indicator], x_test[:, categorical_indicator], y_train, y_val, y_test = method(x_train[:, categorical_indicator], x_val[:, categorical_indicator], x_test[:, categorical_indicator], y_train, y_val, y_test, **target_config, rng=rng)
        else:
            break
        i += 1

    return x_train, x_val, x_test, y_train, y_val, y_test

def data_to_train_test(x, y, config, rng=None):
    n_rows = x.shape[0]
    if "data__keyword" in config.keys() and config["data__keyword"] == "year": #TODO move this to a less hidden place
        if config["max_train_samples"] < 463715:
            indices_train = rng.choice(list(range(463715)), config["max_train_samples"],
                                             replace=False)
            x_train = x[indices_train]
            y_train = y[indices_train]
        else:
            x_train = x[:463715]
            y_train = y[:463715]

        x_val_test = x[463715:]
        y_val_test = y[463715:]
        i_val_test = (463715 + np.arange(x_val_test.shape[0])).tolist()
        x_val, x_test, y_val, y_test, i_val, i_test = train_test_split(x_val_test, y_val_test, i_val_test, train_size=config["val_test_prop"],
                                                        random_state=rng)
    else:
        if not config["max_train_samples"] is None:
            train_set_prop = min(config["max_train_samples"] / n_rows, config["train_prop"])
        else:
            train_set_prop = config["train_prop"]
        
        indices = list(range(n_rows))
        x_train, x_val_test, y_train, y_val_test, i_train, i_val_test = train_test_split(x, y, indices, train_size= train_set_prop, random_state=rng)
        x_val, x_test, y_val, y_test, i_val, i_test = train_test_split(x_val_test, y_val_test, i_val_test, train_size= config["val_test_prop"], random_state=rng)
    if not config["max_val_samples"] is None and x_val.shape[0] > config["max_val_samples"]:
        x_val = x_val[:config["max_val_samples"]]
        y_val = y_val[:config["max_val_samples"]]
        i_val = i_val[:config["max_val_samples"]]
    if not config["max_test_samples"] is None and x_test.shape[0] > config["max_test_samples"]:
        x_test = x_test[:config["max_test_samples"]]
        y_test = y_test[:config["max_test_samples"]]
        i_test = i_test[:config["max_test_samples"]]
    return x_train, x_val, x_test, y_train, y_val, y_test, i_train, i_val, i_test


def generate_dataset(config, rng, split_iter=None):
    data = generate_data(config, rng)
    if data is None:
        return None
    categorical_indicator = None
    if len(data) == 3:
        x, y, categorical_indicator = data
    #if "data__categorical" in config.keys() and config["data__categorical"]:
    #    x, y, categorical_indicator = data
    elif len(data) == 2: #if generate data returns x, y #TODO something cleaner
        x, y = data
        x = x.astype(np.float32) #FIXME
    else:
        x = data
        x = x.astype(np.float32)
        y = generate_target(x, config, rng)

    x_train, x_val, x_test, y_train, y_val, y_test, i_train, i_val, i_test = data_to_train_test(x, y, config, rng=rng)

    if split_iter is not None:
        indices_path = Path(f"data/train_val_test_indices.pkl")
        if not indices_path.exists():
            indices_path.parent.mkdir(parents=True, exist_ok=True)
            indices_path.touch()
            with open(indices_path, "wb") as f:
                pickle.dump({}, f)
        
        with open(indices_path, "rb") as f:
            indices_saved = pickle.load(f)

        openml_id = config['data__keyword']
        size = config['max_train_samples']

        if openml_id not in indices_saved.keys():
            indices_saved[openml_id] = {}
        if size not in indices_saved[openml_id].keys():
            indices_saved[openml_id][size] = []

        if split_iter == 0:
            indices_saved[openml_id][size] = []

        assert len(indices_saved[openml_id][size]) == split_iter

        indices_saved[openml_id][size].append((i_train, i_val, i_test))

        with open(indices_path, "wb") as f:
            pickle.dump(indices_saved, f)


    x_train, x_val, x_test, y_train, y_val, y_test = transform_data(x_train, x_val, x_test, y_train, y_val, y_test, config, rng,
                                                                    categorical_indicator=categorical_indicator)
    return x_train, x_val, x_test, y_train, y_val, y_test, categorical_indicator
