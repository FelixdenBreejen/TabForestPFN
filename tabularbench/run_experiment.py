import traceback  # Needed for pulling out your full stackframe info
import platform
import time
import torch
import sys
import inspect

from tabularbench.train import *
from tabularbench.generate_dataset_pipeline import generate_dataset


def modify_config(config):
    if config["model_name"] == "ft_transformer" or config["model_name"] == "ft_transformer_regressor":
        config["model__module__d_token"] = (config["d_token"] // config["model__module__n_heads"]) * config["model__module__n_heads"]
    for key in config.keys():
        if key.endswith("_temp"):
            new_key = "model__" + key[:-5]
            print("Replacing value from key", key, "to", new_key)
            if config[key] == "None":
                config[new_key] = None
            else:
                config[new_key] = config[key]
    
    return config

def debugger_is_active() -> bool:
    """Return if the debugger is currently active"""
    return hasattr(sys, 'gettrace') and sys.gettrace() is not None

def train_model_on_config(config=None) -> dict:
    print("GPU?")
    print(torch.cuda.device_count())
    print(torch.cuda.is_available())
    #    print(torch.cuda.current_device())
    #    print(torch.cuda.get_device_name(torch.cuda.current_device()))
    print("#####")
    CONFIG_DEFAULT = {"train_prop": 0.70,
                      "val_test_prop": 0.3,
                      "max_val_samples": 50000,
                      "max_test_samples": 50000}
    # "model__use_checkpoints": True} #TODO
   
    config = {**config, **CONFIG_DEFAULT}
    print(config)
    config = modify_config(config)

    if debugger_is_active():
        return train_model_config(config)
    
    try:
        return train_model_config(config)
    except Exception as e:
        # Print to the console
        print("ERROR")
        # To get the traceback information
        print(traceback.format_exc())
        print(config)

        if config["model_type"] == "skorch" and config["model__use_checkpoints"]:
            print("crashed, trying to remove checkpoint files")
            model_id = inspect.trace()[-1][0].f_locals['model_id']
            try:
                os.remove(r"skorch_cp/params_{}.pt".format(model_id))
            except:
                print("could not remove params file")
        if config["model_type"] == "tab_survey":
            print("Removing checkpoint files")
            print("Removing ")
            
            model_id = inspect.trace()[-1][0].f_locals['model_id']
            print(r"output/saint/{}/tmp/m_{}_best.pt".format(config["data__keyword"], model_id))
            #try:
            os.remove(r"output/saint/{}/tmp/m_{}_best.pt".format(config["data__keyword"], model_id))
            #except:
            #print("could not remove params file")

    return -1


def train_model_config(config) -> dict:


    train_scores = []
    val_scores = []
    test_scores = []
    r2_train_scores = []
    r2_val_scores = []
    r2_test_scores = []
    times = []
    if config["n_iter"] == "auto":
        x_train, x_val, x_test, y_train, y_val, y_test, categorical_indicator = generate_dataset(config, np.random.RandomState(0))
        if x_test.shape[0] > 6000:
            n_iter = 1
        elif x_test.shape[0] > 3000:
            n_iter = 2
        elif x_test.shape[0] > 1000:
            n_iter = 3
        else:
            n_iter = 5
    else:
        n_iter = config["n_iter"]
        
    for i in range(n_iter):
        if config["model_type"] in ['skorch', 'torch', 'tab_survey']:
            config_str = ".".join(list(str(a) for a in config.values()))  + "." + str(iter)
            model_id = hash(config_str)  # uniquely identify the run (useful for checkpointing)
        elif config["model_type"] == "sklearn":
            model_id = 0 # not used
        # if config["log_training"]: #FIXME
        #    config["model__wandb_run"] = run
        rng = np.random.RandomState(i)
        print(rng.randn(1))
        t = time.time()
        x_train, x_val, x_test, y_train, y_val, y_test, categorical_indicator = generate_dataset(config, rng, i)
        data_generation_time = time.time() - t
        print("Data generation time:", data_generation_time)
        # print(y_train)
        print(x_train.shape)

        if config["model_type"] == "skorch" and config["regression"]:
            print("YES")
            y_train, y_val, y_test = y_train.reshape(-1, 1), y_val.reshape(-1, 1), y_test.reshape(-1, 1)
            y_train, y_val, y_test = y_train.astype(np.float32), y_val.astype(np.float32), y_test.astype(
                np.float32)
        else:
            y_train, y_val, y_test = y_train.reshape(-1), y_val.reshape(-1), y_test.reshape(-1)
            # y_train, y_val, y_test = y_train.astype(np.float32), y_val.astype(np.float32), y_test.astype(np.float32)
        x_train, x_val, x_test = x_train.astype(np.float32), x_val.astype(np.float32), x_test.astype(
            np.float32)

        start_time = time.time()
        print(y_train.shape)
        model = train_model(i, x_train, y_train, categorical_indicator, config, model_id)

        if config["regression"]:
            try:
                r2_train, r2_val, r2_test = evaluate_model(model, x_train, y_train, x_val, y_val, x_test,
                                                            y_test, config, model_id, return_r2=True)
            except:
                print("R2 score cannot be computed")
                print(np.any(np.isnan(y_train)))
                r2_train, r2_val, r2_test = np.nan, np.nan, np.nan
            r2_train_scores.append(r2_train)
            r2_val_scores.append(r2_val)
            r2_test_scores.append(r2_test)
        else:
            r2_train, r2_val, r2_test = np.nan, np.nan, np.nan
            r2_train_scores.append(r2_train)
            r2_val_scores.append(r2_val)
            r2_test_scores.append(r2_test)
        train_score, val_score, test_score = evaluate_model(model, x_train, y_train, x_val, y_val, x_test,
                                                            y_test, config, model_id,  return_r2=True,)

        end_time = time.time()
        print("Train score:", train_score)
        print("Val score:", val_score)
        print("Test score:", test_score)

        times.append(end_time - start_time)
        train_scores.append(train_score)
        val_scores.append(val_score)
        test_scores.append(test_score)

    if "model__device" in config.keys():
        if config["model__device"] == "cpu":
            processor = platform.processor()
        elif config["model__device"].startswith("cuda"):
            processor = torch.cuda.get_device_name(torch.cuda.current_device())
        else:
            raise ValueError("Unknown device")
    else:
        processor = platform.processor()

    if n_iter > 1:
        config.update({"train_scores": train_scores,
                    "val_scores": val_scores,
                    "test_scores": test_scores,
                    "mean_train_score": np.mean(train_scores),
                    "mean_val_score": np.mean(val_scores),
                    "mean_test_score": np.mean(test_scores),
                    "std_train_score": np.std(train_scores),
                    "std_val_score": np.std(val_scores),
                    "std_test_score": np.std(test_scores),
                    "max_train_score": np.max(train_scores),
                    "max_val_score": np.max(val_scores),
                    "max_test_score": np.max(test_scores),
                    "min_train_score": np.min(train_scores),
                    "min_val_score": np.min(val_scores),
                    "min_test_score": np.min(test_scores),
                    "mean_r2_train": np.mean(r2_train_scores),
                    "mean_r2_val": np.mean(r2_val_scores),
                    "mean_r2_test": np.mean(r2_test_scores),
                    "std_r2_train": np.std(r2_train_scores),
                    "std_r2_val": np.std(r2_val_scores),
                    "std_r2_test": np.std(r2_test_scores),
                    "mean_time": np.mean(times),
                    "std_time": np.std(times),
                    "times": times,
                    "processor": processor})
    else:
        config.update({"mean_train_score": train_score,
                    "mean_val_score": val_score,
                    "mean_test_score": test_score,
                    "mean_r2_train": r2_train,
                    "mean_r2_val": r2_val,
                    "mean_r2_test": r2_test,
                    "mean_time": end_time - start_time,
                    "processor": processor})

    config.update({"n_train": x_train.shape[0], "n_test": x_test.shape[0],
                "n_features": x_train.shape[1],
                "data_generation_time": data_generation_time})   
    
    return config


if __name__ == """__main__""":
    
    config = {'data__categorical': True, 'data__method_name': 'openml_no_transform', 'data__regression': False, 'regression': False, 'n_iter': 'auto', 'max_train_samples': 10000, 'data__keyword': 361282, 'model__lr_scheduler': True, 'model__module__n_layers': 4, 'model__module__d_layers': 256, 'model__module__dropout': 0.0, 'model__lr': 0.001, 'model__module__d_embedding': 128, 'use_gpu': True, 'log_training': True, 'model__device': 'cuda', 'model_type': 'skorch', 'model__use_checkpoints': True, 'model__optimizer': 'adamw', 'model__batch_size': 512, 'model__max_epochs': 300, 'transform__0__method_name': 'gaussienize', 'transform__0__type': 'quantile', 'transform__0__apply_on': 'numerical', 'transformed_target': True, 'model_name': 'rtdl_mlp', 'hp': 'default', 'seed': 0, 'train_prop': 0.7, 'val_test_prop': 0.3, 'max_val_samples': 50000, 'max_test_samples': 50000}

    train_model_on_config(config)
