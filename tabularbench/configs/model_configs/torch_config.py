
# Default config for all skorch model
# Can be overwritten in the config file of a model

torch_config = {
    "log_training": {
        "value": True
    },
    "model__device": {
        "value": "cuda"
    },
    "model_type": {
        "value": "torch"
    },
    "model__use_checkpoints": {
        "value": True
    },
    "model__optimizer": {
        "value": "adamw"
    },
    "model__batch_size": {
        "values": [256, 512, 1024]
    },
    "model__max_epochs": {
        "value": 300
    },
    "transform__0__method_name": {
        "value": "gaussienize"
    },
    "transform__0__type": {
        "value": "quantile",
    },
    "transform__0__apply_on": {
        "value": "numerical",
    },
    "transformed_target": {
        "values": [False, True]
    },
    "use_gpu": {
        "value": True
    }
}

torch_config_default = torch_config.copy()
torch_config_default["model__batch_size"] = {"value": 512}
torch_config_default["transformed_target"] = {"value": True}