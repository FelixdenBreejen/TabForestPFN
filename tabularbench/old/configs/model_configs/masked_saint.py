from tabularbench.configs.model_configs.torch_config import torch_config, torch_config_default

config_random = {
    "model__lr": {
        "distribution": "log_uniform_values",
        "min": 1e-7,
        "max": 1e-3
    },
    "model__optimizer__weight_decay": {
        "distribution": "log_uniform_values",
        "min": 1e-7,
        "max": 1e-1
    },
    "model__batch_size": {
        "value": 10000
    },
    "model__n_ensembles": {
        "value": 1
    },
    "model__lr_scheduler": {
        "values": [True, False]
    },
    "model__finetune": {
        "value": True
    },
    "model__use_checkpoints": {
        "value": True
    },
    "model__use_pretrained_weights": {
        "value": True
    },
    "model__pretrained_weights_path": { 
        "value": "outputs/2023-09-03/10-23-33/epoch_299_weights.pt"
    },
}

config_default = {
    "model__lr": {
        "value": 1.e-5
    },
    "model__optimizer__weight_decay": {
        "value": 0.
    },
    "model__batch_size": {
        "value": 1000
    },
    "model__n_ensembles": {
        "value": 1
    },
    "model__lr_scheduler": {
        "value": True
    },
    "model__finetune": {
        "value": True
    },
    "model__use_checkpoints": {
        "value": True
    },
    "model__use_pretrained_weights": {
        "value": True
    },
    "model__pretrained_weights_path": { 
        "value": "outputs/2023-09-03/10-23-33/epoch_299_weights.pt"
    },
}

config_model_name_regressor = {
    "model_name": {
        "value": "masked_saint_regressor"
    },
}

config_model_name_classif = {
    "model_name": {
        "value": "masked_saint"
    },
}

config_regression = {
    **torch_config,
    **config_model_name_regressor,
    **config_random,
}

config_regression_default = {
    **torch_config_default,
    **config_model_name_regressor,
    **config_default,
}

config_classif = {
    **torch_config,
    **config_model_name_classif,
    **config_random,
}

config_classif_default = {
    **torch_config_default,
    **config_model_name_classif,
    **config_default,
}