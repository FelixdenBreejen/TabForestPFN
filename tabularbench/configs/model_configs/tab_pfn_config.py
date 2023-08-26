from tabularbench.configs.model_configs.torch_config import torch_config, torch_config_default

config_random = {
    "model__lr": {
        "distribution": "log_uniform_values",
        "min": 1e-6,
        "max": 1e-3
    },
    "model__optimizer__weight_decay": {
        "value": 0.
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
    }
}

config_default = {
    "model__lr": {
        "value": 0.
    },
    "model__optimizer__weight_decay": {
        "value": 0.
    },
    "model__batch_size": {
        "value": 10000
    },
    "model__n_ensembles": {
        "value": 1
    },
    "model__lr_scheduler": {
        "value": True
    },
    "model__finetune": {
        "value": False
    },
    "model__use_checkpoints": {
        "value": False
    }
}

config_model_name_regressor = {
    "model_name": {
        "value": "tab_pfn_regressor"
    },
}

config_model_name_classif = {
    "model_name": {
        "value": "tab_pfn"
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