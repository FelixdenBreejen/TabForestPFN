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
    "model__lr_scheduler": {
        "values": [True, False]
    }
}

config_default = {
    "model__lr": {
        "value": 5e-5
    },
    "model__optimizer__weight_decay": {
        "value": 0.
    },
    "model__lr_scheduler": {
        "value": True
    }
}

config_regression = dict(config_random,
                                        **torch_config,
                                        **{
                                            "model_name": {
                                                "value": "tab_pfn_regressor"
                                            },
                                        })

config_regression_default = dict(config_default,
                                        **torch_config_default,
                                        **{
                                            "model_name": {
                                                "value": "tab_pfn_regressor"
                                            },
                                        })

config_classif = dict(config_random,
                                     **torch_config,
                                     **{
                                         "model_name": {
                                             "value": "tab_pfn"
                                         },
                                     })

config_classif_default = dict(config_default,
                                     **torch_config_default,
                                     **{
                                         "model_name": {
                                             "value": "tab_pfn"
                                         },
                                     })