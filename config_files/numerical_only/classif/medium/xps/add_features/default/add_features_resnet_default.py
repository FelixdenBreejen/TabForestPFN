import wandb
import numpy as np

sweep_config = {
  "program": "run_experiment.py",
  "name" : "add_features_resnet_default",
  "project": "thesis-3",
  "method" : "grid",
  "metric": {
    "name": "mean_test_score",
    "goal": "maximize"
  },
  "parameters" : {
    "log_training": {
      "value": True
    },
    "model__device": {
      "value": "cpu"
    },
    "model_type": {
      "value": "skorch"
    },
    "model_name": {
      "value": "rtdl_resnet"
    },
    "model__use_checkpoints": {
      "value": True
    },
    "model__optimizer": {
      "value": "adamw"
    },
    "model__lr_scheduler": {
      "values": [True]
    },
    "model__batch_size": {
      "values": [512]
    },
    "model__max_epochs": {
      "value": 300
    },
    "model__module__activation": {
      "value": "reglu"
    },
    "model__module__normalization": {
      "values": ["batchnorm"]
    },
    "model__module__n_layers": {
      "value": 8,
    },
    "model__module__d": {
      "value": 256,
    },
    "model__module__d_hidden_factor": {
      "value": 2,
    },
    "model__module__hidden_dropout": {
      "value": 0.2,
    },
    "model__module__residual_dropout": {
      "value": 0.2
    },
    "model__lr": {
      "value": 1e-3,
    },
     "model__optimizer__weight_decay": {
      "value": 1e-7,
    },
    "model__module__d_embedding": {
      "value": 128
    },
    "data__method_name": {
      "value": "real_data"
    },
    "data__keyword": {
      "values": ["electricity",
                 "covertype",
                 # "poker",
                 "pol",
                 "house_16H",
                 "kdd_ipums_la_97-small",
                 "MagicTelescope",
                 "bank-marketing",
                 "phoneme",
                 "MiniBooNE",
                 "Higgs",
                 "eye_movements",
                 # "jannis",
                 "credit",
                 "california",
                 "wine"]
    },
    "transform__0__method_name": {
    "value": "add_uninformative_features"
    },
    "transform__0__multiplier": {
    "values": [1., 1.5, 2],
    },
    "transform__1__method_name": {
      "value": "gaussienize"
    },
    "transform__1__type": {
      "value": "quantile",
    },
    "n_iter": {
    "value": "auto",
    },
    "regression": {
    "value": False
    },
    "data__regression": {
    "value": False
    },
    "max_train_samples": {
    "value": 10000
    },
  }
}


sweep_id = wandb.sweep(sweep_config, project="thesis-3")