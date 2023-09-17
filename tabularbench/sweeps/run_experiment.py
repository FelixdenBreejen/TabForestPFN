import sys
from typing import Optional

import numpy as np
from tabularbench.core.enums import FeatureType, Task

from tabularbench.core.get_model import get_model

from tabularbench.core.trainer import Trainer
from tabularbench.data.dataset_openml import OpenMLDataset
from tabularbench.generate_dataset_pipeline import generate_dataset
from tabularbench.sweeps.run_config import RunConfig
from tabularbench.sweeps.sweep_start import set_seed



def run_experiment(cfg: RunConfig) -> Optional[tuple[dict, dict]]:

    cfg.logger.info(f"Start experiment on {cfg.openml_dataset_name} (id={cfg.openml_dataset_id}) with {cfg.model} doing {cfg.task.name} with {cfg.feature_type.name} features")
    
    set_seed(cfg.seed)
    cfg.logger.info(f"Seed is set to {cfg.seed}, device is {str(cfg.device)}")

    cfg.logger.info(f"We are using the following hyperparameters:")
    for key, value in cfg.hyperparams.items():
        cfg.logger.info(f"    {key}: {value}")


    if debugger_is_active():
        scores, losses = run_experiment_(cfg)
        return scores, losses

    try:
        scores, losses = run_experiment_(cfg)
    except Exception as e:
        cfg.logger.exception("Exception occurred while running experiment")        
        return None
    
    cfg.logger.info(f"Finished experiment on {cfg.openml_dataset_name} (id={cfg.openml_dataset_id}) with {cfg.model} doing {cfg.task.name} with {cfg.feature_type.name} features")
    cfg.logger.info(f"Final scores: ")

    for i, split in enumerate(scores):
        match split:
            case {'train': score_train, 'val': score_val, 'test': score_test}:
                cfg.logger.info(f"split_{i} :: train: {score_train:.4f}, val: {score_val:.4f}, test: {score_test:.4f}")

    return scores, losses
    

def debugger_is_active() -> bool:
    """Return if the debugger is currently active"""
    return hasattr(sys, 'gettrace') and sys.gettrace() is not None


def run_experiment_(cfg: RunConfig):
    dataset = OpenMLDataset(cfg.openml_dataset_id, cfg.task, cfg.feature_type, cfg.dataset_size)

    scores: dict[str, list[float]] = {
        "train": [],
        "val": [],
        "test": []
    }
    losses: dict[str, list[float]]  = {
        "train": [],
        "val": [],
        "test": []
    }

    for split_i, (batch) in enumerate(dataset.split_iterator()):

        config = {
            "data__categorical": cfg.feature_type == FeatureType.MIXED,
            "data__method_name": "openml_no_transform",
            "data__regression": cfg.task == Task.REGRESSION,
            "regression": cfg.task == Task.REGRESSION,
            "n_iter": "auto",
            "max_train_samples": 10000,
            "data__keyword": cfg.openml_task_id,
            "train_prop": 0.70,
            "val_test_prop": 0.3,
            "max_val_samples": 50000,
            "max_test_samples": 50000,
            "transform__0__method_name": "gaussienize",
            "transform__0__type": "quantile",
            "transform__0__apply_on": "numerical",
            "transformed_target": True,
        }


        rng = np.random.RandomState(split_i)
        print(rng.randn(1))
        x_train, x_val, x_test, y_train, y_val, y_test, categorical_indicator = generate_dataset(config, rng, split_i)
        x_train = x_train.astype(np.float32)
        x_val = x_val.astype(np.float32)
        x_test = x_test.astype(np.float32)

        cfg.logger.info(f"Start split {split_i+1}/{dataset.n_splits} of {cfg.openml_dataset_name} (id={cfg.openml_dataset_id}) with {cfg.model} doing {cfg.task.name} with {cfg.feature_type.name} features")

        model = get_model(cfg, x_train, y_train, categorical_indicator)
        trainer = Trainer(cfg, model)
        trainer.train(x_train, x_val, y_train, y_val)

        loss_train, score_train = trainer.test(x_train, y_train)
        loss_val, score_val = trainer.test(x_val, y_val)
        loss_test, score_test = trainer.test(x_test, y_test)

        scores["train"].append(score_train)
        scores["val"].append(score_val)
        scores["test"].append(score_test)

        losses["train"].append(loss_train)
        losses["val"].append(loss_val)
        losses["test"].append(loss_test)

    return scores, losses

