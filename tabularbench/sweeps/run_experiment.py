import sys
from typing import Optional


from tabularbench.core.get_model import get_model

from tabularbench.core.trainer import Trainer
from tabularbench.data.dataset_openml import OpenMLDataset
from tabularbench.sweeps.run_config import RunConfig



def run_experiment(cfg: RunConfig) -> Optional[tuple[dict, dict]]:

    cfg.logger.info(f"Start experiment on {cfg.openml_dataset_name} (id={cfg.openml_dataset_id}) with {cfg.model} doing {cfg.task.name} with {cfg.feature_type.name} features")

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

    for split_i, (x_train, x_val, x_test, y_train, y_val, y_test, categorical_indicator) in enumerate(dataset.split_iterator()):

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

