import sys
from typing import Optional
from omegaconf import DictConfig

from tabularbench.core.enums import DatasetSize, FeatureType, Task
from tabularbench.core.get_model import get_model
from tabularbench.core.get_trainer import get_trainer

from tabularbench.data.dataset_openml import OpenMLDataset
from tabularbench.results.run_metrics import RunMetrics
from tabularbench.sweeps.config_run import ConfigRun
from tabularbench.sweeps.sweep_start import set_seed



def run_experiment(cfg: ConfigRun) -> Optional[RunMetrics]:

    cfg.logger.info(f"Start experiment on {cfg.openml_dataset_name} (id={cfg.openml_dataset_id}) with {cfg.model_name.value} doing {cfg.task.value} with {cfg.feature_type.value} features")

    set_seed(cfg.seed)
    cfg.logger.info(f"Set seed to {cfg.seed}")

    cfg.logger.info(f"We are using the following hyperparameters:")
    for key, value in cfg.hyperparams.items():
        cfg.logger.info(f"    {key}: {value}")


    if debugger_is_active():
        metrics = run_experiment_(cfg)
    else:
        try:
            metrics = run_experiment_(cfg)
        except Exception as e:
            cfg.logger.exception("Exception occurred while running experiment")        
            return None
    
    cfg.logger.info(f"Finished experiment on {cfg.openml_dataset_name} (id={cfg.openml_dataset_id}) with {cfg.model} doing {cfg.task.name} with {cfg.feature_type.name} features")
    cfg.logger.info(f"Final scores: ")

    for i in range(len(metrics)):
        cfg.logger.info(f"split_{i} :: train: {metrics.scores_train[i]:.4f}, val: {metrics.scores_val[i]:.4f}, test: {metrics.scores_test[i]:.4f}")

    return metrics
    

def debugger_is_active() -> bool:
    """Return if the debugger is currently active"""
    return hasattr(sys, 'gettrace') and sys.gettrace() is not None


def run_experiment_(cfg: ConfigRun) -> RunMetrics:

    dataset = OpenMLDataset(cfg.openml_dataset_id, cfg.task, cfg.feature_type, cfg.dataset_size)
    metrics = RunMetrics()

    for split_i, (x_train, x_val, x_test, y_train, y_val, y_test, categorical_indicator) in enumerate(dataset.split_iterator()):

        cfg.logger.info(f"Start split {split_i+1}/{dataset.n_splits} of {cfg.openml_dataset_name} (id={cfg.openml_dataset_id}) with {cfg.model.name} doing {cfg.task.name} with {cfg.feature_type.name} features")

        model = get_model(cfg, x_train, y_train, categorical_indicator)
        trainer = get_trainer(cfg, model)
        trainer.train(x_train, y_train)

        loss_train, score_train = trainer.test(x_train, y_train, x_train, y_train)
        loss_val, score_val = trainer.test(x_train, y_train, x_val, y_val)
        loss_test, score_test = trainer.test(x_train, y_train, x_test, y_test)

        metrics.append(score_train, score_val, score_test, loss_train, loss_val, loss_test)

    return metrics



if __name__ == "__main__":

    import logging
    import torch

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[
            logging.StreamHandler(sys.stdout)
        ]
    )

    cfg = ConfigRun(
        logger = logging.getLogger("run_experiment"),
        device = torch.device("cuda:5"),
        model = "ft_transformer",
        seed = 0,
        task = Task.CLASSIFICATION,
        feature_type = FeatureType.MIXED,
        dataset_size = DatasetSize.MEDIUM,
        openml_task_id = 361111,
        openml_dataset_id = 44157,
        openml_dataset_name = "eye-movements",
        hyperparams = DictConfig({
            'batch_size': 512,
            'max_epochs': 300,
            'optimizer': 'adamw',
            'lr': 1.e-4,
            'weight_decay': 1.e-5,
            'lr_scheduler': True,
            'lr_scheduler_patience': 30,
            'early_stopping_patience': 40,
            'd_token': 192,
            'activation': 'reglu',
            'token_bias': True,
            'prenormalization': True,
            'kv_compression': True,
            'kv_compression_sharing': 'headwise',
            'initialization': 'kaiming',
            'n_layers': 3,
            'n_heads': 8,
            'd_ffn_factor': 1.333,
            'ffn_dropout': 0.1,
            'attention_dropout': 0.2,
            'residual_dropout': 0.0
        })
    )

    results = run_experiment(cfg)
    assert results is not None
    scores, losses = results
    print(scores)
