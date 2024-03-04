from pathlib import Path
from typing import Optional

from loguru import logger
from omegaconf import DictConfig

from tabularbench.core.dataset_split import make_dataset_split
from tabularbench.core.enums import DatasetSize, ModelName, Task
from tabularbench.core.get_model import get_model
from tabularbench.core.get_trainer import get_trainer
from tabularbench.data.dataset_openml import OpenMLDataset
from tabularbench.results.run_metrics import RunMetrics
from tabularbench.utils.config_run import ConfigRun
from tabularbench.utils.debugger import debugger_is_active
from tabularbench.utils.set_seed import set_seed


def run_experiment(cfg: ConfigRun) -> Optional[RunMetrics]:

    cfg.save()

    logger.info(f"Start experiment on {cfg.openml_dataset_name} (id={cfg.openml_dataset_id}) with {cfg.model_name.value} doing {cfg.task.value}")

    set_seed(cfg.seed)
    logger.info(f"Set seed to {cfg.seed}")

    logger.info(f"We are using the following hyperparameters:")
    for key, value in cfg.hyperparams.items():
        logger.info(f"    {key}: {value}")

    if debugger_is_active():
        metrics = run_experiment_(cfg)
    else:
        try:
            metrics = run_experiment_(cfg)
        except Exception as e:
            logger.exception("Exception occurred while running experiment")        
            return None
    
    logger.info(f"Finished experiment on {cfg.openml_dataset_name} (id={cfg.openml_dataset_id}) with {cfg.model_name} doing {cfg.task.name}")
    logger.info(f"Final scores: ")

    for i in range(len(metrics)):
        logger.info(f"split_{i} :: train: {metrics.scores_train[i]:.4f}, val: {metrics.scores_val[i]:.4f}, test: {metrics.scores_test[i]:.4f}")

    score_train_avg = sum(metrics.scores_train) / len(metrics.scores_train)
    score_val_avg = sum(metrics.scores_val) / len(metrics.scores_val)
    score_test_avg = sum(metrics.scores_test) / len(metrics.scores_test)

    logger.info(f"average :: train: {score_train_avg:.4f}, val: {score_val_avg:.4f}, test: {score_test_avg:.4f}")

    return metrics


def run_experiment_(cfg: ConfigRun) -> RunMetrics:

    dataset = OpenMLDataset(cfg.datafile_path, cfg.task)
    metrics = RunMetrics()

    for split_i, (x_train, x_val, x_test, y_train, y_val, y_test, categorical_indicator) in enumerate(dataset.split_iterator()):

        logger.info(f"Start split {split_i+1}/{dataset.n_splits} of {cfg.openml_dataset_name} (id={cfg.openml_dataset_id}) with {cfg.model_name.name} doing {cfg.task.name}")

        x_val_hyperparams = x_val
        y_val_hyperparams = y_val

        x_train_cut, x_val_earlystop, y_train_cut, y_val_earlystop = make_dataset_split(x_train, y_train, task=cfg.task)

        model = get_model(cfg, x_train_cut, y_train_cut, categorical_indicator)
        trainer = get_trainer(cfg, model, dataset.n_classes)
        trainer.train(x_train_cut, y_train_cut, x_val_earlystop, y_val_earlystop)

        loss_train, score_train = trainer.test(x_train, y_train, x_train, y_train)
        loss_val, score_val = trainer.test(x_train, y_train, x_val_hyperparams, y_val_hyperparams)
        loss_test, score_test = trainer.test(x_train, y_train, x_test, y_test)

        logger.info(f"split_{split_i} :: train: {score_train:.4f}, val: {score_val:.4f}, test: {score_test:.4f}")

        metrics.append(score_train, score_val, score_test, loss_train, loss_val, loss_test)

    return metrics



if __name__ == "__main__":

    import torch



    cfg = ConfigRun(
        output_dir = Path("output_run_experiment"),
        device = torch.device("cuda:6"),
        model_name = ModelName.FOUNDATION,
        seed = 0,
        task = Task.CLASSIFICATION,
        dataset_size = DatasetSize.MEDIUM,
        openml_dataset_id = 44156,
        openml_dataset_name = "electricity",
        datafile_path = Path("data/datasets/whytrees_44156_MEDIUM.nc"),
        hyperparams = DictConfig({
            'n_features': 100,
            'n_classes': 10,
            'dim': 512,
            'n_layers': 12,
            'n_heads': 4,
            'attn_dropout': 0.0,
            'y_as_float_embedding': True,
            'linear_attention': False,
            'max_samples_support': 10000,
            'max_samples_query': 10000,
            'max_epochs': 300,
            'optimizer': 'adamw',
            'lr': 1.e-5,
            'weight_decay': 0,
            'lr_scheduler': False,
            'lr_scheduler_patience': 30,
            'early_stopping_patience': 40,
            'use_pretrained_weights': True,
            'path_to_weights': Path("outputs_done/foundation_forest_big_300k/weights/model_step_300000.pt"),
            'n_ensembles': 1,
            'use_quantile_transformer': True,
            'use_feature_count_scaling': True
        })
    )

    run_experiment(cfg)
