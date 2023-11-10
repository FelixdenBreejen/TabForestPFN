from __future__ import annotations
from dataclasses import dataclass
from typing import Optional
import numpy as np
from omegaconf import DictConfig

import torch

from tabularbench.core.enums import DatasetSize, ModelName, Task, FeatureType, SearchType
from tabularbench.results.run_metrics import RunMetrics
from tabularbench.sweeps.config_run import ConfigRun


@dataclass
class RunResults():
    model: ModelName
    openml_task_id: int
    openml_dataset_id: int
    openml_dataset_name: str
    task: Task
    feature_type: FeatureType
    dataset_size: DatasetSize
    search_type: SearchType
    seed: int
    device: Optional[torch.device]
    scores_train: list[float]
    scores_val: list[float]
    scores_test: list[float]
    losses_train: list[float]
    losses_val: list[float]
    losses_test: list[float]
    hyperparams: DictConfig


    def __post_init__(self):

        self.score_train_mean = sum(self.scores_train) / len(self.scores_train)
        self.score_val_mean = sum(self.scores_val) / len(self.scores_val)
        self.score_test_mean = sum(self.scores_test) / len(self.scores_test)
        self.loss_train_mean = sum(self.losses_train) / len(self.losses_train)
        self.loss_val_mean = sum(self.losses_val) / len(self.losses_val)
        self.loss_test_mean = sum(self.losses_test) / len(self.losses_test)
        

    def to_dict(self):

        d = {
            'model': self.model.name,
            'openml_task_id': self.openml_task_id,
            'openml_dataset_id': self.openml_dataset_id,
            'openml_dataset_name': self.openml_dataset_name,
            'task': self.task.name,
            'feature_type': self.feature_type.name,
            'dataset_size': self.dataset_size.name,
            'search_type': self.search_type.name,
            'seed': self.seed,
            'device': str(self.device),
            'score_train_mean': self.score_train_mean,
            'score_val_mean': self.score_val_mean,
            'score_test_mean': self.score_test_mean,
            'loss_train_mean': self.loss_train_mean,
            'loss_val_mean': self.loss_val_mean,
            'loss_test_mean': self.loss_test_mean,
            'scores_train': self.scores_train,
            'scores_val': self.scores_val,
            'scores_test': self.scores_test,
            'losses_train': self.losses_train,
            'losses_val': self.losses_val,
            'losses_test': self.losses_test,
        }

        for key, value in self.hyperparams.items():
            d["hp__"+str(key)] = value

        return d
    

    @classmethod
    def from_dict(cls, d: dict) -> RunResults:
            
        hyperparams = {}
        for key, value in d.items():
            if key.startswith("hp__"):
                hyperparams[key[4:]] = value

        hyperparams_dc: DictConfig = DictConfig(hyperparams)
        
        return cls(
            model=d['model'],
            openml_task_id=d['openml_task_id'],
            openml_dataset_id=d['openml_dataset_id'],
            openml_dataset_name=d['openml_dataset_name'],
            task=d['task'],
            feature_type=d['feature_type'],
            dataset_size=d['dataset_size'],
            search_type=d['search_type'],
            seed=d['seed'],
            device=d['device'],
            scores_train=d['scores_train'],
            scores_val=d['scores_val'],
            scores_test=d['scores_test'],
            losses_train=d['losses_train'],
            losses_val=d['losses_val'],
            losses_test=d['losses_test'],
            hyperparams=hyperparams_dc,
        )
    

    @classmethod
    def from_benchmark_row(cls, row):

        if (
               not np.isfinite(row['max_train_samples']) 
            or not np.isfinite(row['data__regression'])
            or not np.isfinite(row['data__categorical'])
            or not np.isfinite(row['mean_val_score'])
            or not np.isfinite(row['mean_test_score'])
        ):
            print("Skipping row because of NaNs")
            return None

        feature_type = FeatureType.MIXED if row['data__categorical'] else FeatureType.NUMERICAL
        task = Task.REGRESSION if row['data__regression'] else Task.CLASSIFICATION
        search_type = SearchType.RANDOM if row['hp'] == 'random' else SearchType.DEFAULT
        dataset_size = DatasetSize(row['max_train_samples'])

        match task:
            case Task.CLASSIFICATION:
                if row['val_scores'] is np.nan or row['test_scores'] is np.nan:
                    train_scores = [row['mean_train_score']]
                    val_scores = [row['mean_val_score']]
                    test_scores = [row['mean_test_score']]
                else:
                    train_scores = [float(score) for score in row['train_scores'].strip('[]').split(',')]
                    val_scores = [float(score) for score in row['val_scores'].strip('[]').split(',')]
                    test_scores = [float(score) for score in row['test_scores'].strip('[]').split(',')]

            case Task.REGRESSION:
                train_scores = [row['mean_r2_train']]
                val_scores = [max(row['mean_r2_val'], 0)]
                test_scores = [max(row['mean_r2_test'], 0)]

                assert np.isfinite(val_scores[0])
                assert np.isfinite(test_scores[0])

        model_name_dict = {
            'MLP': ModelName.MLP,
            'FT Transformer': ModelName.FT_TRANSFORMER,
            'Resnet': ModelName.RESNET,
            'SAINT': ModelName.SAINT,
            'RandomForest': ModelName.RANDOM_FOREST,
            'XGBoost': ModelName.XGBOOST,
            'GradientBoostingTree': ModelName.GRADIENT_BOOSTING_TREE,
            'HistGradientBoostingTree': ModelName.HIST_GRADIENT_BOOSTING_TREE,
        }
        model = model_name_dict[row['model_name']]

        if any(score > 1 for score in train_scores+val_scores+test_scores):
            print(f"Scores above 1: {row['model_name'], row['data__keyword']}")

        
        return cls(
            model=model,
            openml_task_id=-1,
            openml_dataset_id=-1,
            openml_dataset_name=row['data__keyword'],
            task=task,
            feature_type=feature_type,
            dataset_size=dataset_size,
            search_type=search_type,
            seed=-1,
            device=row['model__device'],
            scores_train=train_scores,
            scores_val=val_scores,
            scores_test=test_scores,
            losses_train=[-1 for _ in train_scores],
            losses_val=[-1 for _ in val_scores],
            losses_test=[-1 for _ in test_scores],
            hyperparams={},
        )


    @classmethod
    def from_run_config(
        cls,
        cfg: ConfigRun, 
        search_type: SearchType,
        metrics: RunMetrics
    ) -> RunResults:

        return cls(
            model=cfg.model,
            openml_task_id=cfg.openml_task_id,
            openml_dataset_id=cfg.openml_dataset_id,
            openml_dataset_name=cfg.openml_dataset_name,
            task=cfg.task,
            feature_type=cfg.feature_type,
            dataset_size=cfg.dataset_size,
            search_type=search_type,
            seed=cfg.seed,
            device=cfg.device,
            scores_train=metrics.scores_train,
            scores_val=metrics.scores_val,
            scores_test=metrics.scores_test,
            losses_train=metrics.losses_train,
            losses_val=metrics.losses_val,
            losses_test=metrics.losses_test,
            hyperparams=cfg.hyperparams,
        )



