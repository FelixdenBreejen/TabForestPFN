from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Self

import numpy as np
import torch
from loguru import logger
from omegaconf import DictConfig

from tabularbench.core.enums import (BenchmarkName, DatasetSize, FeatureType,
                                     ModelName, SearchType, Task)
from tabularbench.data.benchmarks import BENCHMARKS
from tabularbench.results.run_metrics import RunMetrics
from tabularbench.utils.config_run import ConfigRun


@dataclass
class ResultsRun():
    model_name: ModelName
    openml_dataset_id: int
    openml_dataset_name: str
    task: Task
    dataset_size: DatasetSize
    search_type: SearchType
    seed: int
    device: Optional[torch.device]
    metrics: RunMetrics
    hyperparams: DictConfig


    def to_dict(self):

        d = {
            'model': self.model_name.name,
            'openml_dataset_id': self.openml_dataset_id,
            'openml_dataset_name': self.openml_dataset_name,
            'task': self.task.name,
            'dataset_size': self.dataset_size.name,
            'search_type': self.search_type.name,
            'seed': self.seed,
            'device': str(self.device),
        }

        for key, value in self.hyperparams.items():
            d["hp__"+str(key)] = value

        return d
    

    @classmethod
    def from_dict(cls, d: dict) -> Self:
            
        hyperparams = {}
        for key, value in d.items():
            if key.startswith("hp__"):
                hyperparams[key[4:]] = value

        hyperparams_dc: DictConfig = DictConfig(hyperparams)
        
        return cls(
            model=d['model'],
            openml_dataset_id=d['openml_dataset_id'],
            openml_dataset_name=d['openml_dataset_name'],
            task=d['task'],
            dataset_size=d['dataset_size'],
            search_type=d['search_type'],
            seed=d['seed'],
            device=d['device'],
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
            logger.info("Skipping row because of NaNs")
            return None

        task = Task.REGRESSION if row['data__regression'] else Task.CLASSIFICATION
        search_type = SearchType.RANDOM if row['hp'] == 'random' else SearchType.DEFAULT
        feature_type = FeatureType.CATEGORICAL if row['data__categorical'] else FeatureType.NUMERICAL
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
        model_name = model_name_dict[row['model_name']]

        if any(score > 1 for score in train_scores+val_scores+test_scores):
            print(f"Scores above 1: {row['model_name'], row['data__keyword']}")


        match (task, dataset_size, feature_type):
            case (Task.CLASSIFICATION, DatasetSize.LARGE, FeatureType.CATEGORICAL):
                benchmark_name = BenchmarkName.CATEGORICAL_CLASSIFICATION_LARGE
            case (Task.CLASSIFICATION, DatasetSize.LARGE, FeatureType.NUMERICAL):
                benchmark_name = BenchmarkName.NUMERICAL_CLASSIFICATION_LARGE
            case (Task.CLASSIFICATION, DatasetSize.MEDIUM, FeatureType.CATEGORICAL):
                benchmark_name = BenchmarkName.CATEGORICAL_CLASSIFICATION
            case (Task.CLASSIFICATION, DatasetSize.MEDIUM, FeatureType.NUMERICAL):
                benchmark_name = BenchmarkName.NUMERICAL_CLASSIFICATION
            case (Task.REGRESSION, DatasetSize.LARGE, FeatureType.CATEGORICAL):
                benchmark_name = BenchmarkName.CATEGORICAL_REGRESSION_LARGE
            case (Task.REGRESSION, DatasetSize.LARGE, FeatureType.NUMERICAL):
                benchmark_name = BenchmarkName.NUMERICAL_REGRESSION_LARGE
            case (Task.REGRESSION, DatasetSize.MEDIUM, FeatureType.CATEGORICAL):
                benchmark_name = BenchmarkName.CATEGORICAL_REGRESSION
            case (Task.REGRESSION, DatasetSize.MEDIUM, FeatureType.NUMERICAL):
                benchmark_name = BenchmarkName.NUMERICAL_REGRESSION
        
        benchmark = BENCHMARKS[benchmark_name]

        dataset_name = row['data__keyword']

        assert dataset_name in benchmark.openml_dataset_names, f"Dataset {dataset_name} not in benchmark {benchmark_name}"
        index = benchmark.openml_dataset_names.index(dataset_name)
        dataset_id = benchmark.openml_dataset_ids[index]


        return cls(
            model_name=model_name,
            openml_dataset_id=dataset_id,
            openml_dataset_name=dataset_name,
            task=task,
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
    ) -> ResultsRun:

        return cls(
            model_name=cfg.model_name,
            openml_dataset_id=cfg.openml_dataset_id,
            openml_dataset_name=cfg.openml_dataset_name,
            task=cfg.task,
            dataset_size=cfg.dataset_size,
            search_type=search_type,
            seed=cfg.seed,
            device=cfg.device,
            metrics=metrics,
            hyperparams=cfg.hyperparams,
        )



