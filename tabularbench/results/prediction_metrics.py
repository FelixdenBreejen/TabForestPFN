from dataclasses import dataclass
from typing import Self

import numpy as np
from sklearn.metrics import (accuracy_score, f1_score, log_loss,
                             mean_squared_error, r2_score, roc_auc_score)

from tabularbench.core.enums import Task


@dataclass
class PredictionMetrics():
    task: Task
    loss: float
    score: float
    metrics: dict


    @classmethod
    def from_prediction(cls, y_true: np.ndarray, y_pred: np.ndarray, task: Task) -> Self:

        loss, score, metrics = compute_metrics(y_true, y_pred, task)

        return PredictionMetrics(task=task, loss=loss, score=score, metrics=metrics)


    
def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray, task: Task) -> tuple[float, float, dict]:

    match task:
        case Task.CLASSIFICATION:
            return compute_classification_metrics(y_true, y_pred)
        case Task.REGRESSION:
            return compute_regression_metrics(y_true, y_pred)
        

def compute_classification_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> tuple[float, float, dict]:

    metrics = {
        "accuracy": accuracy_score(y_true, y_pred),
        "f1": f1_score(y_true, y_pred),
        "auc": roc_auc_score(y_true, y_pred),
        "log_loss": log_loss(y_true, y_pred)
    }

    loss = metrics["log_loss"]
    score = metrics["accuracy"]

    return loss, score, metrics


def compute_regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> tuple[float, float, dict]:

    metrics = {
        "mse": mean_squared_error(y_true, y_pred),
        "r2": r2_score(y_true, y_pred)
    }

    loss = metrics["mse"]
    score = metrics["r2"]

    return loss, score, metrics