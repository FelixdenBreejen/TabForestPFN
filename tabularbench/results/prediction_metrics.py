from dataclasses import dataclass
from typing import Self

import numpy as np
import scipy
from sklearn.metrics import accuracy_score, f1_score, log_loss, mean_squared_error, r2_score, roc_auc_score

from tabularbench.core.enums import MetricName, Task


@dataclass
class PredictionMetrics():
    task: Task
    loss: float
    score: float
    metrics: dict[MetricName, float]


    @classmethod
    def from_prediction(cls, y_pred: np.ndarray, y_true: np.ndarray, task: Task) -> Self:

        loss, score, metrics = compute_metrics(y_pred, y_true, task)

        return PredictionMetrics(task=task, loss=loss, score=score, metrics=metrics)


    
def compute_metrics(y_pred: np.ndarray, y_true: np.ndarray, task: Task) -> tuple[float, float, dict]:

    match task:
        case Task.CLASSIFICATION:
            return compute_classification_metrics(y_pred, y_true)
        case Task.REGRESSION:
            return compute_regression_metrics(y_pred, y_true)
        

def compute_classification_metrics(y_pred: np.ndarray, y_true: np.ndarray) -> tuple[float, float, dict]:
    # predictions are assumed to be log-probabilities

    y_pred_class = np.argmax(y_pred, axis=1)
    y_pred_proba = scipy.special.softmax(y_pred, axis=1)
    labels = np.arange(y_pred_proba.shape[1])

    metrics = {
        MetricName.ACCURACY: accuracy_score(y_true, y_pred_class),
        MetricName.F1: f1_score(y_true, y_pred_class, average="weighted"),
        MetricName.AUC: roc_auc_score_multiclass(y_true, y_pred_proba, multi_class='ovo', average='macro', labels=labels),
        MetricName.LOG_LOSS: log_loss(y_true, y_pred_proba, labels=labels)
    }

    loss = metrics[MetricName.LOG_LOSS]
    score = metrics[MetricName.ACCURACY]

    return loss, score, metrics


def roc_auc_score_multiclass(y_true, y_pred_proba, multi_class='ovo', average='macro', labels=None) -> float:
    """ 
    The roc_auc_score multi_class is not supported for binary classification
    """

    if np.unique(y_true).shape[0] == 1:
        # AUC is not defined if there is only one class
        return float('nan')

    if y_pred_proba.shape[1] == 2:
        return roc_auc_score(y_true, y_pred_proba[:, 1])
    else:
        return roc_auc_score(y_true, y_pred_proba, multi_class=multi_class, average=average, labels=labels)


def compute_regression_metrics(y_pred: np.ndarray, y_true: np.ndarray) -> tuple[float, float, dict]:

    metrics = {
        MetricName.MSE: mean_squared_error(y_true, y_pred),
        MetricName.R2: r2_score(y_true, y_pred)
    }

    loss = metrics[MetricName.MSE]
    score = metrics[MetricName.R2]

    return loss, score, metrics