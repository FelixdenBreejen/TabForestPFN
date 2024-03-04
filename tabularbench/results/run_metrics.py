
from tabularbench.core.enums import DataSplit
from tabularbench.results.prediction_metrics import PredictionMetrics


class RunMetrics():
    """
    Metrics for a whole run, so multiple train/val/test splits.
    """

    def __init__(self):

        self.metrics = []


    def append(self, metrics_train: PredictionMetrics, metrics_val: PredictionMetrics, metrics_test: PredictionMetrics):

        self.metrics.append({
            DataSplit.TRAIN: metrics_train,
            DataSplit.VALID: metrics_val,
            DataSplit.TEST: metrics_test
        })


    def to_xarray(self):

        splits


