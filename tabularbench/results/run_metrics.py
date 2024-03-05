from pathlib import Path

import numpy as np
import xarray as xr

from tabularbench.core.enums import DataSplit
from tabularbench.results.prediction_metrics import PredictionMetrics


class RunMetrics():
    """
    Metrics for a whole run, so multiple train/val/test splits.
    """

    def __init__(self):

        self.metrics_by_split: list[dict[DataSplit, PredictionMetrics]] = []


    def append(self, metrics_train: PredictionMetrics, metrics_val: PredictionMetrics, metrics_test: PredictionMetrics):

        self.metrics_by_split.append({
            DataSplit.TRAIN: metrics_train,
            DataSplit.VALID: metrics_val,
            DataSplit.TEST: metrics_test
        })


    def post_process(self):

        metrics_dict = self.turn_metrics_by_split_into_dict_of_xarray_data_vars(self.metrics_by_split)

        self.ds = xr.Dataset(
            data_vars=metrics_dict,
            coords={
                'cv_split': range(len(self.metrics_by_split)),
                'data_split': [data_split.value for data_split in DataSplit]
            }
        )
    

    def turn_metrics_by_split_into_dict_of_xarray_data_vars(self, metrics_by_split: list[dict[DataSplit, PredictionMetrics]]) -> dict[str, np.ndarray]:

        metric_dict = {}

        n_splits = len(metrics_by_split)
        n_data_splits = len(DataSplit)

        metric_dict['score'] = (['cv_split', 'data_split'], np.zeros((n_splits, n_data_splits)))
        metric_dict['loss'] = (['cv_split', 'data_split'], np.zeros((n_splits, n_data_splits)))

        for metric_name in metrics_by_split[0][DataSplit.TRAIN].metrics.keys():
            metric_dict[metric_name.value] = (['cv_split', 'data_split'], np.zeros((n_splits, n_data_splits)))

        for cv_split_index, cv_split_metrics in enumerate(metrics_by_split):
            for data_split_index, data_split in enumerate(DataSplit):
                metrics = cv_split_metrics[data_split]
                metric_dict['score'][1][cv_split_index, data_split_index] = metrics.score
                metric_dict['loss'][1][cv_split_index, data_split_index] = metrics.loss
                for metric_name, metric_value in metrics.metrics.items():
                    metric_dict[metric_name.value][1][cv_split_index, data_split_index] = metric_value

        return metric_dict


    def save(self, filepath: Path):
        self.ds.to_netcdf(filepath)

