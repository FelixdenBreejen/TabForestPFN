import functools

import numpy as np
import xarray as xr
from loguru import logger

from tabularbench.core.enums import BenchmarkOrigin, DataSplit, Task
from tabularbench.results.reformat_results_get import get_reformatted_results
from tabularbench.utils.config_benchmark_sweep import ConfigBenchmarkSweep


def get_combined_normalized_scores(cfg: ConfigBenchmarkSweep, openml_ids: list[int], data_split: DataSplit, scores: list[float]) -> float:
    """
    Based on a list of scores belonging to dataset ids, we compute the normalized test score.
    """

    normalized_scores = []

    for openml_dataset_id, score in zip(openml_ids, scores):
        score_min, score_max = scores_min_max(cfg, openml_dataset_id, data_split)
        normalized_score = (score - score_min) / (score_max - score_min)
        normalized_score = max(0.0, normalized_score)
        normalized_scores.append(normalized_score)

    combined_normalized_score = sum(normalized_scores) / len(normalized_scores)
    return combined_normalized_score


def normalize_scores(cfg: ConfigBenchmarkSweep, ds: xr.Dataset) -> xr.Dataset:
    """
    Normalize a score based on the benchmark results.
    """

    boundaries = np.full((2, ds.sizes['openml_dataset_id'], ds.sizes['data_split']), np.nan)

    for i_id, openml_dataset_id in enumerate(ds.coords['openml_dataset_id'].values):
        for i_split, data_split_str in enumerate(ds.coords['data_split'].values):

            data_split = DataSplit[data_split_str]
            boundaries[:, i_id, i_split] = scores_min_max(cfg, openml_dataset_id, data_split)

    boundaries = xr.DataArray(boundaries, dims=('min_max', 'openml_dataset_id', 'data_split'), coords=dict(min_max=['min', 'max']))

    ds = (ds - boundaries.sel(min_max='min')) / (boundaries.sel(min_max='max') - boundaries.sel(min_max='min'))
    ds = ds.where(ds >= 0.0, float('nan'))
    ds = ds.drop_vars('min_max')

    return ds



def scores_min_max(cfg: ConfigBenchmarkSweep, openml_dataset_id: int, data_split: DataSplit) -> tuple[float, float]:
    """
    Based on the benchmark results, we normalize the scores of the sweep.
    Returns the min and max scores to normalize with
    """

    benchmark_model_names = tuple(model_name.name for model_name in cfg.config_plotting.benchmark_model_names)
    score_min, score_max = scores_min_max_(openml_dataset_id, benchmark_model_names, cfg.benchmark.task, data_split, cfg.benchmark.origin)

    return score_min, score_max



@functools.lru_cache(maxsize=None)
def scores_min_max_(
        openml_dataset_id: int, 
        benchmark_model_names: tuple[str], 
        task: Task, 
        data_split: DataSplit, 
        benchmark_origin: BenchmarkOrigin
    ) -> tuple[float, float]:

    ds_benchmark = get_reformatted_results(benchmark_origin)
    ds_benchmark = ds_benchmark.sel(model_name=list(benchmark_model_names), openml_dataset_id=openml_dataset_id, data_split=data_split.name)

    match task:
        case Task.REGRESSION:
            score_min = ds_benchmark['score'].quantile(0.5).values.item()
        case Task.CLASSIFICATION:
            score_min = ds_benchmark['score'].quantile(0.1).values.item()

    score_max = ds_benchmark['score'].max().values.item()

    logger.info(f"For dataset id {openml_dataset_id} and split {data_split}, we will normalize with min {score_min:.4f} and max {score_max:.4f}")

    return score_min, score_max
