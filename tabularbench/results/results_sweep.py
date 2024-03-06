from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Self

import numpy as np
import xarray as xr
from loguru import logger

from tabularbench.core.enums import DataSplit
from tabularbench.results.results_run import ResultsRun


@dataclass
class ResultsSweep():
    ds: xr.Dataset

    @classmethod
    def from_run_results_dict(cls, run_results_dict: dict[int, list[ResultsRun]]) -> Self:

        run_results_dict = remove_ids_from_run_results_dict_that_have_no_runs(run_results_dict)
        
        ds = xr.Dataset(
            data_vars=make_data_vars_dict_with_empty_initialization(run_results_dict),
            coords=make_coords_dict(run_results_dict),
            attrs=make_attr_dict(run_results_dict)
        )

        for openml_dataset_id, results_runs in run_results_dict.items():
            for run_id, results_run in enumerate(results_runs):
                fill_ds_with_results_run(ds, results_run, openml_dataset_id, run_id)
                
        return cls(ds)
    

    def save(self, path: Path):
        self.ds.to_netcdf(path)
        runs_total = self.ds['runs_actual'].sum().item()
        logger.info(f"Saved results ({runs_total} runs total) to {path}")



def remove_ids_from_run_results_dict_that_have_no_runs(run_results_dict: dict[int, list[ResultsRun]]) -> dict[int, list[ResultsRun]]:

    run_results_dict2 = {}
    for dataset_id, results_runs in run_results_dict.items():
        if len(results_runs):
            run_results_dict2[dataset_id] = results_runs

    assert len(run_results_dict2) > 0, "No datasets with runs in run_results_dict"
    return run_results_dict2


def sample_run_result_from_run_results_dict(run_results_dict: dict[int, list[ResultsRun]]) -> ResultsRun:

    for results_runs in run_results_dict.values():
        if len(results_runs):
            return results_runs[0]

    raise ValueError("No runs in run_results_dict")


def make_data_vars_dict_with_empty_initialization(run_results_dict: dict[int, list[ResultsRun]]) -> dict[str, tuple]:

    n_datasets = len(run_results_dict)
    runs_actual = get_n_runs_of_all_datasets(run_results_dict)
    n_runs = max(runs_actual)
    cv_splits = get_n_cv_splits_of_all_datasets(run_results_dict)
    n_cv_splits = max(cv_splits)
    n_data_splits = len(DataSplit)

    metric_names = get_union_of_metric_names_of_first_run_result_of_all_datasets(run_results_dict)

    data_vars_dict = {
        "cv_splits_actual": (['openml_dataset_id'], cv_splits),
        "runs_actual": (['openml_dataset_id'], runs_actual),
        "seed": (['openml_dataset_id', 'run_id'], np.full((n_datasets, n_runs), -9999, dtype=int)),
        "search_type": (['openml_dataset_id', 'run_id'], np.full((n_datasets, n_runs), "", dtype=object)),
        "openml_dataset_name": (['openml_dataset_id'], np.full((n_datasets,), "", dtype=object)),
    }

    for metric_name in metric_names:
        data_vars_dict[metric_name] = (['openml_dataset_id', 'run_id', 'cv_split', 'data_split'], np.full((n_datasets, n_runs, n_cv_splits, n_data_splits), np.nan))


    return data_vars_dict


def get_union_of_metric_names_of_first_run_result_of_all_datasets(run_results_dict: dict[int, list[ResultsRun]]) -> list[str]:

    var_names = []
    for results_runs in run_results_dict.values():
        if len(results_runs):
            var_names.extend(results_runs[0].metrics.ds.data_vars)

    return var_names


def make_coords_dict(run_results_dict: dict[int, list[ResultsRun]]) -> dict[str, list]:
    
    runs_actual = get_n_runs_of_all_datasets(run_results_dict)
    n_runs = max(runs_actual)
    cv_splits_actual = get_n_cv_splits_of_all_datasets(run_results_dict)
    n_cv_splits = max(cv_splits_actual)

    coords_dict = {
        'openml_dataset_id': list(run_results_dict.keys()),
        'run_id': range(n_runs),
        'cv_split': range(n_cv_splits),
        'data_split': [data_split.value for data_split in DataSplit]
    }

    return coords_dict


def get_n_runs_of_all_datasets(run_results_dict: dict[int, list[ResultsRun]]) -> list[int]:

    runs_actual = []
    for results_runs in run_results_dict.values():
        runs_actual.append(len(results_runs))

    return runs_actual


def get_n_cv_splits_of_all_datasets(run_results_dict: dict[int, list[ResultsRun]]) -> list[int]:

    cv_splits_actual = []
    for results_runs in run_results_dict.values():
        cv_splits_actual.append(results_runs[0].metrics.ds.sizes['cv_split'])

    return cv_splits_actual


def make_attr_dict(run_results_dict: dict[int, list[ResultsRun]]) -> dict[str, str]:
    
    sample_run_result = sample_run_result_from_run_results_dict(run_results_dict)
    d = {
        'model_name': sample_run_result.model_name.value,
        'task': sample_run_result.task.value,
    }

    return d



def fill_ds_with_results_run(ds: xr.Dataset, results_run: ResultsRun, openml_dataset_id: int, run_id: int) -> None:

    ds['seed'].loc[dict(openml_dataset_id=openml_dataset_id, run_id=run_id)] = results_run.seed
    ds['search_type'].loc[dict(openml_dataset_id=openml_dataset_id, run_id=run_id)] = results_run.search_type.value
    ds['openml_dataset_name'].loc[dict(openml_dataset_id=openml_dataset_id)] = results_run.openml_dataset_name

    for var_name in results_run.metrics.ds.data_vars:
        n_cv_splits_this_run = results_run.metrics.ds.sizes['cv_split']
        ds[var_name].loc[dict(openml_dataset_id=openml_dataset_id, run_id=run_id, cv_split=slice(0, n_cv_splits_this_run))] = results_run.metrics.ds[var_name].values
