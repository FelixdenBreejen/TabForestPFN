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

        run_results_dict2 = {}
        for dataset_id, results_runs in run_results_dict.items():
            if len(results_runs):
                run_results_dict2[dataset_id] = results_runs

        run_results_dict = run_results_dict2


        all_runs_results = [results_run for results_runs in run_results_dict.values() for results_run in results_runs]
        sample_run_result = all_runs_results[0]

        n_datasets = len(run_results_dict)
        n_data_splits = len(DataSplit)
        var_names = list(set.union(*[set(run_results_dict[dataset_id][0].metrics.ds.keys()) for dataset_id in run_results_dict]))

        runs_actual = [len(run_results_dict[dataset_id]) for dataset_id in run_results_dict]
        n_runs = max(runs_actual)
        cv_splits = [run_results_dict[dataset_id][0].metrics.ds.sizes['cv_split'] for dataset_id in run_results_dict]
        n_cv_splits = max(cv_splits)

        data_vars_dict = {
            "cv_splits_actual": (['openml_dataset_id'], cv_splits),
            "runs_actual": (['openml_dataset_id'], runs_actual),
            "seed": (['openml_dataset_id', 'run_id'], np.full((n_datasets, n_runs), -9999, dtype=int)),
            "search_type": (['openml_dataset_id', 'run_id'], np.full((n_datasets, n_runs), "", dtype=object)),
            "openml_dataset_name": (['openml_dataset_id'], np.full((n_datasets,), "", dtype=object)),
        }

        for var_name in var_names:
            data_vars_dict[var_name] = (['openml_dataset_id', 'run_id', 'cv_split', 'data_split'], np.full((n_datasets, n_runs, n_cv_splits, n_data_splits), np.nan))

        ds = xr.Dataset(
            data_vars=data_vars_dict,
            coords={
                'openml_dataset_id': list(run_results_dict.keys()),
                'run_id': range(n_runs),
                'cv_split': range(n_cv_splits),
                'data_split': [data_split.value for data_split in DataSplit]
            },
            attrs={
                'model_name': sample_run_result.model_name.value,
                'task': sample_run_result.task.value,
                'dataset_size': sample_run_result.dataset_size.value,
            }
        )

        for openml_dataset_id, results_runs in run_results_dict.items():
            for run_id, results_run in enumerate(results_runs):
                
                ds['seed'].loc[dict(openml_dataset_id=openml_dataset_id, run_id=run_id)] = results_run.seed
                ds['search_type'].loc[dict(openml_dataset_id=openml_dataset_id, run_id=run_id)] = results_run.search_type.value
                ds['openml_dataset_name'].loc[dict(openml_dataset_id=openml_dataset_id)] = results_run.openml_dataset_name

                for var_name in results_run.metrics.ds.data_vars:
                    n_cv_splits_this_run = results_run.metrics.ds.sizes['cv_split']
                    ds[var_name].loc[dict(openml_dataset_id=openml_dataset_id, run_id=run_id, cv_split=slice(0, n_cv_splits_this_run))] = results_run.metrics.ds[var_name].values


        return cls(ds)
    

    def save(self, path: Path):
        self.ds.to_netcdf(path)
        logger.info(f"Saved results ({self.ds['runs_actual'].sum()} runs total) to {path}")