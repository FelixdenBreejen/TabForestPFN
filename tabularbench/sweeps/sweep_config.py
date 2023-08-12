from __future__ import annotations
from pathlib import Path
import openml

import pandas as pd


class SweepConfig():

    def __init__(self, sweep_config: dict):

        self.model = sweep_config['model']
        self.plot_name = sweep_config['plot_name']
        self.task = sweep_config['task']
        self.benchmark = sweep_config['benchmark']
        self.random_search = sweep_config['random_search']
        self.runs_per_dataset = sweep_config['runs_per_dataset']
        self.dataset_size = sweep_config['dataset_size']
        self.path: Path = sweep_config['path']
        
        self.suite_id = sweep_config['suite_id']
        self.task_ids = openml.study.get_suite(self.suite_id).tasks
        self.dataset_ids = [openml.tasks.get_task(id, download_data=False, download_splits=False, download_qualities=False, download_features_meta_data=False).dataset_id for id in self.task_ids]
        self.dataset_names = [openml.datasets.get_dataset(id, download_data=False, download_qualities=False, download_features_meta_data=False).name for id in self.dataset_ids]

        assert self.task in ['regression', 'classif']


def sweep_config_maker(sweep_csv: pd.DataFrame, output_dir) -> list[SweepConfig]:
    """
    From the sweep_csv file, create a list of SweepConfig objects.
    """

    sweeps = []

    for _, row in sweep_csv.iterrows():

        if row['random_search']:
            random_search_str = 'random'
        else:
            random_search_str = 'default'

        benchmark_path = Path(output_dir) / f"{row['benchmark']}_{random_search_str}_{row['model']}"

        sweep_config = row.to_dict()
        sweep_config['path'] = benchmark_path

        sweeps.append(SweepConfig(sweep_config))

    return sweeps