from __future__ import annotations
from pathlib import Path

import pandas as pd


class SweepConfig():

    def __init__(self, sweep_config: dict):

        self.model = sweep_config['model']
        self.plot_name = sweep_config['plot_name']
        self.task = sweep_config['task']
        self.suite_id = sweep_config['suite_id']
        self.benchmark = sweep_config['benchmark']
        self.random_search = sweep_config['random_search']
        self.runs_per_dataset = sweep_config['runs_per_dataset']
        self.dataset_size = sweep_config['dataset_size']
        self.path: Path = sweep_config['path']

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