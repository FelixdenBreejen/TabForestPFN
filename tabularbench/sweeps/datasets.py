from __future__ import annotations
from pathlib import Path

import pandas as pd


def get_unfinished_dataset_ids(dataset_ids_all: list[int], results_path: Path, runs_per_dataset: int) -> list[int]:
    """Return a list of dataset ids that have not been run the required number of times."""

    if not results_path.exists():
        return dataset_ids_all
    
    results_df = pd.read_csv(results_path)
    
    datasets_run_count = results_df.groupby('openml_dataset_id').count()['model'].to_dict()

    datasets_unfinished = []
    for dataset_id in dataset_ids_all:
        if dataset_id not in datasets_run_count:
            datasets_unfinished.append(dataset_id)
        elif datasets_run_count[dataset_id] < runs_per_dataset:
            datasets_unfinished.append(dataset_id)

    return datasets_unfinished