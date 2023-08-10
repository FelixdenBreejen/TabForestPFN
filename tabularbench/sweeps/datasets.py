from __future__ import annotations
from pathlib import Path

import pandas as pd


def get_unfinished_task_ids(task_ids_all: list[int], results_path: Path, runs_per_dataset: int) -> list[int]:
    """Return a list of dataset ids that have not been run the required number of times."""

    if not results_path.exists():
        return task_ids_all
    
    results_df = pd.read_csv(results_path)
    tasks_run_count = results_df.groupby('data__keyword').count()['data__categorical'].to_dict()

    tasks_unfinished = []
    for task_id in task_ids_all:
        if task_id not in tasks_run_count:
            tasks_unfinished.append(task_id)
        elif tasks_run_count[task_id] < runs_per_dataset:
            tasks_unfinished.append(task_id)

    return tasks_unfinished