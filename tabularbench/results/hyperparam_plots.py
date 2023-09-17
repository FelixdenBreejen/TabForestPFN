from __future__ import annotations

import pandas as pd

from tabularbench.configs.all_model_configs import total_config
from tabularbench.sweeps.sweep_config import SweepConfig
from tabularbench.sweeps.paths_and_filenames import (
    RESULTS_MODIFIED_FILE_NAME
)


def make_hyperparam_plots(sweep: SweepConfig):
    
    df = pd.read_csv(sweep.path / RESULTS_MODIFIED_FILE_NAME)
    config = total_config[sweep.model][sweep.task]

    for dataset_name in sweep.dataset_names:
        for random_var in config['random'].keys():
                
            this_dataset = df['data__keyword'] == dataset_name
            fig = None

            if 'min' in config['random'][random_var]:
                is_log = 'log' in config['random'][random_var]['distribution']
                fig = df[this_dataset].plot(kind='scatter', x=random_var, y='mean_test_score', logx=is_log).get_figure()

            elif 'values' in config['random'][random_var]:
                fig = df[this_dataset].boxplot(column='mean_test_score', by=random_var).get_figure()

            if fig is not None:
                png_path = sweep.path / 'hyperparam_plots' / f'{dataset_name}_{random_var}.png'
                png_path.parent.mkdir(parents=True, exist_ok=True)
                fig.savefig(png_path)