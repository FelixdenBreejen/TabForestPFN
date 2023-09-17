from __future__ import annotations

import pandas as pd

from tabularbench.sweeps.sweep_config import SweepConfig
from tabularbench.sweeps.paths_and_filenames import (
    RESULTS_FILE_NAME
)


def make_hyperparam_plots(sweep: SweepConfig):
    
    df = pd.read_csv(sweep.sweep_dir / RESULTS_FILE_NAME)

    for dataset_name in sweep.openml_dataset_names:
        for random_var, settings in sweep.hyperparams.items():
                
            this_dataset = df['openml_dataset_name'] == dataset_name
            fig = None
            random_var_name = 'hp__' + random_var

            match settings:
                case {'distribution': distribution}:
                    is_log = 'log' in distribution
                    fig = df[this_dataset].plot(kind='scatter', x=random_var_name, y='score_test_mean', logx=is_log).get_figure()
                case {'values': _}:
                    fig = df[this_dataset].boxplot(column='score_test_mean', by=random_var_name).get_figure()
                    

            if fig is not None:
                png_path = sweep.sweep_dir / 'hyperparam_plots' / f'{dataset_name}_{random_var_name}.png'
                png_path.parent.mkdir(parents=True, exist_ok=True)
                fig.savefig(png_path)