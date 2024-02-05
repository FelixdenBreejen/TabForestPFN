from __future__ import annotations

import pandas as pd
from tabularbench.utils.config_benchmark_sweep import ConfigBenchmarkSweep



def make_hyperparam_plots(cfg: ConfigBenchmarkSweep, df_run_results: pd.DataFrame) -> None:

    for dataset_id in cfg.openml_dataset_ids_to_use:

        df_dataset = df_run_results[ df_run_results['openml_dataset_id'] == dataset_id ]
        output_dir = cfg.output_dir / f'{dataset_id}'

        if len(df_dataset) == 0:
            # no results yet for this dataset id
            continue

        for random_var, settings in cfg.hyperparams_object.items():
            
            fig = None
            random_var_name = 'hp__' + random_var

            match settings:
                case {'distribution': distribution}:
                    is_log = 'log' in distribution
                    fig = df_dataset.plot(kind='scatter', x=random_var_name, y='score_val_mean', logx=is_log).get_figure()
                case {'values': _}:
                    fig = df_dataset.boxplot(column='score_val_mean', by=random_var_name).get_figure()
                case _:
                    continue

            fig.suptitle(f'Hyperparameter {random_var} vs. validation score for dataset {dataset_id}')
            png_path = output_dir / f'hyperparam_plot_{random_var}.png'
            png_path.parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(png_path)