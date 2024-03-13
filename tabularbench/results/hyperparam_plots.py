from __future__ import annotations

from matplotlib import pyplot as plt

from tabularbench.core.enums import DataSplit
from tabularbench.results.dataset_manipulations import apply_mean_over_cv_split
from tabularbench.results.results_sweep import ResultsSweep
from tabularbench.utils.config_benchmark_sweep import ConfigBenchmarkSweep


def make_hyperparam_plots(cfg: ConfigBenchmarkSweep, results_sweep: ResultsSweep) -> None:

    ds = results_sweep.ds
    ds = ds.sel(data_split=DataSplit.VALID.name)
    ds = apply_mean_over_cv_split(ds)

    for dataset_id in cfg.openml_dataset_ids_to_use:

        if dataset_id not in ds['openml_dataset_id']:
            # no results yet for this dataset id
            continue

        ds_dataset = ds.sel(openml_dataset_id=dataset_id)
        ds_dataset = ds_dataset.dropna('run_id', how='any')
        output_dir = cfg.output_dir / f'{dataset_id}'

        for random_var, settings in cfg.hyperparams_object.items():
            
            fig = None
            random_var_name = 'hp_' + random_var

            match settings:
                case {'distribution': distribution}:
                    xscale = 'log' if 'log' in distribution else 'linear'
                    fig = ds_dataset.plot.scatter(x=random_var_name, y='score', xscale=xscale).get_figure()
                case {'values': values}:
                    data = []
                    for value in values:
                        data.append(ds_dataset['score'].where(ds_dataset[random_var_name] == value).values)

                    fig, ax = plt.subplots()
                    ax.boxplot(data, labels=values)
                case _:
                    continue

            fig.suptitle(f'Hyperparameter {random_var} vs. validation score for dataset {dataset_id}')
            png_path = output_dir / f'hyperparam_plot_{random_var}.png'
            png_path.parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(png_path)
            plt.close(fig)