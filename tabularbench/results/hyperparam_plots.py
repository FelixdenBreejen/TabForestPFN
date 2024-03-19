from __future__ import annotations

import xarray as xr
from matplotlib import pyplot as plt

from tabularbench.core.enums import DataSplit
from tabularbench.results.dataset_manipulations import average_out_the_cv_split
from tabularbench.results.results_sweep import ResultsSweep
from tabularbench.utils.config_benchmark_sweep import ConfigBenchmarkSweep


def make_hyperparam_plots(cfg: ConfigBenchmarkSweep, results_sweep: ResultsSweep) -> None:

    ds = results_sweep.ds
    ds = ds.sel(data_split=DataSplit.VALID.name)
    ds = average_out_the_cv_split(ds)

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
                    fig = make_boxplot(ds_dataset, values, random_var_name)
                case _:
                    continue

            fig.suptitle(f'Hyperparameter {random_var} vs. validation score for dataset {dataset_id}')
            png_path = output_dir / f'hyperparam_plot_{random_var}.png'
            png_path.parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(png_path)
            plt.close(fig)


def make_boxplot(ds: xr.Dataset, values: list[str], random_var_name: str) -> plt.Figure:
    
    data = []
    for value in values:
        data.append(ds['score'].where(ds[random_var_name] == value).values)

    fig, ax = plt.subplots()
    ax.boxplot(data, labels=values)
    
    return fig