import xarray as xr

from tabularbench.config.config_benchmark_sweep import ConfigBenchmarkSweep
from tabularbench.core.enums import DataSplit, ModelName, SearchType


def add_model_plot_names(ds: xr.Dataset) -> xr.Dataset:
    """
    Add model plot names to the dataset as additional variable.
    """

    ds = ds.copy()
    model_plot_names = [ ModelName[x].value for x in ds['model_name'].values ]
    ds['model_plot_name'] = xr.DataArray(model_plot_names, coords=dict(model_name=ds.coords['model_name']))
    return ds


def add_placeholder_as_model_name_dim(ds: xr.Dataset, model_plot_name: str) -> xr.Dataset:
    """
    For result datasets, they don't have a model dimension.
    This dimension is necessary for merging 
    """

    ds = ds.copy()
    var_names_with_run_id = get_var_names_depending_on_runs(ds)
    ds[var_names_with_run_id] = ds[var_names_with_run_id].expand_dims(dim='model_name').assign_coords({'model_name': [ModelName.PLACEHOLDER.name]})
    ds['model_plot_name'] = xr.DataArray([model_plot_name], coords=dict(model_name=ds.coords['model_name']))

    return ds


def select_only_the_first_default_run_of_every_model_and_dataset(cfg: ConfigBenchmarkSweep, ds: xr.Dataset) -> xr.Dataset:
    
    ds = ds.copy()
    var_names_with_run_id = get_var_names_depending_on_runs(ds)
    ds[var_names_with_run_id] = ds[var_names_with_run_id].where(ds['search_type'] == SearchType.DEFAULT.name, drop=True)
    ds[var_names_with_run_id] = ds[var_names_with_run_id].where(ds['seed'] == cfg.seed, drop=True) # when using multiple default runs, the seed changes
    ds = ds.isel(run_id=0).reset_coords('run_id', drop=True)

    return ds

def select_only_default_runs_and_average_over_them(ds: xr.Dataset) -> xr.Dataset:

    ds = ds.copy()
    vars_with_run_id = ['search_type', 'score', 'runs_actual']
    ds[vars_with_run_id] = ds[vars_with_run_id].where(ds['search_type'] == SearchType.DEFAULT.name, drop=True)
    ds = ds.mean(dim='run_id', keep_attrs=True)

    return ds


def average_out_the_cv_split(ds: xr.Dataset) -> xr.Dataset:
    """
    Average out the cv_split dimension.
    """

    ds = ds.copy()
    metric_vars = [var for var in ds.data_vars if 'cv_split' in ds[var].dims]

    if len(metric_vars) == 1:
        metric = metric_vars[0]
        ds[metric] = ds[metric] / ds['cv_splits_actual']
        mask = ds[metric].mean(dim='cv_split', skipna=True, keep_attrs=True)
        ds[metric] = ds[metric].sum(dim='cv_split', skipna=True, keep_attrs=True) 
        ds[metric] = ds[metric] * mask / mask
    else:
        # This assignment only works for multiple metric_vars for some reason
        ds[metric_vars] = ds[metric_vars] / ds['cv_splits_actual']
        mask = ds[metric_vars].mean(dim='cv_split', skipna=True, keep_attrs=True)
        # When using sum with skipna=True, it will set all nan values to zero when calculating
        # This is great when the max cv_split = 10 but we only have 3 splits: it will ignore the remaining 7
        # However, when we have zero splits (no runs) it will also set the value to zero, which is not what we want
        # So we need to set the value to nan again when sum outputs zero, which we do with the mean function
        ds = ds.sum(dim='cv_split', skipna=True, keep_attrs=True)
        ds[metric_vars] = ds[metric_vars] * mask / mask


    return ds


def only_use_models_and_datasets_specified_in_cfg(cfg: ConfigBenchmarkSweep, ds: xr.Dataset) -> xr.Dataset:

    benchmark_model_names = [model_name.name for model_name in cfg.plotting.get_benchmark_model_names(cfg.benchmark.origin)]
    ds = ds.sel(model_name=benchmark_model_names, openml_dataset_id=cfg.openml_dataset_ids_to_use)
    return ds


def get_var_names_depending_on_runs(ds: xr.Dataset) -> list[str]:
    """
    These are all the variable names in the dataset that depend on the number of runs.
    Most variables have a run_id dimension, but 'runs_actual' is also a variable that depends on the number of runs.
    """

    var_names = [ var_name for var_name in ds.data_vars if 'run_id' in ds[var_name].dims ]
    if 'runs_actual' in ds.data_vars:
        var_names.append('runs_actual')

    return var_names


def take_run_with_best_validation_loss(ds: xr.Dataset) -> xr.Dataset:
    """
    Take the run with the best validation loss.
    """

    loss = ds.sel(data_split=DataSplit.VALID.name)['log_loss']
    loss = loss.fillna(float('inf'))
    best_runs = loss.argmin('run_id').reset_coords('data_split', drop=True)
    ds = ds.sel(run_id=best_runs).reset_coords('run_id', drop=True)

    return ds