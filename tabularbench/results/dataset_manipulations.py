import xarray as xr

from tabularbench.core.enums import ModelName, SearchType


def add_model_plot_names(ds: xr.Dataset):
    """
    Add model plot names to the dataset as additional variable.
    """

    model_plot_names = [ ModelName[x].value for x in ds['model_name'].values ]
    ds['model_plot_name'] = xr.DataArray(model_plot_names, coords=dict(model_name=ds.coords['model_name']))


def add_placeholder_as_model_name_dim(ds: xr.Dataset, model_plot_name: str):
    """
    For result datasets, they don't have a model dimension.
    This dimension is necessary for merging 
    """

    var_names_with_run_id = get_var_names_depending_on_runs(ds)
    ds[var_names_with_run_id] = ds[var_names_with_run_id].expand_dims(dim='model_name').assign_coords({'model_name': [ModelName.PLACEHOLDER.name]})
    ds['model_plot_name'] = xr.DataArray([model_plot_name], coords=dict(model_name=ds.coords['model_name']))


def select_only_the_first_default_run_of_every_model_and_dataset(cfg, ds: xr.Dataset) -> xr.Dataset:
    
    var_names_with_run_id = get_var_names_depending_on_runs(ds)
    ds[var_names_with_run_id] = ds[var_names_with_run_id].where(ds['search_type'] == SearchType.DEFAULT.name, drop=True)
    ds[var_names_with_run_id] = ds[var_names_with_run_id].where(ds['seed'] == cfg.seed, drop=True) # when using multiple default runs, the seed changes
    ds = ds.isel(run_id=0)

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


def apply_mean_over_cv_split(ds: xr.Dataset) -> xr.Dataset:
    """
    Apply mean over the cv_split dimension.
    """

    ds = ds.copy()
    metric_vars = [var for var in ds.data_vars if 'cv_split' in ds[var].dims]
    ds[metric_vars] = ds[metric_vars].sum(dim='cv_split') / ds['cv_splits_actual']

    return ds


