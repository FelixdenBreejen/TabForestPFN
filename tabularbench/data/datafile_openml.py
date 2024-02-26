from pathlib import Path
import xarray as xr



class OpenmlDatafile():

    def __init__(self, dataset_file_path: Path):

        self.data_path = dataset_file_path            
        self.ds = xr.open_dataset(self.data_path)
        self.x = self.ds['x'].values
        self.y = self.ds['y'].values
        self.categorical_indicator = self.ds['categorical_indicator'].values
        self.attribute_names = self.ds['attribute_names'].values

        self.indices_train = self.ds['indices_train'].values
        self.indices_val = self.ds['indices_val'].values
        self.indices_test = self.ds['indices_test'].values





