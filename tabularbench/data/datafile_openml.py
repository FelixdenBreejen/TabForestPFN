from pathlib import Path
import xarray as xr
from tabularbench.core.enums import DatasetSize

from tabularbench.utils.paths_and_filenames import PATH_TO_OPENML_DATASETS


class OpenmlDatafile():

    def __init__(self, openml_dataset_id: int, dataset_size: DatasetSize):

        self.openml_dataset_id = openml_dataset_id
        self.dataset_size = dataset_size

        self.data_path = Path(f"{PATH_TO_OPENML_DATASETS}/{openml_dataset_id}_{dataset_size.name}.nc")
        self.data_path.parent.mkdir(parents=True, exist_ok=True)

        self.get_dataset_from_disk()


    def get_dataset_from_disk(self):
            
        self.ds = xr.open_dataset(self.data_path)
        self.x = self.ds['x'].values
        self.y = self.ds['y'].values
        self.categorical_indicator = self.ds['categorical_indicator'].values
        self.attribute_names = self.ds['attribute_names'].values





