# TabSGFD

Welcome to the TabSGFD repository!

Setup a new environment using python 3.11, then install the requirements and the package:

```
pip install -r requirements.txt

pip install -e .
```

## Benchmark Results

First make a dictory named `data/`

For Whytrees, the benchmark results are available with:

curl -L -o data/benchmark_total.csv https://figshare.com/ndownloader/files/40081681

For Tabzilla, the benchmark results are available with:

gdown --id 1tsnWuvwH77aDEsx5rf0Pg_Kv8WhMtsRQ -O data/metadataset_clean.csv

Or check this google drive: https://drive.google.com/drive/folders/1cHisTmruPHDCYVOYnaqvTdybLngMkB8R

If any of these links don't work, please check the github of the original authors of the benchmark paper.

## Data

All data can be downloaded and preprocessed by running:

```
python preprocess.py
```

## Running

There are two main running files:

- `main.py` runs one or multiple benchmarks for a given model defined in `config/main.py`
- `pretrain.py` runs the pretraining for a given model defined in `config/pretrain.py`    
