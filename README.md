# PIMFNet: A multi-task wind speed forecasting framework integrating physics-informed feature construction and Bayesian-Optimized composite loss function

This repository is the official PyTorch implementation of the paper "PIMFNet: A multi-task wind speed forecasting framework integrating physics-informed feature construction and Bayesian-Optimized composite loss function"

### Data Download
1. Download the ECMWF-TIGGE data from 2021 to 2024 from the website https://apps.ecmwf.int/datasets/data/tigge/levtype=sfc/type=cf/.
2. Download the DEM data for the corresponding study area from https://doi.org/10.5067/ASTER/ASTGTM.003.
3. Download the ERA5 wind speed data from 2021 to 2024 from https://cds.climate.copernicus.eu/datasets/reanalysis-era5-single-levels?tab=overview.

### Data Preprocessing
1. After data download, use the `data_alignment.py` file in the `Data preprocessing` folder for initial data preprocessing.
2. Then, use the `data_relevance_visualization_CatBoost.py` file for feature selection.

