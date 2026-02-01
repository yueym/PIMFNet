# PIMFNet: A multi-task wind speed forecasting framework integrating physics-informed feature construction and Bayesian-Optimized composite loss function

This repository is the official PyTorch implementation of the paper "PIMFNet: A multi-task wind speed forecasting framework integrating physics-informed feature construction and Bayesian-Optimized composite loss function"

### Data Download
1. Download the ECMWF-TIGGE data from 2021 to 2024 from the website https://apps.ecmwf.int/datasets/data/tigge/levtype=sfc/type=cf/.
2. Download the DEM data for the corresponding study area from https://doi.org/10.5067/ASTER/ASTGTM.003.
3. Download the ERA5 wind speed data from 2021 to 2024 from https://cds.climate.copernicus.eu/datasets/reanalysis-era5-single-levels?tab=overview.

### Data Preprocessing
1. After data download, use the `data_alignment.py` file in the `Data preprocessing` folder for initial data preprocessing.
2. Then, use the `data_feature_constructing_last.py` file for physical-informed feature construction.

### SHAP analysis
1.After feature selection, use the `SHAP_importance.py` file in the `SHAP analysis` folder to analyze the importance of all input features and provide interpretability for the proposed model.

### Model Code
1. First, use the `train.py` file in the `Model code` folder for model training.
2. Next, use the `optimization.py` file in the `Model code` folder to optimize the hyperparameters of the proposed model, including learning rate, batch size, and number of hidden layers.
3. Finally, use the `test.py` file in the `Model code` folder to test the wind speed prediction results.



