# Implementation of ST-ResNet for NYCTaxi Dataset

## Authors

[X. Reparon](https://github.com/wooepeke), [Q. Luo](https://github.com/FelixxLuo), [K. Scholer](https://github.com/Kajscholer), [T. Kortekaas](https://github.com/Thijsjk), [Y. Zhu](https://github.com/Technic1005)

## Data Cleaning and Preparation
- Filename: `data\data_preparation.py`
- Function: Transform raw data (`*.parquet`) to heat maps
- Output: `pickup_counts.npy` and `dropoff_counts.npy` that are heat maps of pickup and dropoff demands
- To do so, you should put `yellow_tripdata_2010-01.parquet` until `yellow_tripdata_2010-05.parquet` in the `data` folder.
├── data
│   ├── data_preparation.py
│   ├── yellow_tripdata_2010-01.parquet  
│   ├── ......
│   ├── ......
│   ├── yellow_tripdata_2010-01.parquet

## Run the Model
- Filename: `experimentTaxiNYC.py`
- Function: Main function
- Premise: Create an Anaconda environment by `environment.yml`
- Output: RMSE and MAE in test set (in console) and the best model

