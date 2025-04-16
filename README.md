# Implementation of ST-ResNet for NYCTaxi Dataset

## Authors

[X. Reparon](https://github.com/wooepeke), [Q. Luo](https://github.com/FelixxLuo), [K. Scholer](https://github.com/Kajscholer), [T. Kortekaas](https://github.com/Thijsjk), [Y. Zhu](https://github.com/Technic1005)

## Data Cleaning and Preparation
- Filename: `data\data_preparation.py`
- Function: Transform raw data (`*.parquet`) to heat maps
- Output: `pickup_counts.npy` and `dropoff_counts.npy` that are heat maps of pickup and dropoff demands
- To do so, you should put `yellow_tripdata_2010-01.parquet` until `yellow_tripdata_2010-05.parquet` in the `data` folder.
├── data
│   ├── data_preparation.py             # Script for cleaning, transforming, and processing raw data
│   ├── yellow_tripdata_2010-01.parquet # Raw NYC taxi trip data for January 2010
│   ├── yellow_tripdata_2010-02.parquet # Raw NYC taxi trip data for February 2010
│   ├── ...                             # Additional monthly Parquet files
│   ├── yellow_tripdata_2010-12.parquet # Raw NYC taxi trip data for December 2010

## Run the Model
- Filename: `experimentTaxiNYC.py`
- Function: Main function
- Premise: Create an Anaconda environment by `environment.yml`
- Output: RMSE and MAE in test set (in console) and the best model

