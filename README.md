# ML4Transport


## Introduction

Welcome
## Authors

[X. Reparon](https://github.com/wooepeke), [Q. Luo](https://github.com/FelixxLuo), [K. Scholer](https://github.com/Kajscholer), [T. Kortekaas](https://github.com/Thijsjk), [Y. Zhu](https://github.com/Technic1005)

## Progress on 2025-03-31
- The interface of the model has been written and successfully tested on the processed TaxiNYC data. 
- However, this processed TaxiNYC data can not be used directly in the project because we have to perform data preprocessing on our own.

## Progress on 2025-04-06
- Data cleaning and preparation are finished. Now the data can be transformed to required format from `*.parquet` files via `data\data_preparation.py`
  - To do so, you should put `yellow_tripdata_2010-01.parquet` until `yellow_tripdata_2010-05.parquet` in the `data` folder.
ML4Transport   
├── data
│   ├── data_preparation.py
│   ├── yellow_tripdata_2010-01.parquet  
│   ├── ......
│   ├── ......
│   ├── yellow_tripdata_2010-01.parquet
- The main function (in `experimentTaxiNYC.py`) is modified for future experiments.

## How to run this model
- Create an anaconda environment by `environment.yml`
- run `experimentTaxiNYC.py`


## Things to be done
- Write the code of ARIMA for comparison.
- Change hyperparameters to see the influence of trend, period, closeness and number of residual units.
- I will start writing the report 
