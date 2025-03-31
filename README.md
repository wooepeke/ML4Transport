# ML4Transport


## Introduction

Welcome
## Authors

[X. Reparon](https://github.com/wooepeke), [Q. Luo](https://github.com/FelixxLuo), [K. Scholer](https://github.com/Kajscholer), [T. Kortekaas](https://github.com/Thijsjk), [Y. Zhu](https://github.com/Technic1005)

## Progress on 2025-03-31
- The interface of the model has been written and successfully tested on the processed TaxiNYC data. 
- However, this processed TaxiNYC data can not be used directly in the project because we have to perform data preprocessing on our own.

## How to run this model
- create anaconda environment by `environment.yml`
- run `experimentTaxiNYC.py`

## Required data format after preprocessing
- The required data format is exactly the same as `d_map.npy` (dropoff map) and `p_map.npy` (pickup map)
- They are arrays in shape of (n, 32, 32), where 
  - `n=days * Time Interval`, time interval is a period of time (e.g., one hour)
  - 32 is the grid size. The New York is divided to grid map.
  - The value is the pickup/dropoff demand accumulated in that time interval
- you can load the data and visualize it by: (The darker the color, the higher the demand) 
```bash
arr2 = np.load('p_map.npy')
print(arr2.shape)

plt.imshow(arr2[0][0], cmap='Reds')
plt.show() 
```

## Things to be done
- Clean the origin GPS data (maybe from July 1st 2014 to June 30th 2016) and prepare it to the format specified above. (with the help of https://github.com/Lab-Work/gpsresilience)
- Write code of ARIMA for comparison
- I could start writing the report now. And I hope the data cleaning and preparation could be done in two days?