import numpy as np
import matplotlib.pyplot as plt 
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error
from tqdm import tqdm
from statsmodels.tsa.stattools import adfuller


np.random.seed(42)

class SArima():
    def __init__(self, p=1, d=1, q=1, P=0, D=0, Q=0, m=12, ar_params=None, ma_params=None, sar_params=None, sma_params=None):
        # Non-seasonal components
        self.p = p  # Non-seasonal AR order
        self.d = d  # Non-seasonal differencing
        self.q = q  # Non-seasonal MA order
        
        # Seasonal components
        self.P = P  # Seasonal AR order
        self.D = D  # Seasonal differencing
        self.Q = Q  # Seasonal MA order
        self.m = m  # Seasonality period
        
        # Default parameter values if not provided
        if ar_params is None:
            self.ar_params = [0.7/p] * p if p > 0 else []
        else:
            self.ar_params = ar_params[:p]
            
        if ma_params is None:
            self.ma_params = [0.5/q] * q if q > 0 else []
        else:
            self.ma_params = ma_params[:q]
        
        # Seasonal parameters
        if sar_params is None:
            self.sar_params = [0.5/P] * P if P > 0 else []
        else:
            self.sar_params = sar_params[:P]
            
        if sma_params is None:
            self.sma_params = [0.3/Q] * Q if Q > 0 else []
        else:
            self.sma_params = sma_params[:Q]
            
        self.residuals = [0] * max(q, Q*m, 1)  # Initialize residuals with enough history
        self.predictions = []
        self.ts = None
        self.diff_ts = None
        self.forecasted_full = None
        self.MAEs = []
        self.RMSEs = []
        self.y_trues = []
        self.y_preds = []

    def check_stationarity(self, ts):
        result = adfuller(ts)
        # print("ADF Statistic:", result[0])
        # print("p-value:", result[1])
        
        if result[1] < 0.05:
            print("The series is likely stationary.")
        else:
            print("The series is likely non-stationary.")

    def run_model(self, data):
        """
        Fit SARIMA model to the data
        """
        self.ts = pd.Series(data)
        
        # Apply non-seasonal differencing d times
        diff = self.ts.copy()
        for _ in range(self.d):
            diff = diff.diff().dropna()
        
        # Apply seasonal differencing D times
        for _ in range(self.D):
            if len(diff) <= self.m:
                print("Warning: Series too short for seasonal differencing")
                break
            diff = diff.diff(self.m).dropna()
        
        self.diff_ts = diff
        mu = diff.mean()

        # Clear old predictions & residuals
        self.predictions = []
        # Reset residuals with enough history for both regular and seasonal components
        self.residuals = [0] * max(self.q, self.Q*self.m, 1)

        # Determine starting point based on AR and MA orders
        min_start_idx = max(
            self.p,               # Regular AR
            self.q,               # Regular MA
            self.P * self.m,      # Seasonal AR
            self.Q * self.m       # Seasonal MA
        )
        
        # Loop through time series
        for t in range(min_start_idx, len(diff)):
            # Calculate non-seasonal AR component
            ar_component = 0
            for i in range(self.p):
                if t - i - 1 >= 0:
                    ar_component += self.ar_params[i] * diff.iloc[t - i - 1]
            
            # Calculate seasonal AR component
            sar_component = 0
            for i in range(self.P):
                seasonal_idx = t - (i + 1) * self.m
                if seasonal_idx >= 0:
                    sar_component += self.sar_params[i] * diff.iloc[seasonal_idx]
            
            # Calculate non-seasonal MA component
            ma_component = 0
            for j in range(self.q):
                if len(self.residuals) > j:
                    ma_component += self.ma_params[j] * self.residuals[-(j+1)]
            
            # Calculate seasonal MA component
            sma_component = 0
            for j in range(self.Q):
                seasonal_idx = (j + 1) * self.m
                if len(self.residuals) > seasonal_idx:
                    sma_component += self.sma_params[j] * self.residuals[-seasonal_idx]
            
            # Forecast
            y_hat = mu + ar_component + ma_component + sar_component + sma_component
            
            # Get actual value and error
            actual = diff.iloc[t]
            error = actual - y_hat
            
            # Save prediction and residual
            self.predictions.append(y_hat)
            self.residuals.append(error)
            
            # Keep residuals list at appropriate length
            max_history_needed = max(self.q * 2, self.Q * self.m * 2, 10)
            if len(self.residuals) > max_history_needed:
                self.residuals = self.residuals[-max_history_needed:]

        # Reconstruct the forecasted full time series (reverse the differencing)
        forecasted_diff = pd.Series(self.predictions, index=diff.index[min_start_idx:])
        
        # Undo the seasonal differencing first
        self.forecasted_full = forecasted_diff.copy()
        
        # Undo seasonal differencing (if any)
        for _ in range(self.D):
            reconstructed = pd.Series(index=self.ts.index[min_start_idx-self.m:], dtype=float)
            
            # Fill with original values for the first m periods
            reconstructed.iloc[:self.m] = self.ts.iloc[min_start_idx-self.m:min_start_idx]
            
            # Calculate cumulative sums
            for i in range(self.m, len(reconstructed)):
                if i - self.m >= 0 and i < len(reconstructed):
                    seasonal_prev = reconstructed.iloc[i - self.m]
                    if i - min_start_idx >= 0 and i - min_start_idx < len(self.forecasted_full):
                        diff_value = self.forecasted_full.iloc[i - min_start_idx]
                        reconstructed.iloc[i] = seasonal_prev + diff_value
            
            # Keep only the relevant part for the next iteration or final result
            self.forecasted_full = reconstructed.iloc[self.m:]
        
        # Undo regular differencing
        for _ in range(self.d):
            reconstructed = pd.Series(index=self.ts.index[:len(self.forecasted_full) + 1], dtype=float)
            reconstructed.iloc[0] = self.ts.iloc[min_start_idx - self.D * self.m - 1] if min_start_idx - self.D * self.m - 1 >= 0 else self.ts.iloc[0]
            
            # Calculate cumulative sums
            for i in range(1, len(reconstructed)):
                if i-1 < len(self.forecasted_full):
                    reconstructed.iloc[i] = reconstructed.iloc[i-1] + self.forecasted_full.iloc[i-1]
            
            self.forecasted_full = reconstructed.iloc[1:]

        # Align indices
        valid_indices = self.ts.index[min_start_idx:min_start_idx + len(self.forecasted_full)]
        if len(valid_indices) < len(self.forecasted_full):
            self.forecasted_full = self.forecasted_full.iloc[:len(valid_indices)]
        self.forecasted_full.index = valid_indices
        
        _, mae, rmse = self.compute_metrics()
        self.MAEs.append(mae)
        self.RMSEs.append(rmse)

        return self.forecasted_full


    def compute_metrics(self):
        if self.forecasted_full is None or self.ts is None:
            raise ValueError("Model must be run before computing metrics.")

        # Determine starting point based on all model components
        min_start_idx = max(
            self.p,               # Regular AR
            self.q,               # Regular MA
            self.P * self.m,      # Seasonal AR
            self.Q * self.m,      # Seasonal MA
            self.d,               # Regular differencing
            self.D * self.m       # Seasonal differencing
        )

        # Align real values to forecasted
        y_true = self.ts.iloc[min_start_idx:min_start_idx + len(self.forecasted_full)]
        y_pred = self.forecasted_full

        # Handle possible index mismatch
        if len(y_true) > len(y_pred):
            y_true = y_true.iloc[:len(y_pred)]
        elif len(y_pred) > len(y_true):
            y_pred = y_pred.iloc[:len(y_true)]

        self.y_trues.append(y_true.values)
        self.y_preds.append(y_pred.values)

        mae = mean_absolute_error(y_true, y_pred)
        mse = mean_squared_error(y_true, y_pred)
        rmse = np.sqrt(mse)

        return {'MAE': mae, 'MSE': mse, 'RMSE': rmse}, mae, rmse

    def compute_full_metrics(self):
        all_y_true = np.concatenate(self.y_trues)
        all_y_pred = np.concatenate(self.y_preds)

        total_mae = mean_absolute_error(all_y_true, all_y_pred)
        total_rmse = np.sqrt(mean_squared_error(all_y_true, all_y_pred))

        return total_mae, total_rmse
                
    def forecast(self, steps=1):
        if self.diff_ts is None:
            raise ValueError("Model must be run before forecasting.")
        
        # Determine the minimum starting index
        min_start_idx = max(
            self.p,               # Regular AR
            self.q,               # Regular MA
            self.P * self.m,      # Seasonal AR
            self.Q * self.m,      # Seasonal MA
            self.d,               # Regular differencing
            self.D * self.m       # Seasonal differencing
        )
        
        # Get the differenced series values
        diff_series = self.diff_ts.values
        mu = self.diff_ts.mean()
        
        # Initialize arrays to store recent values for both regular and seasonal components
        last_diff_values = diff_series[-self.p:] if self.p > 0 else []
        last_seasonal_values = []
        for i in range(self.P):
            idx = -(i+1)*self.m
            if abs(idx) < len(diff_series):
                last_seasonal_values.append(diff_series[idx])
            else:
                last_seasonal_values.append(0)  # Pad with zeros if not enough history
        
        # Get recent residuals for both regular and seasonal MA terms
        last_residuals = self.residuals[-self.q:] if self.q > 0 else []
        last_seasonal_residuals = []
        for i in range(self.Q):
            idx = -(i+1)*self.m
            if abs(idx) < len(self.residuals):
                last_seasonal_residuals.append(self.residuals[idx])
            else:
                last_seasonal_residuals.append(0)  # Pad with zeros if not enough history
        
        # Generate forecasts for differenced series
        forecasted_diff_values = []
        
        for step in range(steps):
            # AR component (non-seasonal)
            ar_component = 0
            for i in range(self.p):
                if i < len(last_diff_values):
                    ar_component += self.ar_params[i] * last_diff_values[-(i+1)]
            
            # Seasonal AR component
            sar_component = 0
            for i in range(self.P):
                if i < len(last_seasonal_values):
                    sar_component += self.sar_params[i] * last_seasonal_values[i]
            
            # MA component (non-seasonal)
            ma_component = 0
            for j in range(self.q):
                if j < len(last_residuals):
                    ma_component += self.ma_params[j] * last_residuals[-(j+1)]
            
            # Seasonal MA component
            sma_component = 0
            for j in range(self.Q):
                if j < len(last_seasonal_residuals):
                    sma_component += self.sma_params[j] * last_seasonal_residuals[j]
            
            # Generate forecast for this step
            forecast_diff = mu + ar_component + ma_component + sar_component + sma_component
            forecasted_diff_values.append(forecast_diff)
            
            # Update values for next iteration
            if self.p > 0:
                last_diff_values = np.append([forecast_diff], last_diff_values[:-1] if len(last_diff_values) > 1 else [])
            
            # Update seasonal values if we have enough forecasts
            if step % self.m == 0 and step >= self.m and self.P > 0:
                for i in range(self.P):
                    if i < len(last_seasonal_values) and i < len(forecasted_diff_values):
                        last_seasonal_values[i] = forecasted_diff_values[-self.m]
            
            # For MA terms, we can't know future errors, so we gradually reduce their impact
            print(f"Last residual: {last_residuals}")
            if self.q > 0:
                dampened_residual = last_residuals[-1] * 0.5 if len(last_residuals) > 0 else 0
                last_residuals = np.append([dampened_residual], last_residuals[:-1] if len(last_residuals) > 1 else [])
            
            # Same for seasonal MA terms
            if step % self.m == 0 and step >= self.m and self.Q > 0:
                for i in range(self.Q):
                    if i < len(last_seasonal_residuals):
                        last_seasonal_residuals[i] *= 0.5  # Dampen seasonal residuals too
        
        # Create a pandas Series from the forecasted differenced values
        forecasted_diff = pd.Series(forecasted_diff_values)
        
        # If no differencing was done, return the forecasted values directly
        if self.d == 0 and self.D == 0:
            forecasted_diff.index = range(len(self.ts), len(self.ts) + steps)
            return forecasted_diff
        
        # Handle both regular and seasonal un-differencing
        result = forecasted_diff.copy()
        
        # Undo seasonal differencing first (if any)
        for _ in range(self.D):
            seasonal_values = []
            seasonal_idx = -self.m
            
            # Get the last known values before forecasting
            last_known_values = self.ts.iloc[-self.m:].values if len(self.ts) >= self.m else self.ts.iloc[-1:].values * self.m
            
            # Initialize with the last known values
            for i in range(steps):
                if i < self.m:
                    # For the first period, use known values plus forecast differences
                    if i < len(last_known_values) and i < len(result):
                        seasonal_values.append(last_known_values[i] + result.iloc[i])
                    else:
                        # Not enough history, just use the forecast difference
                        seasonal_values.append(result.iloc[i] if i < len(result) else 0)
                else:
                    # For subsequent periods, use previous seasonal value plus forecast difference
                    prev_seas_idx = i - self.m
                    if prev_seas_idx < len(seasonal_values) and i < len(result):
                        seasonal_values.append(seasonal_values[prev_seas_idx] + result.iloc[i])
                    else:
                        # Not enough data, just use the last value
                        seasonal_values.append(seasonal_values[-1] if seasonal_values else 0)
            
            result = pd.Series(seasonal_values)
        
        # Undo regular differencing
        for _ in range(self.d):
            # Start with the last known value before the forecast period
            last_value = self.ts.iloc[-1] if len(self.ts) > 0 else 0
            
            # Apply integration (cumulative sum)
            integrated_values = [last_value]
            for i in range(len(result)):
                integrated_values.append(integrated_values[-1] + result.iloc[i])
            
            # Remove the initial value used for integration
            result = pd.Series(integrated_values[1:])
        
        # Set the proper index for the forecast
        result.index = range(len(self.ts), len(self.ts) + steps)
        return result

    def evaluate_test_data(self, test_data):
        if self.forecasted_full is None:
            raise ValueError("Run the model before evaluating test data.")
        
        # Generate forecasts for the test period
        test_length = len(test_data)
        forecast = self.forecast(test_length)
        
        # Compute metrics
        mae = mean_absolute_error(test_data, forecast)
        mse = mean_squared_error(test_data, forecast)
        rmse = np.sqrt(mse)
        
        return {'MAE': mae, 'MSE': mse, 'RMSE': rmse}