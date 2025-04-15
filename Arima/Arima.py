import numpy as np
import matplotlib.pyplot as plt 
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error
from tqdm import tqdm
from statsmodels.tsa.stattools import adfuller


np.random.seed(42)

class Arima():
    def __init__(self, p=1, d=1, q=1, ar_params=None, ma_params=None):
        self.p = p
        self.d = d
        self.q = q
        
        # Default parameter values if not provided
        if ar_params is None:
            self.ar_params = [0.7/p] * p if p > 0 else []
        else:
            self.ar_params = ar_params[:p]
            
        if ma_params is None:
            self.ma_params = [0.5/q] * q if q > 0 else []
        else:
            self.ma_params = ma_params[:q]
            
        self.residuals = [0] * max(q, 1)  # Initialize residuals
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
        Fit ARIMA model to the data
        """
        self.ts = pd.Series(data)
        self.check_stationarity(self.ts)
        # Apply differencing d times
        diff = self.ts.copy()
        for _ in range(self.d):
            diff = diff.diff().dropna()
        
        self.diff_ts = diff
        mu = diff.mean()

        # Clear old predictions & residuals
        self.predictions = []
        self.residuals = [0] * max(self.q, 1)  # Reset residuals

        # Loop through time series
        for t in range(max(self.p, self.q), len(diff)):
            # Calculate AR component (sum of AR parameters * past values)
            ar_component = 0
            for i in range(self.p):
                if t - i - 1 >= 0:
                    ar_component += self.ar_params[i] * diff.iloc[t - i - 1]
            
            # Calculate MA component (sum of MA parameters * past residuals)
            ma_component = 0
            for j in range(self.q):
                if len(self.residuals) > j:
                    ma_component += self.ma_params[j] * self.residuals[-(j+1)]
            
            # Forecast
            y_hat = mu + ar_component + ma_component
            
            # Get actual value and error
            actual = diff.iloc[t]
            error = actual - y_hat
            
            # Save prediction and residual
            self.predictions.append(y_hat)
            self.residuals.append(error)
            
            # Keep residuals list at appropriate length
            if len(self.residuals) > max(self.q * 2, 10):
                self.residuals = self.residuals[-self.q * 2:]

        # Reconstruct the forecasted full time series (reverse the differencing)
        forecasted_diff = pd.Series(self.predictions, index=diff.index[max(self.p, self.q):])
        
        # Undo the differencing
        self.forecasted_full = forecasted_diff.copy()
        for _ in range(self.d):
            # Create a cumulative sum and add the original first value(s)
            self.forecasted_full = self.forecasted_full.cumsum()
            
            # Add back the original value before differencing
            if _ == self.d - 1:  # Last iteration
                orig_values = self.ts.iloc[:self.d]
                for i, val in enumerate(orig_values):
                    if i == 0:
                        self.forecasted_full = val + self.forecasted_full
                    else:
                        # Handle multiple levels of differencing
                        first_values = self.ts.iloc[:i+1].values
                        self.forecasted_full.iloc[0] += first_values[i] - first_values[i-1]

        _, mae, rmse = self.compute_metrics()
        self.MAEs.append(mae)
        self.RMSEs.append(rmse)

        return self.forecasted_full


    def compute_metrics(self):
        if self.forecasted_full is None or self.ts is None:
            raise ValueError("Model must be run before computing metrics.")

        # Align real values to forecasted (skip initial entries due to differencing and lags)
        start_idx = max(self.p, self.q) + self.d
        y_true = self.ts.iloc[start_idx:]
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
                
    def forecast_test_period(self, steps=1):
        if self.diff_ts is None:
            raise ValueError("Model must be run before forecasting.")

        last_differenced = self.diff_ts.copy()
        mu = last_differenced.mean()

        future_preds = []
        residuals = self.residuals[-self.q:]  # Use last residuals

        # Forecast the future steps
        for _ in range(steps):
            # AR component
            ar_component = 0
            for i in range(self.p):
                idx = -i-1
                if abs(idx) <= len(last_differenced):
                    ar_component += self.ar_params[i] * last_differenced.iloc[idx]

            # MA component (apply residuals from the last prediction)
            ma_component = 0
            for j in range(self.q):
                if j < len(residuals):
                    ma_component += self.ma_params[j] * residuals[-j-1]

            # Forecast
            y_hat = mu + ar_component + ma_component
            future_preds.append(y_hat)

            # Update residuals for future steps
            residuals.append(0)  # We don’t know true residuals in future
            if len(residuals) > self.q * 2:
                residuals = residuals[-self.q * 2:]

            # Update differenced time series with forecasted value
            last_differenced = pd.concat([last_differenced, pd.Series([y_hat], index=[last_differenced.index[-1] + 1])])

        # Undo differencing to bring the forecast back to the original scale
        forecast = pd.Series(future_preds)

        # Add back the previous original values (before differencing)
        last_values = self.ts.copy()
        for _ in range(self.d):
            last_val = last_values.iloc[-1]
            forecast = forecast.cumsum() + last_val
            last_values = pd.concat([last_values, forecast])

        forecast.index = range(len(self.ts), len(self.ts) + steps)
        return forecast


    def evaluate_test_data(self, test_data):
        if self.forecasted_full is None:
            raise ValueError("Run the model before evaluating test data.")
        
        # Generate forecasts for the test period
        test_length = len(test_data)
        forecast = self.forecast_test_period(test_length)
        
        # Compute metrics
        mae = mean_absolute_error(test_data, forecast)
        mse = mean_squared_error(test_data, forecast)
        rmse = np.sqrt(mse)
        
        return {'MAE': mae, 'MSE': mse, 'RMSE': rmse}