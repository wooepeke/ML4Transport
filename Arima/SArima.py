import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error
from statsmodels.tsa.stattools import adfuller

class Sarima:
    def __init__(self, p=1, d=1, q=1, P=1, D=1, Q=1, s=24,
                 ar_params=None, ma_params=None,
                 sar_params=None, sma_params=None):
        # Non-seasonal
        self.p = p
        self.d = d
        self.q = q
        self.ar_params = ar_params if ar_params is not None else [0.5] * p
        self.ma_params = ma_params if ma_params is not None else [0.5] * q

        # Seasonal
        self.P = P
        self.D = D
        self.Q = Q
        self.s = s
        self.sar_params = sar_params if sar_params is not None else [0.5] * P
        self.sma_params = sma_params if sma_params is not None else [0.5] * Q

        self.ts = None
        self.diff_ts = None
        self.residuals = []
        self.fitted_values = None
        self.forecast_values = None

    def difference(self, series, d, seasonal_d=0, seasonal_period=1):
        for _ in range(d):
            series = series.diff().dropna()
        for _ in range(seasonal_d):
            series = series.diff(seasonal_period).dropna()
        return series

    def inverse_difference(self, diff_series, original_series, d, D, s):
        result = diff_series.copy()
        for i in range(D):
            result = result.add(original_series.shift((i + 1) * s).dropna(), fill_value=0)
        for i in range(d):
            result = result.add(original_series.shift(i + 1).dropna(), fill_value=0)
        return result

    def fit(self, series):
        self.ts = pd.Series(series)
        self.diff_ts = self.difference(self.ts, self.d, self.D, self.s)

        n = len(self.diff_ts)
        fitted = []
        self.residuals = []

        for t in range(max(self.p, self.q, self.P * self.s, self.Q * self.s), n):
            y_t = self.diff_ts.iloc[t]

            # AR + SAR terms
            ar_term = sum(
                self.ar_params[i] * self.diff_ts.iloc[t - i - 1]
                for i in range(self.p)
            ) + sum(
                self.sar_params[j] * self.diff_ts.iloc[t - (j + 1) * self.s]
                for j in range(self.P)
            )

            # MA + SMA terms
            ma_term = sum(
                self.ma_params[j] * self.residuals[-j - 1]
                for j in range(min(self.q, len(self.residuals)))
            ) + sum(
                self.sma_params[k] * self.residuals[-(k + 1) * self.s]
                for k in range(min(self.Q, len(self.residuals) // self.s))
                if (k + 1) * self.s <= len(self.residuals)
            )

            y_hat = ar_term + ma_term
            error = y_t - y_hat

            fitted.append(y_hat)
            self.residuals.append(error)

        self.fitted_values = pd.Series(fitted, index=self.diff_ts.index[max(self.p, self.q, self.P * self.s, self.Q * self.s):])
        return self.inverse_difference(self.fitted_values, self.ts, self.d, self.D, self.s)

    def forecast(self, steps=1):
        if self.fitted_values is None:
            raise RuntimeError("Model must be fitted before forecasting.")

        forecast = []
        last_values = list(self.diff_ts.iloc[-max(self.p, self.P * self.s):])
        last_residuals = self.residuals[-max(self.q, self.Q * self.s):]

        for step in range(steps):
            ar_term = sum(
                self.ar_params[i] * last_values[-i - 1]
                for i in range(self.p)
            ) + sum(
                self.sar_params[j] * last_values[-(j + 1) * self.s]
                for j in range(self.P)
                if (j + 1) * self.s <= len(last_values)
            )

            ma_term = sum(
                self.ma_params[j] * (last_residuals[-j - 1] if j < len(last_residuals) else 0)
                for j in range(self.q)
            ) + sum(
                self.sma_params[k] * (last_residuals[-(k + 1) * self.s] if (k + 1) * self.s <= len(last_residuals) else 0)
                for k in range(self.Q)
            )

            y_hat = ar_term + ma_term
            forecast.append(y_hat)

            last_values.append(y_hat)
            if len(last_values) > max(self.p, self.P * self.s):
                last_values.pop(0)

            last_residuals.append(0)
            if len(last_residuals) > max(self.q, self.Q * self.s):
                last_residuals.pop(0)

        forecast = pd.Series(forecast)
        last_value = self.ts.iloc[-1]
        for _ in range(self.D):
            forecast = forecast.cumsum() + self.ts.iloc[-self.s]
        for _ in range(self.d):
            forecast = forecast.cumsum() + last_value

        forecast.index = range(len(self.ts), len(self.ts) + steps)
        self.forecast_values = forecast
        return forecast

    def evaluate(self, predictions, test_data):
        if self.fitted_values is None:
            raise ValueError("Model not fitted yet.")

        actual = test_data
        pred = predictions

        mae = mean_absolute_error(actual, pred)
        rmse = np.sqrt(np.mean((predictions-actual)**2))

        return {'MAE': mae, 'RMSE': rmse}

    def check_stationarity(self):
        if self.ts is None:
            raise ValueError("No time series loaded.")
        result = adfuller(self.ts.dropna())
        print(f"ADF Statistic: {result[0]:.4f}")
        print(f"p-value: {result[1]:.4f}")
        if result[1] < 0.05:
            print("Series is likely stationary.")
        else:
            print("Series is likely non-stationary.")
