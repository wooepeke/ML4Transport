import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error
from statsmodels.tsa.stattools import adfuller

class Arima:
    def __init__(self, p=1, d=1, q=1, ar_params=None, ma_params=None):
        self.p = p
        self.d = d
        self.q = q
        self.ar_params = ar_params if ar_params is not None else [0.5] * p
        self.ma_params = ma_params if ma_params is not None else [0.5] * q

        self.ts = None
        self.diff_ts = None
        self.residuals = []
        self.fitted_values = None
        self.forecast_values = None

    def difference(self, series, d):
        for _ in range(d):
            series = series.diff().dropna()
        return series

    def inverse_difference(self, diff_series, original_series, d):
        result = diff_series.copy()
        for i in range(d):
            last = original_series.iloc[:-(d - i)].values
            result = last + result.cumsum()
        return result

    def fit(self, series):
        self.ts = pd.Series(series)
        self.diff_ts = self.difference(self.ts, self.d)
        print(len(self.diff_ts))
        n = len(self.diff_ts)
        fitted = []
        self.residuals = []

        for t in range(max(self.p, self.q), n):
            y_t = self.diff_ts.iloc[t]

            ar_term = sum(
                self.ar_params[i] * self.diff_ts.iloc[t - i - 1]
                for i in range(self.p)
            )

            ma_term = sum(
                self.ma_params[j] * self.residuals[-j - 1]
                for j in range(min(self.q, len(self.residuals)))
            )

            y_hat = ar_term + ma_term
            error = y_t - y_hat

            fitted.append(y_hat)
            self.residuals.append(error)

        self.fitted_values = pd.Series(fitted, index=self.diff_ts.index[max(self.p, self.q):])
        return self.inverse_difference(self.fitted_values, self.ts, self.d)

    def forecast(self, steps=1):
        if self.fitted_values is None:
            raise RuntimeError("Model must be fitted before forecasting.")

        forecast = []
        last_values = list(self.diff_ts.iloc[-self.p:])
        last_residuals = self.residuals[-self.q:]

        for _ in range(steps):
            ar_term = sum(
                self.ar_params[i] * last_values[-i - 1]
                for i in range(self.p)
            )

            ma_term = sum(
                self.ma_params[j] * (last_residuals[-j - 1] if j < len(last_residuals) else 0)
                for j in range(self.q)
            )

            y_hat = ar_term + ma_term
            forecast.append(y_hat)

            last_values.append(y_hat)
            if len(last_values) > self.p:
                last_values.pop(0)
            last_residuals.append(0)
            if len(last_residuals) > self.q:
                last_residuals.pop(0)

        forecast = pd.Series(forecast)
        last_value = self.ts.iloc[-1]
        for _ in range(self.d):
            forecast = forecast.cumsum() + last_value

        forecast.index = range(len(self.ts), len(self.ts) + steps)
        self.forecast_values = forecast
        return forecast

    def evaluate(self):
        if self.fitted_values is None:
            raise ValueError("Model not fitted yet.")

        actual = self.diff_ts.iloc[max(self.p, self.q):]
        pred = self.fitted_values

        mae = mean_absolute_error(actual, pred)
        rmse = np.sqrt(mean_squared_error(actual, pred))

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
