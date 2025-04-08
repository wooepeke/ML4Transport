import numpy as np
import matplotlib.pyplot as plt 
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error
from tqdm import tqdm

np.random.seed(42)


class Arima():
    def __init__(self, data, alpha=0.7, theta=0.5):
        self.data = data
        self.alpha = alpha
        self.theta = theta
        self.residuals = [0]
        self.predictions = []
        self.ts = None
        self.forecasted_full = None

    def run_model(self):
        self.ts = pd.Series(self.data)
        diff = self.ts.diff().dropna()  # Differencing the data (to make it stationary)
        mu = diff.mean()

        # Clear old predictions & residuals
        self.predictions = []
        self.residuals = [0]

        for t in range(1, len(diff)):
            y_t_minus_1 = diff.iloc[t-1]
            e_t_minus_1 = self.residuals[-1]
            y_hat = np.max((0, mu + self.alpha * y_t_minus_1 + self.theta * e_t_minus_1))
            actual = diff.iloc[t]
            error = actual - y_hat

            self.predictions.append(y_hat)
            self.residuals.append(error)

        # Reconstruct the forecasted full time series from the differenced predictions
        forecasted = pd.Series(self.predictions, index=diff.index[1:])
        forecasted_cumsum = forecasted.cumsum()
        self.forecasted_full = self.ts.iloc[0] + forecasted_cumsum

        return self.forecasted_full

    def compute_metrics(self):
        if self.forecasted_full is None or self.ts is None:
            raise ValueError("Model must be run before computing metrics.")

        # Align real values to forecasted (skip first two entries due to differencing)
        y_true = self.ts.iloc[2:]
        y_pred = self.forecasted_full

        mae = mean_absolute_error(y_true, y_pred)
        mse = mean_squared_error(y_true, y_pred)
        rmse = np.sqrt(mse)

        return {'MAE': mae, 'MSE': mse, 'RMSE': rmse}

    def grid_search(self, alpha_range, theta_range, metric='MAE'):
        best_score = float('inf')
        best_params = (None, None)

        for alpha in alpha_range:
            for theta in theta_range:
                self.alpha = alpha
                self.theta = theta
                self.residuals = [0]
                self.predictions = []
                self.run_model()

                scores = self.compute_metrics()
                score = scores[metric]

                if score < best_score:
                    best_score = score
                    best_params = (alpha, theta)

        return best_params, best_score
    
    def forecast_test_period(self, test_length):
        """
        Forecast values for the test period starting from the end of training data
        """
        if self.forecasted_full is None:
            raise ValueError("Run the model before forecasting test period.")

        diff = self.ts.diff().dropna()
        mu = diff.mean()

        # Start from the last value in training data
        last_value = self.ts.iloc[-1]
        y_prev = diff.iloc[-1]
        e_prev = self.residuals[-1]

        test_preds = []

        for _ in range(test_length):
            y_hat = mu + self.alpha * y_prev + self.theta * e_prev
            y_prev = y_hat
            e_prev = 0  # assume zero error in future
            last_value += y_hat
            test_preds.append(last_value)

        # Create indices starting from the end of training data
        start_idx = len(self.ts)
        return pd.Series(test_preds, index=range(start_idx, start_idx + test_length))

