import numpy as np
import pandas as pd
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_absolute_error, mean_squared_error
from tqdm import tqdm


class ARIMAGridSearch:
    def __init__(self, time_series):
        self.ts = pd.Series(time_series).dropna()
        self.best_model = None
        self.best_order = None
        self.best_score = float('inf')
        self.metric = None

    def evaluate_model(self, order):
        try:
            model = ARIMA(self.ts, order=order)
            model_fit = model.fit()
            predictions = model_fit.predict(start=order[1], end=len(self.ts) - 1)  # skip `d` differenced terms
            actual = self.ts[order[1]:]

            mae = mean_absolute_error(actual, predictions)
            rmse = np.sqrt(mean_squared_error(actual, predictions))
            return mae, rmse, model_fit
        except Exception as e:
            return np.inf, np.inf, None  # return bad scores if model fails

    def grid_search(self, p_values, d_values, q_values, metric='MAE'):
        self.metric = metric
        results = []

        for p in tqdm(p_values, desc="Grid Search"):
            for d in d_values:
                for q in q_values:
                    mae, rmse, model_fit = self.evaluate_model((p, d, q))
                    score = mae if metric == 'MAE' else rmse
                    results.append(((p, d, q), mae, rmse))

                    if score < self.best_score:
                        self.best_score = score
                        self.best_order = (p, d, q)
                        self.best_model = model_fit

        return self.best_order, self.best_score, pd.DataFrame(results, columns=['Order', 'MAE', 'RMSE'])

    def forecast(self, steps=10):
        if self.best_model is None:
            raise ValueError("Run grid_search before forecasting.")
        return self.best_model.forecast(steps=steps)

