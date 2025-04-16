import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tqdm import tqdm
from matplotlib.colors import PowerNorm
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_absolute_error, mean_squared_error

np.random.seed(42)


class SpatialARIMA:
    """
    A class that models the entire spatial grid together rather than
    fitting individual models for each cell.
    """

    def __init__(self, spatial_time_series):
        """
        Initialize with a spatial time series.

        Parameters:
        -----------
        spatial_time_series : np.ndarray
            3D array of shape (time_steps, height, width)
        """
        # Reshape the 3D grid (time, height, width) into a 2D matrix (time, height*width)
        self.original_shape = spatial_time_series.shape
        self.n_timesteps = spatial_time_series.shape[0]
        self.height = spatial_time_series.shape[1]
        self.width = spatial_time_series.shape[2]

        # Flatten the spatial dimensions
        self.flattened_ts = spatial_time_series.reshape(self.n_timesteps, -1)
        self.n_cells = self.flattened_ts.shape[1]

        # Initialize storage for models
        self.model = None
        self.order = None
        self.best_score = float('inf')

    def grid_search(self, p_range, d_range, q_range, metric='MAE'):
        """
        Find the best ARIMA parameters for the spatial data.

        Parameters:
        -----------
        p_range, d_range, q_range : range
            Ranges of p, d, q parameters to search
        metric : str
            'MAE' or 'RMSE' to determine which metric to optimize

        Returns:
        --------
        best_order : tuple
            The (p, d, q) combination with the best score
        best_score : float
            The score of the best model
        results : pd.DataFrame
            DataFrame with all results
        """
        best_order = None
        best_score = float('inf')
        results = []

        # We'll use the mean value across all cells for parameter selection
        mean_ts = np.mean(self.flattened_ts, axis=1)

        for p in tqdm(p_range, desc="Grid Search p"):
            for d in d_range:
                for q in q_range:
                    try:
                        # Fit ARIMA model on the mean time series
                        model = ARIMA(mean_ts, order=(p, d, q))
                        model_fit = model.fit()

                        # Evaluate
                        predictions = model_fit.predict(start=d, end=len(mean_ts) - 1)
                        actual = mean_ts[d:]

                        mae = mean_absolute_error(actual, predictions)
                        rmse = np.sqrt(mean_squared_error(actual, predictions))

                        score = mae if metric == 'MAE' else rmse
                        results.append(((p, d, q), mae, rmse))

                        if score < best_score:
                            best_score = score
                            best_order = (p, d, q)
                    except Exception as e:
                        results.append(((p, d, q), np.inf, np.inf))

        self.order = best_order
        self.best_score = best_score

        return best_order, best_score, pd.DataFrame(results, columns=['Order', 'MAE', 'RMSE'])

    def fit(self, order=None):
        """
        Fit ARIMA models to each cell in the grid.

        Parameters:
        -----------
        order : tuple, optional
            The (p, d, q) parameters for ARIMA. If None, uses the best
            order from grid_search()
        """
        if order is None:
            if self.order is None:
                raise ValueError("Either run grid_search first or provide an order")
            order = self.order

        # Store all fitted models
        models = []

        # Fit a model for each cell in the grid
        for i in tqdm(range(self.n_cells), desc="Fitting models"):
            cell_ts = self.flattened_ts[:, i]
            try:
                model = ARIMA(cell_ts, order=order)
                model_fit = model.fit()
                models.append(model_fit)
            except Exception as e:
                # If fitting fails, use a simple model
                fallback_order = (1, 0, 0)
                try:
                    model = ARIMA(cell_ts, order=fallback_order)
                    model_fit = model.fit()
                    models.append(model_fit)
                except Exception as e:
                    # If all fails, use a moving average
                    models.append(None)

        self.models = models
        return self

    def forecast(self, steps=1):
        """
        Forecast future values for each cell.

        Parameters:
        -----------
        steps : int
            Number of steps to forecast

        Returns:
        --------
        forecasts : np.ndarray
            Array of shape (steps, height, width)
        """
        if self.models is None:
            raise ValueError("Run fit() before forecasting")

        forecasts = np.zeros((steps, self.n_cells))

        for i, model_fit in enumerate(self.models):
            if model_fit is not None:
                try:
                    cell_forecast = model_fit.forecast(steps=steps)
                    forecasts[:, i] = np.maximum(cell_forecast, 0)  # Cap at 0
                except Exception as e:
                    # If forecast fails, use last value
                    last_value = self.flattened_ts[-1, i]
                    forecasts[:, i] = np.maximum(last_value, 0)
            else:
                # If model is None, use the last value
                last_value = self.flattened_ts[-1, i]
                forecasts[:, i] = np.maximum(last_value, 0)

        # Reshape back to spatial grid
        return forecasts.reshape(steps, self.height, self.width)


def plot_heatmap(ax, data, title, vmin, vmax, cmap='plasma'):
    norm = PowerNorm(gamma=0.4, vmin=vmin, vmax=vmax)  # gamma < 1 boosts lower values
    cax = ax.imshow(data, cmap=cmap, norm=norm)
    ax.set_title(title)
    plt.colorbar(cax, ax=ax, shrink=0.8)


def main():
    # Load the data
    dropoff_data = np.load(r'data\dropoff_counts.npy')  # (n, 32, 32)
    pickup_data = np.load(r'data\pickup_counts.npy')  # (n, 32, 32)

    # Define the training and testing indices
    forecast_hours = 1  # Maximum forecast period
    train_st = 0
    train_end = test_st = (train_st + 240)
    test_end = test_st + forecast_hours

    # Define parameter ranges for grid search
    p_range = range(3)  # Adjust depending on complexity and performance
    d_range = range(2)
    q_range = range(3)

    # Process dropoff data
    print("Processing Dropoff model...")
    dropoff_train = dropoff_data[train_st:train_end]
    spatial_dropoff_model = SpatialARIMA(dropoff_train)

    print("Grid searching for best parameters...")
    best_order, best_score, _ = spatial_dropoff_model.grid_search(p_range, d_range, q_range, metric='MAE')
    print(f"Best order for dropoff: {best_order}, score: {best_score}")

    print("Fitting models to all cells...")
    spatial_dropoff_model.fit()

    print("Forecasting...")
    forecast_dropoff = spatial_dropoff_model.forecast(steps=forecast_hours)

    # Process pickup data
    print("\nProcessing Pickup model...")
    pickup_train = pickup_data[train_st:train_end]
    spatial_pickup_model = SpatialARIMA(pickup_train)

    print("Grid searching for best parameters...")
    best_order, best_score, _ = spatial_pickup_model.grid_search(p_range, d_range, q_range, metric='MAE')
    print(f"Best order for pickup: {best_order}, score: {best_score}")

    print("Fitting models to all cells...")
    spatial_pickup_model.fit()

    print("Forecasting...")
    forecast_pickup = spatial_pickup_model.forecast(steps=forecast_hours)

    # Extract actual test data (ground truth)
    dropoff_actual = dropoff_data[test_st:test_end]
    pickup_actual = pickup_data[test_st:test_end]

    # Find the maximum value for plotting
    max_value = max(
        np.max(forecast_dropoff),
        np.max(forecast_pickup),
        np.max(dropoff_actual),
        np.max(pickup_actual)
    )

    # Plot everything in a single grid
    if forecast_hours == 1:
        # Handle the special case when forecast_hours is 1
        fig, axs = plt.subplots(4, 1, figsize=(5, 14), constrained_layout=True)

        data_grids = [
            (forecast_dropoff, "Dropoff Forecast"),
            (dropoff_actual, "Dropoff Actual"),
            (forecast_pickup, "Pickup Forecast"),
            (pickup_actual, "Pickup Actual")
        ]

        for row, (data_grid, label) in enumerate(data_grids):
            # For 3D arrays (forecast_hours=1), get the first slice
            # For actual data, it's already 2D
            if len(data_grid.shape) == 3:
                data = data_grid[0]  # Take first time step
            else:
                data = data_grid[0]  # Take first item

            plot_heatmap(axs[row], data, title=f"{label}", vmin=0, vmax=max_value)
    else:
        # Original code for multiple forecast hours
        fig, axs = plt.subplots(4, forecast_hours, figsize=(4 * forecast_hours, 14), constrained_layout=True)

        for row, (data_grid, label) in enumerate([
            (forecast_dropoff, "Dropoff Forecast"),
            (dropoff_actual, "Dropoff Actual"),
            (forecast_pickup, "Pickup Forecast"),
            (pickup_actual, "Pickup Actual")
        ]):
            for col in range(forecast_hours):
                if row % 2 == 0:  # forecast data
                    data = data_grid[col]
                else:  # actual data
                    data = data_grid[col]
                plot_heatmap(axs[row, col], data, title=f"{label} - Hour {col + 1}", vmin=0, vmax=max_value)

    # Add a big title
    fig.suptitle("Forecast vs Actual Heatmaps (Dropoff & Pickup)", fontsize=20)
    plt.savefig("spatial_forecast_vs_actual.png", dpi=300, bbox_inches='tight')

    # Calculate performance metrics
    dropoff_mae = mean_absolute_error(
        dropoff_actual.reshape(forecast_hours, -1),
        forecast_dropoff.reshape(forecast_hours, -1)
    )
    pickup_mae = mean_absolute_error(
        pickup_actual.reshape(forecast_hours, -1),
        forecast_pickup.reshape(forecast_hours, -1)
    )

    print(f"Dropoff Forecast MAE: {dropoff_mae:.4f}")
    print(f"Pickup Forecast MAE: {pickup_mae:.4f}")


if __name__ == "__main__":
    main()