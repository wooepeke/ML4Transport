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
        self.models = None
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

    def evaluate_test_data(self, test_data):
        """
        Evaluate the model on test data.

        Parameters:
        -----------
        test_data : np.ndarray
            Test data to evaluate against forecasts

        Returns:
        --------
        metrics : dict
            Dictionary containing evaluation metrics
        """
        if self.models is None:
            raise ValueError("Run fit() before evaluating")

        # Reshape test data to match our format if needed
        if len(test_data.shape) == 3:
            # Already in 3D format
            test_flattened = test_data.reshape(test_data.shape[0], -1)
        else:
            # Assuming it's a 2D matrix (time, features)
            test_flattened = test_data

        forecast_steps = test_flattened.shape[0]
        forecasts = self.forecast(steps=forecast_steps)
        forecasts_flattened = forecasts.reshape(forecast_steps, -1)

        # Calculate metrics
        mae = mean_absolute_error(test_flattened, forecasts_flattened)
        mse = mean_squared_error(test_flattened, forecasts_flattened)
        rmse = np.sqrt(mse)

        return {
            'MAE': mae,
            'MSE': mse,
            'RMSE': rmse
        }


def plot_heatmap(ax, data, title, vmin, vmax, cmap='plasma'):
    norm = PowerNorm(gamma=0.4, vmin=vmin, vmax=vmax)  # gamma < 1 boosts lower values
    cax = ax.imshow(data, cmap=cmap, norm=norm)
    ax.set_title(title)
    plt.colorbar(cax, ax=ax, shrink=0.8)


def main():
    # Load the data
    dropoff_data = np.load(r'data\dropoff_counts.npy')  # (time_steps, height, width)
    pickup_data = np.load(r'data\pickup_counts.npy')  # (time_steps, height, width)

    # Define the training and testing indices
    forecast_hours = 24  # Maximum forecast period of 4 hours
    train_st = 0
    train_end = test_st = (train_st + 240)
    test_end = test_st + forecast_hours

    # Define parameter ranges for grid search
    p_range = range(0, 3)  # AR terms
    d_range = range(0, 2)  # Differencing
    q_range = range(0, 3)  # MA terms

    # --- Dropoff Model ---
    print("Processing Dropoff model...")
    dropoff_train = dropoff_data[train_st:train_end]
    dropoff_test = dropoff_data[test_st:test_end]

    spatial_dropoff_model = SpatialARIMA(dropoff_train)

    print("Grid searching for best parameters...")
    best_order, best_score, results_df = spatial_dropoff_model.grid_search(p_range, d_range, q_range, metric='MAE')
    print(f"Best Dropoff Params: p={best_order[0]}, d={best_order[1]}, q={best_order[2]} with MAE={best_score:.4f}")

    print("Fitting models to all cells...")
    spatial_dropoff_model.fit()

    print("Forecasting...")
    forecast_dropoff = spatial_dropoff_model.forecast(steps=forecast_hours)

    # --- Pickup Model ---
    print("\nProcessing Pickup model...")
    pickup_train = pickup_data[train_st:train_end]
    pickup_test = pickup_data[test_st:test_end]

    spatial_pickup_model = SpatialARIMA(pickup_train)

    print("Grid searching for best parameters...")
    best_order, best_score, results_df = spatial_pickup_model.grid_search(p_range, d_range, q_range, metric='MAE')
    print(f"Best Pickup Params: p={best_order[0]}, d={best_order[1]}, q={best_order[2]} with MAE={best_score:.4f}")

    print("Fitting models to all cells...")
    spatial_pickup_model.fit()

    print("Forecasting...")
    forecast_pickup = spatial_pickup_model.forecast(steps=forecast_hours)

    # --- Visualize Results ---
    # First, let's create visualization using grid heatmaps
    fig1, axs1 = plt.subplots(2, 2, figsize=(15, 12), constrained_layout=True)

    # Find the maximum value for consistent colorbar scaling
    max_value = max(
        np.max(forecast_dropoff),
        np.max(forecast_pickup),
        np.max(dropoff_test),
        np.max(pickup_test)
    )

    # Plot first time step forecast vs. actual as heatmaps
    plot_heatmap(axs1[0, 0], forecast_dropoff[0], "Dropoff Forecast (Hour 1)", 0, max_value)
    plot_heatmap(axs1[0, 1], dropoff_test[0], "Dropoff Actual (Hour 1)", 0, max_value)
    plot_heatmap(axs1[1, 0], forecast_pickup[0], "Pickup Forecast (Hour 1)", 0, max_value)
    plot_heatmap(axs1[1, 1], pickup_test[0], "Pickup Actual (Hour 1)", 0, max_value)

    fig1.suptitle("Spatial ARIMA Forecast vs. Actual (Hour 1)", fontsize=16)

    # --- Now create temporal plots similar to original code ---
    # Calculate cell-wise average to visualize temporal patterns
    dropoff_train_avg = np.mean(dropoff_train, axis=(1, 2))
    pickup_train_avg = np.mean(pickup_train, axis=(1, 2))

    dropoff_test_avg = np.mean(dropoff_test, axis=(1, 2))
    pickup_test_avg = np.mean(pickup_test, axis=(1, 2))

    forecast_dropoff_avg = np.mean(forecast_dropoff, axis=(1, 2))
    forecast_pickup_avg = np.mean(forecast_pickup, axis=(1, 2))

    # Create temporal plots
    fig2, axs2 = plt.subplots(3, 1, figsize=(15, 15), gridspec_kw={'height_ratios': [1, 1, 1]})

    # Dropoff Temporal Plot
    axs2[0].plot(range(train_st, train_end), dropoff_train_avg,
                 label='Training Data', color='blue', alpha=0.6)
    axs2[0].plot(range(test_st, test_end), dropoff_test_avg,
                 label='Test Data', color='orange', alpha=0.6)
    axs2[0].plot(range(test_st, test_end), forecast_dropoff_avg,
                 label='Test Period Forecast', color='purple', linestyle='-.')

    axs2[0].set_title('Average Dropoff Counts Over Time')
    axs2[0].set_ylabel('Average Dropoff Count')
    axs2[0].legend()
    axs2[0].grid(True, alpha=0.3)

    # Pickup Temporal Plot
    axs2[1].plot(range(train_st, train_end), pickup_train_avg,
                 label='Training Data', color='blue', alpha=0.6)
    axs2[1].plot(range(test_st, test_end), pickup_test_avg,
                 label='Test Data', color='orange', alpha=0.6)
    axs2[1].plot(range(test_st, test_end), forecast_pickup_avg,
                 label='Test Period Forecast', color='purple', linestyle='-.')

    axs2[1].set_title('Average Pickup Counts Over Time')
    axs2[1].set_ylabel('Average Pickup Count')
    axs2[1].legend()
    axs2[1].grid(True, alpha=0.3)

    # Add vertical line to mark where train data ends for main plots
    for ax in axs2[:2]:
        ax.axvline(x=train_end, color='gray', linestyle='--', alpha=0.5)
        ax.annotate('End of Training', xy=(train_end, ax.get_ylim()[1] * 0.85),
                    xytext=(train_end + 2, ax.get_ylim()[1] * 0.85),
                    arrowprops=dict(facecolor='gray', shrink=0.05, width=1, headwidth=6),
                    fontsize=9, horizontalalignment='left')

    # Add text box with metrics for main plots
    dropoff_metrics = spatial_dropoff_model.evaluate_test_data(dropoff_test)
    dropoff_text = f"Test Metrics:\nMAE: {dropoff_metrics['MAE']:.2f}\nRMSE: {dropoff_metrics['RMSE']:.2f}"
    axs2[0].text(0.02, 0.05, dropoff_text, transform=axs2[0].transAxes,
                 bbox=dict(facecolor='white', alpha=0.8, boxstyle='round'))

    pickup_metrics = spatial_pickup_model.evaluate_test_data(pickup_test)
    pickup_text = f"Test Metrics:\nMAE: {pickup_metrics['MAE']:.2f}\nRMSE: {pickup_metrics['RMSE']:.2f}"
    axs2[1].text(0.02, 0.05, pickup_text, transform=axs2[1].transAxes,
                 bbox=dict(facecolor='white', alpha=0.8, boxstyle='round'))

    # New subplot: Zoomed-in view of the last 20 timesteps of training + test period
    axs2[2].set_title('Zoomed View: Last 20 Training Steps + Test Period')
    axs2[2].set_xlabel('Time')

    # Define the zoom window
    zoom_start = train_end - 20  # Last 20 timesteps of training
    zoom_end = test_end  # Including all test data

    # Plot dropoff data (zoomed)
    axs2[2].plot(range(zoom_start, train_end), dropoff_train_avg[zoom_start - train_st:],
                 label='Dropoff Training', color='blue', alpha=0.6)
    axs2[2].plot(range(test_st, test_end), dropoff_test_avg,
                 label='Dropoff Test', color='blue')
    axs2[2].plot(range(test_st, test_end), forecast_dropoff_avg,
                 label='Dropoff Forecast', color='blue', linestyle='-.')

    # Plot pickup data (zoomed)
    axs2[2].plot(range(zoom_start, train_end), pickup_train_avg[zoom_start - train_st:],
                 label='Pickup Training', color='green', alpha=0.6)
    axs2[2].plot(range(test_st, test_end), pickup_test_avg,
                 label='Pickup Test', color='green')
    axs2[2].plot(range(test_st, test_end), forecast_pickup_avg,
                 label='Pickup Forecast', color='green', linestyle='-.')

    # Add vertical line for end of training in zoomed plot
    axs2[2].axvline(x=train_end, color='red', linestyle='--', alpha=0.5)
    axs2[2].annotate('End of Training', xy=(train_end, axs2[2].get_ylim()[1] * 0.9),
                     xytext=(train_end + 0.5, axs2[2].get_ylim()[1] * 0.9),
                     arrowprops=dict(facecolor='red', shrink=0.05, width=1, headwidth=6),
                     fontsize=9, horizontalalignment='left')

    axs2[2].grid(True, alpha=0.3)
    axs2[2].legend(loc='upper left')

    # Calculate and display test data metrics
    print("\nEvaluating dropoff model on test data:")
    for k, v in dropoff_metrics.items():
        print(f"  {k}: {v:.4f}")

    print("\nEvaluating pickup model on test data:")
    for k, v in pickup_metrics.items():
        print(f"  {k}: {v:.4f}")

    # Adjust layout
    fig2.suptitle("Spatial ARIMA Model with Test Period Forecast")
    plt.tight_layout()
    plt.subplots_adjust(top=0.95, hspace=0.3)  # Adjust space for title and between subplots

    plt.show()


if __name__ == "__main__":
    main()