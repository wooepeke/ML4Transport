import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from Arima import Arima
from tqdm import tqdm
from matplotlib.colors import PowerNorm

np.random.seed(42)

def forecast_grid(model_class, train_data, forecast_period, alpha_range, theta_range):
    """
    Forecast demand for all grid cells using the given ARIMA model.
    Capping negative forecasted values at 0.
    """
    forecast_grid = np.zeros((32, 32, forecast_period))  # Initialize the grid for predictions

    for i in tqdm(range(32)):
        for j in range(32):
            # Extract data for the current grid cell
            cell_data = train_data[:, i, j]
            
            # Initialize and fit ARIMA model
            model = model_class(cell_data)
            
            # Grid search for optimal parameters
            #best_params, _ = model.grid_search(alpha_range, theta_range, metric='MAE')
            #model.alpha, model.theta = best_params
            model.alpha, model.theta = 0.1, 0.05
            model.run_model()

            # Forecast for the given period
            forecast_values = model.forecast_test_period(forecast_period)

            # Cap any negative forecast values at 0
            forecast_grid[i, j, :] = np.maximum(forecast_values, 0)

    return forecast_grid


def plot_heatmap(ax, data, title, vmin, vmax, cmap='plasma'):
    norm = PowerNorm(gamma=0.4, vmin=vmin, vmax=vmax)  # gamma < 1 boosts lower values
    cax = ax.imshow(data, cmap=cmap, norm=norm)
    ax.set_title(title)
    plt.colorbar(cax, ax=ax, shrink=0.8)


def main():
    # Load the data
    dropoff_data = np.load(r'data\dropoff_counts.npy')  # (n, 32, 32)
    pickup_data = np.load(r'data\pickup_counts.npy')    # (n, 32, 32)

    # Define the training and testing indices
    forecast_hours = 2  # Maximum forecast period of 4 hours
    T = 24  # number of time intervals in one day
    train_st = 0
    train_end = test_st = (train_st + 240)
    test_end = test_st + forecast_hours

    # --- Dropoff Forecast ---
    print("Optimizing Dropoff model...")
    dropoff_train = dropoff_data[train_st:train_end]

    # Grid search for optimal parameters
    alpha_range = np.linspace(0.1, 0.9, 50)
    theta_range = np.linspace(0.1, 0.9, 50)

    forecast_dropoff_grid = forecast_grid(Arima, dropoff_train, forecast_hours, alpha_range, theta_range)

    # --- Pickup Forecast ---
    print("\nOptimizing Pickup model...")
    pickup_train = pickup_data[train_st:train_end]

    forecast_pickup_grid = forecast_grid(Arima, pickup_train, forecast_hours, alpha_range, theta_range)

    # Find the maximum value across both dropoff and pickup forecast grids to set a common scale
    max_value = max(np.max(forecast_dropoff_grid), np.max(forecast_pickup_grid))

    # --- Extract actual test data (ground truth) ---
    dropoff_actual = dropoff_data[test_st:test_end]
    pickup_actual = pickup_data[test_st:test_end]

    # Recompute max for plotting, including actuals
    max_value = max(max_value, np.max(dropoff_actual), np.max(pickup_actual))

    # --- Plot everything in a single 4x4 grid with better spacing ---
    fig, axs = plt.subplots(4, forecast_hours, figsize=(4 * forecast_hours, 14), constrained_layout=True)
    
    row_titles = ["Dropoff Forecast", "Dropoff Actual", "Pickup Forecast", "Pickup Actual"]
    for row, (data_grid, label) in enumerate([
        (forecast_dropoff_grid, "Dropoff Forecast"),
        (dropoff_actual, "Dropoff Actual"),
        (forecast_pickup_grid, "Pickup Forecast"),
        (pickup_actual, "Pickup Actual")
    ]):
        for col in range(forecast_hours):
            data = data_grid[:, :, col] if row % 2 == 0 else data_grid[col]
            plot_heatmap(axs[row, col], data, title=f"{label} - Hour {col+1}", vmin=0, vmax=max_value)

    # Add a big title
    fig.suptitle("Forecast vs Actual Heatmaps (Dropoff & Pickup)", fontsize=20)
    plt.savefig("forecast_vs_actual_heatmaps.png", dpi=300, bbox_inches='tight')


if __name__ == "__main__":
    main()
