import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from Arima import Arima
from tqdm import tqdm
from matplotlib.colors import PowerNorm

np.random.seed(42)

from arima2 import ARIMAGridSearch  # Assuming you saved the previous class in this file

def forecast_grid(model_class, train_data, forecast_period, p_range, d_range, q_range):
    forecast_grid = np.zeros((32, 32, forecast_period))  # Initialize the grid for predictions

    for i in tqdm(range(32)):
        for j in range(32):
            cell_data = train_data[:, i, j]
            model = model_class(cell_data)

            # Grid search for best (p,d,q)
            try:
                best_order, _, _ = model.grid_search(p_range, d_range, q_range, metric='MAE')
                forecast_values = model.forecast(steps=forecast_period)
            except Exception as e:
                forecast_values = np.zeros(forecast_period)  # fallback if model fails

            forecast_grid[i, j, :] = np.maximum(forecast_values, 0)  # cap at 0

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
    forecast_hours = 1  # Maximum forecast period of 4 hours
    T = 24  # number of time intervals in one day
    train_st = 0
    train_end = test_st = (train_st + 240)
    test_end = test_st + forecast_hours

    p_range = range(3)  # Adjust depending on complexity and performance
    d_range = range(2)
    q_range = range(3)
    print("Optimizing Dropoff model...")
    dropoff_train = dropoff_data[train_st:train_end]
    forecast_dropoff_grid = forecast_grid(ARIMAGridSearch, dropoff_train, forecast_hours, p_range, d_range, q_range)

    print("\nOptimizing Pickup model...")
    pickup_train = pickup_data[train_st:train_end]
    forecast_pickup_grid = forecast_grid(ARIMAGridSearch, pickup_train, forecast_hours, p_range, d_range, q_range)


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
