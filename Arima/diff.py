import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from Arima import Arima
from tqdm import tqdm
from matplotlib.colors import PowerNorm, TwoSlopeNorm
import json
import os

np.random.seed(42)


def forecast_grid(model_class, train_data, forecast_period, alpha_range, theta_range, name):
    """
    Forecast demand for all grid cells using the given ARIMA model.
    Capping negative forecasted values at 0.
    """
    forecast_grid = np.zeros((32, 32, forecast_period))  # Initialize the grid for predictions
    model = model_class()
    for i in tqdm(range(32)):
        for j in range(32):
            # Extract data for the current grid cell
            cell_data = train_data[:, i, j]

            # Grid search for optimal parameters
            # best_params, _ = model.grid_search(alpha_range, theta_range, metric='MAE')
            # model.alpha, model.theta = best_params
            model.alpha, model.theta = 0.5, 0.5
            model.run_model(cell_data)

            # Forecast for the given period
            forecast_values = model.forecast_test_period(forecast_period)

            # Cap any negative forecast values at 0
            forecast_grid[i, j, :] = np.maximum(forecast_values, 0)

    pixel_mae, pixel_rmse = model.MAEs, model.RMSEs
    total_mae, total_rmse = model.compute_full_metrics()

    data = {
        "pixel_mae":pixel_mae,
        "pixel_rmse":pixel_rmse,
        "total_mae":total_mae,
        "total_rmse":total_rmse
    }
    
    output_folder = "Arima\metrics"
    os.makedirs(output_folder, exist_ok=True)

    with open(f"{output_folder}/forecast_data_{name}.json", "w") as f:
        json.dump(data, f, indent=4)

    return forecast_grid


def plot_heatmap(ax, data, title, vmin, vmax, cmap='plasma'):
    norm = PowerNorm(gamma=0.4, vmin=vmin, vmax=1000)  # gamma < 1 boosts lower values
    cax = ax.imshow(data, cmap=cmap, norm=norm)
    ax.set_title(title)
    plt.colorbar(cax, ax=ax, shrink=0.8)


def plot_difference_heatmap(ax, forecast, actual, title):
    """
    Plot the difference between forecast and actual values.
    Red indicates overestimation, blue indicates underestimation.
    """
    difference = forecast - actual

    # Find the maximum absolute difference for a symmetric colorbar
    max_diff = max(abs(np.min(difference)), abs(np.max(difference)))

    # Use diverging colormap with TwoSlopeNorm for better visualization
    norm = TwoSlopeNorm(vmin=-max_diff, vcenter=0, vmax=max_diff)
    cax = ax.imshow(difference, cmap='coolwarm', norm=norm)
    ax.set_title(title)
    cbar = plt.colorbar(cax, ax=ax, shrink=0.8)
    cbar.set_label('Forecast - Actual')


def main():
    # Load the data
    dropoff_data = np.load(r'data\dropoff_counts.npy')  # (n, 32, 32)
    pickup_data = np.load(r'data\pickup_counts.npy')  # (n, 32, 32)

    # Define the training and testing indices
    forecast_hours = 2  # Maximum forecast period of 4 hours
    T = 24  # number of time intervals in one day
    train_st = 2600 - 240
    train_end = test_st = (train_st + 240)
    test_end = test_st + forecast_hours

    # --- Dropoff Forecast ---
    print("Optimizing Dropoff model...")
    dropoff_train = dropoff_data[train_st:train_end]

    # Grid search for optimal parameters
    alpha_range = np.linspace(0.1, 0.9, 50)
    theta_range = np.linspace(0.1, 0.9, 50)

    forecast_dropoff_grid = forecast_grid(Arima, dropoff_train, forecast_hours, alpha_range, theta_range, "dropoff")

    # --- Pickup Forecast ---
    print("\nOptimizing Pickup model...")
    pickup_train = pickup_data[train_st:train_end]

    forecast_pickup_grid = forecast_grid(Arima, pickup_train, forecast_hours, alpha_range, theta_range, "pickup")

    # --- Extract actual test data (ground truth) ---
    dropoff_actual = dropoff_data[test_st:test_end]
    pickup_actual = pickup_data[test_st:test_end]

    # Find the maximum value across both dropoff and pickup forecast grids to set a common scale
    max_value = max(np.max(forecast_dropoff_grid), np.max(forecast_pickup_grid))
    max_value = max(max_value, np.max(dropoff_actual), np.max(pickup_actual))

    # --- Plot everything in a 6x2 grid with better spacing (added difference plots) ---
    fig, axs = plt.subplots(6, forecast_hours, figsize=(4 * forecast_hours, 20), constrained_layout=True)

    # Plot forecast, actual, and difference for dropoff
    for col in range(forecast_hours):
        # Dropoff forecast
        plot_heatmap(axs[0, col], forecast_dropoff_grid[:, :, col],
                     f"Dropoff Forecast - Hour {col + 1}", vmin=0, vmax=max_value)

        # Dropoff actual
        plot_heatmap(axs[1, col], dropoff_actual[col],
                     f"Dropoff Actual - Hour {col + 1}", vmin=0, vmax=max_value)

        # Dropoff difference
        plot_difference_heatmap(axs[2, col], forecast_dropoff_grid[:, :, col], dropoff_actual[col],
                                f"Dropoff Difference - Hour {col + 1}")

        # Pickup forecast
        plot_heatmap(axs[3, col], forecast_pickup_grid[:, :, col],
                     f"Pickup Forecast - Hour {col + 1}", vmin=0, vmax=max_value)

        # Pickup actual
        plot_heatmap(axs[4, col], pickup_actual[col],
                     f"Pickup Actual - Hour {col + 1}", vmin=0, vmax=max_value)

        # Pickup difference
        plot_difference_heatmap(axs[5, col], forecast_pickup_grid[:, :, col], pickup_actual[col],
                                f"Pickup Difference - Hour {col + 1}")

    # Add row labels to clarify the visualization
    row_labels = ["Dropoff Forecast", "Dropoff Actual", "Dropoff Difference",
                  "Pickup Forecast", "Pickup Actual", "Pickup Difference"]

    for i, label in enumerate(row_labels):
        axs[i, 0].set_ylabel(label, fontsize=12, rotation=90, labelpad=10)
    
    output_folder = "Arima\images"
    # Add a big title
    fig.suptitle("Forecast vs Actual Heatmaps with Differences", fontsize=20)
    plt.savefig(f"{output_folder}/forecast_vs_actual_with_differences_{forecast_hours}_hours.png", dpi=300, bbox_inches='tight')

    # Create a separate figure focusing just on the differences
    fig_diff, axs_diff = plt.subplots(2, forecast_hours, figsize=(4 * forecast_hours, 8), constrained_layout=True)

    for col in range(forecast_hours):
        # Dropoff difference
        plot_difference_heatmap(axs_diff[0, col], forecast_dropoff_grid[:, :, col], dropoff_actual[col],
                                f"Dropoff Difference - Hour {col + 1}")

        # Pickup difference
        plot_difference_heatmap(axs_diff[1, col], forecast_pickup_grid[:, :, col], pickup_actual[col],
                                f"Pickup Difference - Hour {col + 1}")

    # Add row labels
    axs_diff[0, 0].set_ylabel("Dropoff", fontsize=12)
    axs_diff[1, 0].set_ylabel("Pickup", fontsize=12)

    fig_diff.suptitle("Forecast Error Heatmaps (Forecast - Actual)", fontsize=20)
    plt.savefig(f"{output_folder}/forecast_error_heatmaps_{forecast_hours}_hours.png", dpi=300, bbox_inches='tight')

if __name__ == "__main__":
    main()