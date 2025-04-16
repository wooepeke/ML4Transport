import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tqdm import tqdm
from matplotlib.colors import PowerNorm, TwoSlopeNorm
import torch
import os

# Import your model classes
from Arima.Arima import Arima
from Arima.SArima import Sarima
from model.STResNet import STResNet
from data.data_preparation import get_sequential_data

np.random.seed(42)


def load_and_prepare_data():
    """Load the data and prepare indices for train/test splits"""
    dropoff_data = np.load('data/dropoff_counts.npy')
    pickup_data = np.load('data/pickup_counts.npy')

    forecast_hours = 2  # Number of timesteps to visualize
    train_st = 500
    train_end = test_st = train_st + 240
    test_end = test_st + forecast_hours
    test_period_length = test_end - test_st

    return dropoff_data, pickup_data, train_st, train_end, test_st, test_end, test_period_length


def run_arima_model(train_data, test_period_length):
    """Run ARIMA model on each cell of the grid"""
    print("Running ARIMA model...")
    arima_forecast_grid = np.zeros((test_period_length, 32, 32))

    # Iterate over each cell in the 32x32 grid
    for i in tqdm(range(32)):
        for j in range(32):
            cell_data = train_data[:, i, j]
            # Only apply ARIMA if there's actual data (sum > 0)
            if np.sum(cell_data) > 0:
                try:
                    arima_model = Arima(p=1, d=0, q=2)
                    arima_model.fit(cell_data)
                    cell_forecast = arima_model.forecast(test_period_length)
                    # Ensure we have a numpy array
                    if isinstance(cell_forecast, pd.Series):
                        cell_forecast = cell_forecast.values

                    # Add to forecast grid
                    for t in range(test_period_length):
                        arima_forecast_grid[t, i, j] = cell_forecast[t]
                except:
                    # In case of error (e.g., not enough data), use zero or mean
                    pass

    return arima_forecast_grid


def run_sarima_model(train_data, test_period_length):
    """Run SARIMA model on each cell of the grid"""
    print("Running SARIMA model...")
    sarima_forecast_grid = np.zeros((test_period_length, 32, 32))

    # Iterate over each cell in the 32x32 grid
    for i in tqdm(range(32)):
        for j in range(32):
            cell_data = train_data[:, i, j]
            # Only apply SARIMA if there's actual data (sum > 0)
            if np.sum(cell_data) > 0:
                try:
                    sarima_model = Sarima(p=1, d=0, q=0, P=1, D=0, Q=0, s=24)
                    sarima_model.fit(cell_data)
                    cell_forecast = sarima_model.forecast(test_period_length)
                    # Ensure we have a numpy array
                    if isinstance(cell_forecast, pd.Series):
                        cell_forecast = cell_forecast.values

                    # Add to forecast grid
                    for t in range(test_period_length):
                        sarima_forecast_grid[t, i, j] = cell_forecast[t]
                except:
                    # In case of error (e.g., not enough data), use zero or mean
                    pass

    return sarima_forecast_grid


def generate_stresnet_predictions(model, X_test):
    """Generate predictions using the ST-ResNet model"""
    model.eval()
    with torch.no_grad():
        if torch.cuda.is_available():
            xc = X_test[0].to(torch.device("cuda:0"))
            xp = X_test[1].to(torch.device("cuda:0"))
            xt = X_test[2].to(torch.device("cuda:0"))
            ext = X_test[3].to(torch.device("cuda:0"))
        else:
            xc, xp, xt, ext = X_test

        predictions = model(xc, xp, xt, ext)

        if torch.cuda.is_available():
            predictions = predictions.cpu()

        return predictions.numpy()


def run_stresnet_model(X_test_torch, mmn):
    """Load and run the ST-ResNet model"""
    print("Running ST-ResNet model...")
    # Parameters for the model (use the same as in your ResNet diff.py)
    len_closeness = 3
    len_period = 1
    len_trend = 1
    nb_residual_unit = 8
    external_dim = X_test_torch[3].shape[1] if X_test_torch[3].numel() > 0 else 0

    # Initialize the model
    model = STResNet(
        learning_rate=0.0002,
        epoches=100,
        batch_size=32,
        len_closeness=len_closeness,
        len_period=len_period,
        len_trend=len_trend,
        external_dim=external_dim,
        map_heigh=32,
        map_width=32,
        nb_flow=2,
        nb_residual_unit=nb_residual_unit,
        data_min=mmn._min,
        data_max=mmn._max
    )

    # Load the trained model
    save_path = f"L{nb_residual_unit}_C{len_closeness}_P{len_period}_T{len_trend}/"
    # model.load_model("best")

    model.load_state_dict(torch.load(f"{save_path}/best.pt", map_location=torch.device('cpu')))
    model.eval()

    if torch.cuda.is_available():
        model = model.to(torch.device("cuda:0"))

    # Generate predictions
    predictions = generate_stresnet_predictions(model, X_test_torch)

    # Denormalize predictions
    predictions_denorm = denormalize_data(predictions, mmn)

    return predictions_denorm


def denormalize_data(data, mmn):
    """Denormalize the data using the min-max scaler"""
    return data * (mmn._max - mmn._min) / 2 + (mmn._max + mmn._min) / 2


def plot_heatmap(ax, data, title, vmin, vmax, cmap='plasma'):
    """Plot a heatmap of data"""
    # Ensure data is a numpy array
    if isinstance(data, pd.Series):
        data = data.values

    norm = PowerNorm(gamma=0.4, vmin=vmin, vmax=vmax)  # gamma < 1 boosts lower values
    cax = ax.imshow(data, cmap=cmap, norm=norm)
    ax.set_title(title)
    return cax


def plot_difference_heatmap(ax, forecast, actual, title):
    """
    Plot the difference between forecast and actual values.
    Red indicates overestimation, blue indicates underestimation.
    """
    # Ensure data is a numpy array
    if isinstance(forecast, pd.Series):
        forecast = forecast.values
    if isinstance(actual, pd.Series):
        actual = actual.values

    difference = forecast - actual

    # Find the maximum absolute difference for a symmetric colorbar
    max_diff = max(abs(np.min(difference)), abs(np.max(difference)))

    # Use diverging colormap with TwoSlopeNorm for better visualization
    norm = TwoSlopeNorm(vmin=-max_diff, vcenter=0, vmax=max_diff)
    cax = ax.imshow(difference, cmap='coolwarm', norm=norm)
    ax.set_title(title)
    return cax


def main():
    # Load the data and prepare indices
    print("Loading data...")
    dropoff_data, pickup_data, train_st, train_end, test_st, test_end, test_period_length = load_and_prepare_data()

    # Extract training and test data for both pickup and dropoff
    dropoff_train = dropoff_data[train_st:train_end]
    dropoff_test = dropoff_data[test_st:test_end]
    pickup_train = pickup_data[train_st:train_end]
    pickup_test = pickup_data[test_st:test_end]

    # Run ARIMA model
    dropoff_arima_forecast = run_arima_model(dropoff_train, test_period_length)
    pickup_arima_forecast = run_arima_model(pickup_train, test_period_length)

    # Run SARIMA model
    dropoff_sarima_forecast = run_sarima_model(dropoff_train, test_period_length)
    pickup_sarima_forecast = run_sarima_model(pickup_train, test_period_length)

    # Load data for ST-ResNet (using the data preparation function from your code)
    print("Preparing ST-ResNet data...")
    len_closeness = 3
    len_period = 1
    len_trend = 1
    X_train, Y_train, X_test, Y_test, mmn, external_dim = get_sequential_data(len_closeness, len_period, len_trend)

    # Convert test data to PyTorch tensors
    X_test_torch = [torch.Tensor(x) for x in X_test]

    # Run ST-ResNet model
    resnet_predictions = run_stresnet_model(X_test_torch, mmn)

    # Denormalize the test data for comparison
    Y_test_denorm = denormalize_data(Y_test, mmn)

    # Extract the first few timesteps from ResNet predictions to match ARIMA/SARIMA
    resnet_dropoff = resnet_predictions[:test_period_length, 1]  # Flow index 1 for dropoff
    resnet_pickup = resnet_predictions[:test_period_length, 0]  # Flow index 0 for pickup

    # Extract ground truth for comparison
    gt_dropoff = Y_test_denorm[:test_period_length, 1]
    gt_pickup = Y_test_denorm[:test_period_length, 0]

    # Ensure all data is numpy arrays, not pandas Series
    if isinstance(dropoff_arima_forecast, pd.Series):
        dropoff_arima_forecast = dropoff_arima_forecast.values
    if isinstance(dropoff_sarima_forecast, pd.Series):
        dropoff_sarima_forecast = dropoff_sarima_forecast.values
    if isinstance(pickup_arima_forecast, pd.Series):
        pickup_arima_forecast = pickup_arima_forecast.values
    if isinstance(pickup_sarima_forecast, pd.Series):
        pickup_sarima_forecast = pickup_sarima_forecast.values

    # Print shapes for debugging
    print(f"Shapes - Dropoff ARIMA: {dropoff_arima_forecast.shape}")
    print(f"Shapes - Dropoff SARIMA: {dropoff_sarima_forecast.shape}")
    print(f"Shapes - Dropoff ResNet: {resnet_dropoff.shape}")
    print(f"Shapes - Dropoff Ground Truth: {gt_dropoff.shape}")

    # Maximum value for color scaling across all plots
    max_value = max(
        np.max(dropoff_arima_forecast), np.max(dropoff_sarima_forecast),
        np.max(resnet_dropoff), np.max(gt_dropoff),
        np.max(pickup_arima_forecast), np.max(pickup_sarima_forecast),
        np.max(resnet_pickup), np.max(gt_pickup)
    )

    # Create visualization
    print("Creating visualizations...")
    os.makedirs("comparison_images", exist_ok=True)

    # For each timestep, create a figure with model comparisons
    for t in range(test_period_length):
        # Create a figure with 2 rows and 4 columns for each model and ground truth
        fig, axs = plt.subplots(2, 4, figsize=(16, 8), constrained_layout=True)

        def reshape_if_flat(x):
            return x.reshape(32, 32) if x.ndim == 1 else x


        # Row 0: Original heatmaps for dropoff
        # Ground Truth
        cax0 = plot_heatmap(axs[0, 0], gt_dropoff[t], f"Ground Truth (t={t + 1})", vmin=0, vmax=max_value)
        # ARIMA
        plot_heatmap(axs[0, 1], dropoff_arima_forecast[t], f"ARIMA (p=1,d=0,q=2) (t={t + 1})", vmin=0, vmax=max_value)
        # SARIMA
        plot_heatmap(axs[0, 2], dropoff_sarima_forecast[t], f"SARIMA (p=1,d=0,q=0,s=24) (t={t + 1})", vmin=0,
                     vmax=max_value)
        # ResNet
        plot_heatmap(axs[0, 3], resnet_dropoff[t], f"ST-ResNet (t={t + 1})", vmin=0, vmax=max_value)

        # Row 1: Difference heatmaps
        # Empty plot for ground truth (no difference with itself)
        axs[1, 0].axis('off')
        axs[1, 0].set_title("Difference with Ground Truth")

        # ARIMA difference
        cax1 = plot_difference_heatmap(axs[1, 1], dropoff_arima_forecast[t], gt_dropoff[t], "ARIMA Difference")
        # SARIMA difference
        plot_difference_heatmap(axs[1, 2], dropoff_sarima_forecast[t], gt_dropoff[t], "SARIMA Difference")
        # ResNet difference
        plot_difference_heatmap(axs[1, 3], resnet_dropoff[t], gt_dropoff[t], "ST-ResNet Difference")

        # Add colorbars
        plt.colorbar(cax0, ax=axs[0, :], location='top', shrink=0.6, aspect=40, pad=0.01)
        plt.colorbar(cax1, ax=axs[1, 1:], location='bottom', shrink=0.6, aspect=40, pad=0.01)

        # Add a big title
        fig.suptitle(f"Dropoff Forecast Comparison - Timestep {t + 1}", fontsize=16)

        # Save the figure
        plt.savefig(f"comparison_images/dropoff_comparison_t{t + 1}.png", dpi=300, bbox_inches='tight')
        plt.close(fig)

        # Create another figure for pickup comparison
        fig, axs = plt.subplots(2, 4, figsize=(16, 8), constrained_layout=True)

        # Row 0: Original heatmaps for pickup
        # Ground Truth
        cax0 = plot_heatmap(axs[0, 0], gt_pickup[t], f"Ground Truth (t={t + 1})", vmin=0, vmax=max_value)
        # ARIMA
        plot_heatmap(axs[0, 1], pickup_arima_forecast[t], f"ARIMA (p=1,d=0,q=2) (t={t + 1})", vmin=0, vmax=max_value)
        # SARIMA
        plot_heatmap(axs[0, 2], pickup_sarima_forecast[t], f"SARIMA (p=1,d=0,q=0,s=24) (t={t + 1})", vmin=0,
                     vmax=max_value)
        # ResNet
        plot_heatmap(axs[0, 3], resnet_pickup[t], f"ST-ResNet (t={t + 1})", vmin=0, vmax=max_value)

        # Row 1: Difference heatmaps for pickup
        # Empty plot for ground truth (no difference with itself)
        axs[1, 0].axis('off')
        axs[1, 0].set_title("Difference with Ground Truth")

        # ARIMA difference
        cax1 = plot_difference_heatmap(axs[1, 1], pickup_arima_forecast[t], gt_pickup[t], "ARIMA Difference")
        # SARIMA difference
        plot_difference_heatmap(axs[1, 2], pickup_sarima_forecast[t], gt_pickup[t], "SARIMA Difference")
        # ResNet difference
        plot_difference_heatmap(axs[1, 3], resnet_pickup[t], gt_pickup[t], "ST-ResNet Difference")

        # Add colorbars
        plt.colorbar(cax0, ax=axs[0, :], location='top', shrink=0.6, aspect=40, pad=0.01)
        plt.colorbar(cax1, ax=axs[1, 1:], location='bottom', shrink=0.6, aspect=40, pad=0.01)

        # Add a big title
        fig.suptitle(f"Pickup Forecast Comparison - Timestep {t + 1}", fontsize=16)

        # Save the figure
        plt.savefig(f"comparison_images/pickup_comparison_t{t + 1}.png", dpi=300, bbox_inches='tight')
        plt.close(fig)

    print("All comparisons completed and saved to 'comparison_images' folder.")

    # Calculate and print evaluation metrics
    print("Calculating evaluation metrics...")

    # ARIMA metrics
    dropoff_arima_mae = np.mean(np.abs(dropoff_arima_forecast - gt_dropoff))
    dropoff_arima_rmse = np.sqrt(np.mean((dropoff_arima_forecast - gt_dropoff) ** 2))
    pickup_arima_mae = np.mean(np.abs(pickup_arima_forecast - gt_pickup))
    pickup_arima_rmse = np.sqrt(np.mean((pickup_arima_forecast - gt_pickup) ** 2))

    # SARIMA metrics
    dropoff_sarima_mae = np.mean(np.abs(dropoff_sarima_forecast - gt_dropoff))
    dropoff_sarima_rmse = np.sqrt(np.mean((dropoff_sarima_forecast - gt_dropoff) ** 2))
    pickup_sarima_mae = np.mean(np.abs(pickup_sarima_forecast - gt_pickup))
    pickup_sarima_rmse = np.sqrt(np.mean((pickup_sarima_forecast - gt_pickup) ** 2))

    # ST-ResNet metrics
    dropoff_resnet_mae = np.mean(np.abs(resnet_dropoff - gt_dropoff))
    dropoff_resnet_rmse = np.sqrt(np.mean((resnet_dropoff - gt_dropoff) ** 2))
    pickup_resnet_mae = np.mean(np.abs(resnet_pickup - gt_pickup))
    pickup_resnet_rmse = np.sqrt(np.mean((resnet_pickup - gt_pickup) ** 2))

    # Print results
    print("\n===== Evaluation Metrics =====")
    print("\nDropoff Model Comparison:")
    print(f"ARIMA - MAE: {dropoff_arima_mae:.4f}, RMSE: {dropoff_arima_rmse:.4f}")
    print(f"SARIMA - MAE: {dropoff_sarima_mae:.4f}, RMSE: {dropoff_sarima_rmse:.4f}")
    print(f"ST-ResNet - MAE: {dropoff_resnet_mae:.4f}, RMSE: {dropoff_resnet_rmse:.4f}")

    print("\nPickup Model Comparison:")
    print(f"ARIMA - MAE: {pickup_arima_mae:.4f}, RMSE: {pickup_arima_rmse:.4f}")
    print(f"SARIMA - MAE: {pickup_sarima_mae:.4f}, RMSE: {pickup_sarima_rmse:.4f}")
    print(f"ST-ResNet - MAE: {pickup_resnet_mae:.4f}, RMSE: {pickup_resnet_rmse:.4f}")

    # Save metrics to file
    metrics = {
        "dropoff": {
            "arima": {"mae": float(dropoff_arima_mae), "rmse": float(dropoff_arima_rmse)},
            "sarima": {"mae": float(dropoff_sarima_mae), "rmse": float(dropoff_sarima_rmse)},
            "resnet": {"mae": float(dropoff_resnet_mae), "rmse": float(dropoff_resnet_rmse)}
        },
        "pickup": {
            "arima": {"mae": float(pickup_arima_mae), "rmse": float(pickup_arima_rmse)},
            "sarima": {"mae": float(pickup_sarima_mae), "rmse": float(pickup_sarima_rmse)},
            "resnet": {"mae": float(pickup_resnet_mae), "rmse": float(pickup_resnet_rmse)}
        }
    }

    import json
    with open("comparison_images/evaluation_metrics.json", "w") as f:
        json.dump(metrics, f, indent=4)


if __name__ == "__main__":
    main()