import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.colors import PowerNorm, TwoSlopeNorm
from model.STResNet import STResNet
from data.data_preparation import get_sequential_data
from tqdm import tqdm


def generate_stresnet_predictions(model, X_test):
    """
    Generate predictions using the ST-ResNet model
    """
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


def plot_heatmap(ax, data, title, vmin, vmax, cmap='plasma'):
    """
    Plot a heatmap of data
    """
    norm = PowerNorm(gamma=0.4, vmin=vmin, vmax=vmax)  # gamma < 1 boosts lower values
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


def denormalize_data(data, mmn):
    """
    Denormalize the data using the min-max scaler
    """
    return data * (mmn._max - mmn._min) / 2 + (mmn._max + mmn._min) / 2


def main():
    # Parameters
    len_closeness = 3
    len_period = 1
    len_trend = 1
    nb_residual_unit = 4

    # Load the data
    X_train, Y_train, X_test, Y_test, mmn, external_dim = get_sequential_data(len_closeness, len_period, len_trend)

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
    model.load_model("best")

    if torch.cuda.is_available():
        model = model.to(torch.device("cuda:0"))

    # Convert test data to PyTorch tensors
    X_test_torch = [torch.Tensor(x) for x in X_test]

    # Generate predictions
    print("Generating predictions...")
    predictions = generate_stresnet_predictions(model, X_test_torch)

    # Denormalize predictions and actual values
    print("Denormalizing data...")
    predictions_denorm = denormalize_data(predictions, mmn)
    Y_test_denorm = denormalize_data(Y_test, mmn)

    # Number of timesteps to visualize
    num_timesteps = min(2, predictions.shape[0])  # Use 2 timesteps as in the ARIMA example

    # Maximum value for color scaling across all plots
    max_value = max(np.max(predictions_denorm), np.max(Y_test_denorm))

    # Create visualization
    print("Creating visualizations...")

    # Create a 6 x num_timesteps grid of subplots
    fig, axs = plt.subplots(6, num_timesteps, figsize=(4 * num_timesteps, 20), constrained_layout=True)

    # If num_timesteps is 1, axs will be 1D, so we need to reshape it
    if num_timesteps == 1:
        axs = axs.reshape(6, 1)

    # Loop through timesteps
    for t in range(num_timesteps):
        # Dropoff - Predicted
        plot_heatmap(axs[0, t], predictions_denorm[t, 0],
                     f"Dropoff Forecast - Timestep {t + 1}", vmin=0, vmax=max_value)

        # Dropoff - Actual
        plot_heatmap(axs[1, t], Y_test_denorm[t, 0],
                     f"Dropoff Actual - Timestep {t + 1}", vmin=0, vmax=max_value)

        # Dropoff - Difference
        plot_difference_heatmap(axs[2, t], predictions_denorm[t, 0], Y_test_denorm[t, 0],
                                f"Dropoff Difference - Timestep {t + 1}")

        # Pickup - Predicted
        plot_heatmap(axs[3, t], predictions_denorm[t, 1],
                     f"Pickup Forecast - Timestep {t + 1}", vmin=0, vmax=max_value)

        # Pickup - Actual
        plot_heatmap(axs[4, t], Y_test_denorm[t, 1],
                     f"Pickup Actual - Timestep {t + 1}", vmin=0, vmax=max_value)

        # Pickup - Difference
        plot_difference_heatmap(axs[5, t], predictions_denorm[t, 1], Y_test_denorm[t, 1],
                                f"Pickup Difference - Timestep {t + 1}")

    # Add row labels
    row_labels = ["Dropoff Forecast", "Dropoff Actual", "Dropoff Difference",
                  "Pickup Forecast", "Pickup Actual", "Pickup Difference"]

    for i, label in enumerate(row_labels):
        axs[i, 0].set_ylabel(label, fontsize=12, rotation=90, labelpad=10)

    # Add a title
    fig.suptitle("ST-ResNet: Forecast vs Actual Heatmaps with Differences", fontsize=20)
    plt.savefig("stresnet_forecast_vs_actual_with_differences.png", dpi=300, bbox_inches='tight')

    # Create a separate figure focusing just on the differences
    fig_diff, axs_diff = plt.subplots(2, num_timesteps, figsize=(4 * num_timesteps, 8), constrained_layout=True)

    # If num_timesteps is 1, axs_diff will be 1D, so we need to reshape it
    if num_timesteps == 1:
        axs_diff = axs_diff.reshape(2, 1)

    for t in range(num_timesteps):
        # Dropoff difference
        plot_difference_heatmap(axs_diff[0, t], predictions_denorm[t, 0], Y_test_denorm[t, 0],
                                f"Dropoff Difference - Timestep {t + 1}")

        # Pickup difference
        plot_difference_heatmap(axs_diff[1, t], predictions_denorm[t, 1], Y_test_denorm[t, 1],
                                f"Pickup Difference - Timestep {t + 1}")

    # Add row labels
    axs_diff[0, 0].set_ylabel("Dropoff", fontsize=12)
    axs_diff[1, 0].set_ylabel("Pickup", fontsize=12)

    fig_diff.suptitle("ST-ResNet: Forecast Error Heatmaps (Forecast - Actual)", fontsize=20)
    plt.savefig("stresnet_forecast_error_heatmaps.png", dpi=300, bbox_inches='tight')

    # Calculate and print evaluation metrics
    print("Calculating evaluation metrics...")

    # MSE, RMSE, MAE for each flow type
    mse_dropoff = np.mean((predictions_denorm[:, 0] - Y_test_denorm[:, 0]) ** 2)
    rmse_dropoff = np.sqrt(mse_dropoff)
    mae_dropoff = np.mean(np.abs(predictions_denorm[:, 0] - Y_test_denorm[:, 0]))

    mse_pickup = np.mean((predictions_denorm[:, 1] - Y_test_denorm[:, 1]) ** 2)
    rmse_pickup = np.sqrt(mse_pickup)
    mae_pickup = np.mean(np.abs(predictions_denorm[:, 1] - Y_test_denorm[:, 1]))

    print(f"Dropoff - RMSE: {rmse_dropoff:.4f}, MAE: {mae_dropoff:.4f}")
    print(f"Pickup - RMSE: {rmse_pickup:.4f}, MAE: {mae_pickup:.4f}")

    # Calculate overall metrics
    mse_overall = np.mean((predictions_denorm - Y_test_denorm) ** 2)
    rmse_overall = np.sqrt(mse_overall)
    mae_overall = np.mean(np.abs(predictions_denorm - Y_test_denorm))

    print(f"Overall - RMSE: {rmse_overall:.4f}, MAE: {mae_overall:.4f}")


if __name__ == "__main__":
    main()