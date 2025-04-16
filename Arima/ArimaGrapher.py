import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from Arima import Arima
from sklearn.metrics import mean_absolute_error, mean_squared_error
from tqdm import tqdm

np.random.seed(42)

def grid_search(model_class, p_values, d_values, q_values, data, metric='MAE'):
    best_score = float('inf')
    best_params = (None, None, None, None, None)

    for p in tqdm(p_values, desc="Grid Searching"):
        for d in d_values:
            for q in q_values:
                try:
                    model = model_class(p=p, d=d, q=q)
                    model.fit(data)
                    metrics = model.evaluate()
                    score = metrics[metric]

                    if score < best_score:
                        best_score = score
                        best_params = (p, d, q, model.ar_params, model.ma_params)
                except Exception as e:
                    continue

    print(f"Best ARIMA(p,d,q): ({best_params[0]},{best_params[1]},{best_params[2]})")
    print(f"Best {metric}: {best_score:.4f}")
    return best_params, best_score

def main():
    # --- Load Data ---
    dropoff_data = np.load(r'data\dropoff_counts.npy')[:, 1, 0]
    pickup_data = np.load(r'data\pickup_counts.npy')[:, 1, 0]

    forecast_hours = 24
    train_st = 500
    train_end = test_st = train_st + 240
    test_end = test_st + forecast_hours
    test_period_length = test_end - test_st

    value_range = [0, 1, 2, 3, 4]

    # --- DROP-OFF MODEL ---
    print("Optimizing Dropoff model...")
    dropoff_train = dropoff_data[train_st:train_end]
    dropoff_test = dropoff_data[train_end:test_end]

    dropoff_model = Arima(p=1, d=0, q=1)  # Try d=0 first
    dropoff_model.ts = pd.Series(dropoff_train)
    dropoff_model.check_stationarity()  # Check stationarity
    dropoff_fitted = dropoff_model.fit(dropoff_train)
    dropoff_forecast = dropoff_model.forecast(test_period_length)

    # --- PICK-UP MODEL ---
    print("\nOptimizing Pickup model...")
    pickup_train = pickup_data[train_st:train_end]
    pickup_test = pickup_data[train_end:test_end]

    pickup_model = Arima(p=1, d=0, q=1)  # Try d=0 first
    pickup_model.ts = pd.Series(pickup_train)
    pickup_model.check_stationarity()  # Check stationarity
    pickup_fitted = pickup_model.fit(pickup_train)
    pickup_forecast = pickup_model.forecast(test_period_length)

    # --- Evaluation on Test Data ---
    dropoff_mae = mean_absolute_error(dropoff_test, dropoff_forecast)
    dropoff_rmse = np.sqrt(mean_squared_error(dropoff_test, dropoff_forecast))
    pickup_mae = mean_absolute_error(pickup_test, pickup_forecast)
    pickup_rmse = np.sqrt(mean_squared_error(pickup_test, pickup_forecast))

    # --- Plotting ---
    fig, axs = plt.subplots(3, 1, figsize=(12, 10), gridspec_kw={'height_ratios': [1, 1, 1]})

    # Dropoff plot
    axs[0].plot(range(train_st, train_end), dropoff_train, label='Training Data', color='blue')
    axs[0].plot(dropoff_model.fitted_values.index + train_st, dropoff_model.fitted_values, label='Model Fit', color='green', linestyle='--')
    axs[0].plot(range(test_st, test_end), dropoff_test, label='Test Data', color='orange')
    axs[0].plot(range(test_st, test_end), dropoff_forecast, label='Forecast', color='purple', linestyle='-.')
    axs[0].set_title(f'Dropoff Forecast - ARIMA(1,0,1)')
    axs[0].legend()
    axs[0].grid(True, alpha=0.3)

    # Pickup plot
    axs[1].plot(range(train_st, train_end), pickup_train, label='Training Data', color='blue')
    axs[1].plot(pickup_model.fitted_values.index + train_st, pickup_model.fitted_values, label='Model Fit', color='green', linestyle='--')
    axs[1].plot(range(test_st, test_end), pickup_test, label='Test Data', color='orange')
    axs[1].plot(range(test_st, test_end), pickup_forecast, label='Forecast', color='purple', linestyle='-.')
    axs[1].set_title(f'Pickup Forecast - ARIMA(1,0,1)')
    axs[1].legend()
    axs[1].grid(True, alpha=0.3)

    # New subplot: Zoomed-in view of the last 20 timesteps of training + test period
    axs[2].set_title('Zoomed View: Last 20 Training Steps + Test Period')
    axs[2].set_xlabel('Time')
    
    # Define the zoom window
    zoom_start = train_end - 20  # Last 20 timesteps of training
    zoom_end = test_end  # Including all test data
    
    axs[2].plot(range(zoom_start, train_end), dropoff_data[zoom_start:train_end], label='Dropoff Training', color='blue', alpha=0.6)
    axs[2].plot(range(test_st, test_end), dropoff_test, label='Dropoff Test', color='blue')
    axs[2].plot(range(test_st, test_end), dropoff_forecast, label='Dropoff Forecast', color='blue', linestyle='-.')
    axs[2].plot(range(zoom_start, train_end), pickup_data[zoom_start:train_end], label='Pickup Training', color='green', alpha=0.6)
    axs[2].plot(range(test_st, test_end), pickup_test, label='Pickup Test', color='green')
    axs[2].plot(range(test_st, test_end), pickup_forecast, label='Pickup Forecast', color='green', linestyle='-.')
    axs[2].axvline(x=train_end, color='red', linestyle='--', alpha=0.5)
    axs[2].annotate('End of Training', xy=(train_end, axs[2].get_ylim()[1]*0.9), xytext=(train_end+0.5, axs[2].get_ylim()[1]*0.9),
                   arrowprops=dict(facecolor='red', shrink=0.05, width=1, headwidth=6), fontsize=9, horizontalalignment='left')
    axs[2].grid(True, alpha=0.3)
    axs[2].legend(loc='upper left')

    plt.tight_layout()
    plt.subplots_adjust(top=0.95)
    plt.suptitle(f'ARIMA Forecasts (Dropoff & Pickup)', fontsize=14)
    plt.show()

if __name__ == "__main__":
    main()
