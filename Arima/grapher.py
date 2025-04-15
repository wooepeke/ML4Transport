import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from Arima import Arima

np.random.seed(42)

def main():
    # Load the data
    dropoff_data = np.load(r'data\dropoff_counts.npy')[:, 1, 0]
    pickup_data = np.load(r'data\pickup_counts.npy')[:, 1, 0]

    # Define the training and testing indices
    forecast_hours = 4  # Maximum forecast period of 4 hours
    T = 24  # number of time intervals in one day
    train_st = 0
    train_end = test_st = (train_st + 240)
    test_end = test_st + forecast_hours

    # --- Dropoff Model ---
    print("Optimizing Dropoff model...")
    dropoff_train = dropoff_data[train_st:train_end]
    dropoff_test = dropoff_data[test_st:test_end]
    
    dropoff_model = Arima(dropoff_train)
    
    # Grid search for optimal parameters
    alpha_range = np.linspace(0.1, 0.9, 10)
    theta_range = np.linspace(0.1, 0.9, 10)
    
    best_params, best_score = dropoff_model.grid_search(alpha_range, theta_range, metric='MAE')
    print(f"Best Dropoff Params: alpha={best_params[0]:.2f}, theta={best_params[1]:.2f} with MAE={best_score:.4f}")

    # Set best parameters and run model
    dropoff_model.alpha, dropoff_model.theta = best_params
    dropoff_model.run_model()
    
    # Compute metrics on training data
    dropoff_metrics = dropoff_model.compute_metrics()
    print("Dropoff model metrics on training data:")
    for k, v in dropoff_metrics.items():
        print(f"  {k}: {v:.4f}")
        
    # Make forecast for test period
    test_period_length = test_end - test_st
    forecast_dropoff_test = dropoff_model.forecast_test_period(test_period_length)

    # --- Pickup Model ---
    print("\nOptimizing Pickup model...")
    pickup_train = pickup_data[train_st:train_end]
    pickup_test = pickup_data[test_st:test_end]
    
    pickup_model = Arima(pickup_train)
    
    best_params, best_score = pickup_model.grid_search(alpha_range, theta_range, metric='MAE')
    print(f"Best Pickup Params: alpha={best_params[0]:.2f}, theta={best_params[1]:.2f} with MAE={best_score:.4f}")

    # Set best parameters and run model
    pickup_model.alpha, pickup_model.theta = best_params
    pickup_model.run_model()
    
    # Compute metrics on training data
    pickup_metrics = pickup_model.compute_metrics()
    print("Pickup model metrics on training data:")
    for k, v in pickup_metrics.items():
        print(f"  {k}: {v:.4f}")
        
    # Make forecast for test period
    forecast_pickup_test = pickup_model.forecast_test_period(test_period_length)

    # --- Plot the results ---
    # Create a figure with 3 rows of subplots
    fig, axs = plt.subplots(3, 1, figsize=(15, 15), gridspec_kw={'height_ratios': [1, 1, 1]})
    
    # Dropoff Forecast Plot
    # Training data
    axs[0].plot(range(train_st, train_end), dropoff_train, 
                label='Training Data', color='blue', alpha=0.6)
    
    # Model fit on training data
    forecast_indices = dropoff_model.forecasted_full.index + train_st
    axs[0].plot(forecast_indices, dropoff_model.forecasted_full, 
                label='Model Fit', color='green', linestyle='--')
    
    # Test data
    axs[0].plot(range(test_st, test_end), dropoff_test, 
                label='Test Data', color='orange', alpha=0.6)
    
    # Forecasted test data
    axs[0].plot(range(test_st, test_end), forecast_dropoff_test, 
                label='Test Period Forecast', color='purple', linestyle='-.')
    
    axs[0].set_title('Dropoff Forecast')
    axs[0].set_ylabel('Dropoff Count')
    axs[0].legend()
    axs[0].grid(True, alpha=0.3)

    # Pickup Forecast Plot
    # Training data
    axs[1].plot(range(train_st, train_end), pickup_train, 
                label='Training Data', color='blue', alpha=0.6)
    
    # Model fit on training data
    forecast_indices = pickup_model.forecasted_full.index + train_st
    axs[1].plot(forecast_indices, pickup_model.forecasted_full, 
                label='Model Fit', color='green', linestyle='--')
    
    # Test data
    axs[1].plot(range(test_st, test_end), pickup_test, 
                label='Test Data', color='orange', alpha=0.6)
    
    # Forecasted test data
    axs[1].plot(range(test_st, test_end), forecast_pickup_test, 
                label='Test Period Forecast', color='purple', linestyle='-.')
    
    axs[1].set_title('Pickup Forecast')
    axs[1].set_ylabel('Pickup Count')
    axs[1].legend()
    axs[1].grid(True, alpha=0.3)

    # Add vertical line to mark where train data ends for main plots
    for ax in axs[:2]:
        ax.axvline(x=train_end, color='gray', linestyle='--', alpha=0.5)
        ax.annotate('End of Training', xy=(train_end, ax.get_ylim()[1]*0.85),
                   xytext=(train_end+2, ax.get_ylim()[1]*0.85),
                   arrowprops=dict(facecolor='gray', shrink=0.05, width=1, headwidth=6),
                   fontsize=9, horizontalalignment='left')

    # Add text box with metrics for main plots
    dropoff_text = f"Test Metrics:\nMAE: {dropoff_model.evaluate_test_data(dropoff_test)['MAE']:.2f}\nRMSE: {dropoff_model.evaluate_test_data(dropoff_test)['RMSE']:.2f}"
    axs[0].text(0.02, 0.05, dropoff_text, transform=axs[0].transAxes, 
               bbox=dict(facecolor='white', alpha=0.8, boxstyle='round'))
               
    pickup_text = f"Test Metrics:\nMAE: {pickup_model.evaluate_test_data(pickup_test)['MAE']:.2f}\nRMSE: {pickup_model.evaluate_test_data(pickup_test)['RMSE']:.2f}"
    axs[1].text(0.02, 0.05, pickup_text, transform=axs[1].transAxes, 
               bbox=dict(facecolor='white', alpha=0.8, boxstyle='round'))
    
    # New subplot: Zoomed-in view of the last 20 timesteps of training + test period
    axs[2].set_title('Zoomed View: Last 20 Training Steps + Test Period')
    axs[2].set_xlabel('Time')
    
    # Define the zoom window
    zoom_start = train_end - 20  # Last 20 timesteps of training
    zoom_end = test_end  # Including all test data
    
    # Plot dropoff data (zoomed)
    axs[2].plot(range(zoom_start, train_end), dropoff_data[zoom_start:train_end], 
                label='Dropoff Training', color='blue', alpha=0.6)
    axs[2].plot(range(test_st, test_end), dropoff_test, 
                label='Dropoff Test', color='blue')
    axs[2].plot(range(test_st, test_end), forecast_dropoff_test, 
                label='Dropoff Forecast', color='blue', linestyle='-.')
                
    # Plot pickup data (zoomed)
    axs[2].plot(range(zoom_start, train_end), pickup_data[zoom_start:train_end], 
                label='Pickup Training', color='green', alpha=0.6)
    axs[2].plot(range(test_st, test_end), pickup_test, 
                label='Pickup Test', color='green')
    axs[2].plot(range(test_st, test_end), forecast_pickup_test, 
                label='Pickup Forecast', color='green', linestyle='-.')
    
    # Add vertical line for end of training in zoomed plot
    axs[2].axvline(x=train_end, color='red', linestyle='--', alpha=0.5)
    axs[2].annotate('End of Training', xy=(train_end, axs[2].get_ylim()[1]*0.9),
                   xytext=(train_end+0.5, axs[2].get_ylim()[1]*0.9),
                   arrowprops=dict(facecolor='red', shrink=0.05, width=1, headwidth=6),
                   fontsize=9, horizontalalignment='left')
    
    axs[2].grid(True, alpha=0.3)
    axs[2].legend(loc='upper left')
    
    # Calculate and display test data metrics
    print("\nEvaluating dropoff model on test data:")
    dropoff_test_metrics = dropoff_model.evaluate_test_data(dropoff_test)
    for k, v in dropoff_test_metrics.items():
        print(f"  {k}: {v:.4f}")
        
    print("\nEvaluating pickup model on test data:")
    pickup_test_metrics = pickup_model.evaluate_test_data(pickup_test)
    for k, v in pickup_test_metrics.items():
        print(f"  {k}: {v:.4f}")
    
    # Adjust layout
    plt.suptitle("ARIMA(1,1,1) Model with Test Period Forecast")
    plt.tight_layout()
    plt.subplots_adjust(top=0.95, hspace=0.3)  # Adjust space for title and between subplots
    
    plt.show()

if __name__ == "__main__":
    main()
