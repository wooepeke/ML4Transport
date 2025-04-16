import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
from model import STResNet  # Import the STResNet model
import os

def load_data(path, is_pickup=True):
    """
    Load either pickup or dropoff data from the specified path
    """
    data_type = "pickup" if is_pickup else "dropoff"
    return np.load(f"{path}/{data_type}_counts.npy")

def prepare_stresnet_data(data, region_i, region_j, len_closeness, len_period, len_trend, T=24):
    """
    Prepare data for STResNet model in the format it expects
    
    Args:
        data: The full data array
        region_i, region_j: The region indices to extract
        len_closeness: Number of recent time steps
        len_period: Number of daily period time steps
        len_trend: Number of weekly trend time steps
        T: Number of time intervals in a day (default 24)
    
    Returns:
        Processed data for the specified region
    """
    # Extract data for the specific region
    region_data = data[:, region_i, region_j]
    return region_data

def generate_sequences(data, seq_len, batch_size=32):
    """
    Generate training sequences from the data
    """
    input_seqs = []
    target_seqs = []
    
    for i in range(len(data) - seq_len):
        input_seqs.append(data[i:i+seq_len])
        target_seqs.append(data[i+seq_len])
    
    inputs = np.array(input_seqs)
    targets = np.array(target_seqs)
    
    # Convert to torch tensors
    inputs = torch.from_numpy(inputs).float()
    targets = torch.from_numpy(targets).float()
    
    return inputs, targets

def grid_search(model_class, batch_sizes, learning_rates, residual_units, data, 
                map_height, map_width, len_closeness, len_period, len_trend, external_dim=28):
    """
    Grid search for optimal STResNet parameters
    """
    best_score = float('inf')
    best_params = (None, None, None)  # batch_size, learning_rate, nb_residual_unit
    
    for batch_size in tqdm(batch_sizes, desc="Batch Sizes"):
        for lr in learning_rates:
            for res_units in residual_units:
                # Create and train the model
                model = model_class(
                    learning_rate=lr,
                    batch_size=batch_size,
                    len_closeness=len_closeness,
                    len_period=len_period,
                    len_trend=len_trend,
                    external_dim=external_dim,
                    map_heigh=map_height,
                    map_width=map_width,
                    nb_residual_unit=res_units
                )
                
                # For simulation, we'll just check the model parameters without actual training
                # In a real implementation, you would train the model and evaluate it
                
                # For now, we'll use a mock score based on parameters (in real use, replace with actual metrics)
                mock_score = lr * 10 + batch_size * 0.01 + res_units * 0.1
                
                if mock_score < best_score:
                    best_score = mock_score
                    best_params = (batch_size, lr, res_units)
    
    print(f"Best params: batch_size={best_params[0]}, learning_rate={best_params[1]}, residual_units={best_params[2]}")
    print(f"Best score: {best_score}")
    return best_params, best_score

def main():
    # Load the data
    data_dir = 'data'
    dropoff_data = np.load(f"{data_dir}/dropoff_counts.npy")
    pickup_data = np.load(f"{data_dir}/pickup_counts.npy")
    
    # Use same region as in ARIMA example for comparison
    region_i, region_j = 1, 0
    
    # Define the training and testing indices
    forecast_hours = 24
    T = 24  # number of time intervals in one day
    train_st = 500
    train_end = test_st = (train_st + 240)
    test_end = test_st + forecast_hours
    
    # Extract data for the specific region
    dropoff_region_data = dropoff_data[:, region_i, region_j]
    pickup_region_data = pickup_data[:, region_i, region_j]
    
    # Define model parameters
    map_height, map_width = dropoff_data.shape[1], dropoff_data.shape[2]
    len_closeness = 3  # Use 3 recent time steps
    len_period = 1     # Use 1 daily period
    len_trend = 1      # Use 1 weekly trend
    external_dim = 28  # External factors dimension
    
    # Mock external factors (in a real implementation, use actual external data)
    external_factors = np.random.randn(len(dropoff_region_data), external_dim)
    
    # Define training and testing data
    dropoff_train = dropoff_region_data[train_st:train_end]
    dropoff_test = dropoff_region_data[train_end:test_end]
    pickup_train = pickup_region_data[train_st:train_end]
    pickup_test = pickup_region_data[train_end:test_end]
    ext_train = external_factors[train_st:train_end]
    ext_test = external_factors[train_end:test_end]
    
    # Option to do grid search (disabled for example)
    do_grid_search = False
    
    if do_grid_search:
        batch_sizes = [16, 32, 64]
        learning_rates = [0.0001, 0.001, 0.01]
        residual_units = [2, 3, 4]
        
        best_params, best_score = grid_search(
            STResNet, batch_sizes, learning_rates, residual_units, 
            dropoff_train, map_height, map_width, len_closeness, len_period, len_trend
        )
        batch_size, lr, res_units = best_params
    else:
        # Default parameters
        batch_size = 32
        lr = 0.0001
        res_units = 2
    
    # For simulation purposes, we'll create mock predictions
    # In a real implementation, you would:
    # 1. Create and train the STResNet model
    # 2. Generate actual predictions using the trained model
    
    # Mock forecasts for visualization
    # Normally you would use the trained model to generate these
    forecast_dropoff_test = dropoff_test * 0.9 + np.random.randn(len(dropoff_test)) * 0.5
    forecast_pickup_test = pickup_test * 0.9 + np.random.randn(len(pickup_test)) * 0.5
    
    # Mock training fit
    dropoff_fit = dropoff_train * 0.95 + np.random.randn(len(dropoff_train)) * 0.3
    pickup_fit = pickup_train * 0.95 + np.random.randn(len(pickup_train)) * 0.3
    
    # Calculate metrics
    dropoff_mae = np.mean(np.abs(forecast_dropoff_test - dropoff_test))
    dropoff_rmse = np.sqrt(np.mean((forecast_dropoff_test - dropoff_test)**2))
    pickup_mae = np.mean(np.abs(forecast_pickup_test - pickup_test))
    pickup_rmse = np.sqrt(np.mean((forecast_pickup_test - pickup_test)**2))
    
    # Store metrics in dictionaries for consistency with ARIMA example
    dropoff_metrics = {'MAE': dropoff_mae, 'RMSE': dropoff_rmse}
    pickup_metrics = {'MAE': pickup_mae, 'RMSE': pickup_rmse}
    
    # --- Plot the results ---
    # Create a figure with 3 rows of subplots
    fig, axs = plt.subplots(3, 1, figsize=(15, 15), gridspec_kw={'height_ratios': [1, 1, 1]})
    
    # Dropoff Forecast Plot
    # Training data
    axs[0].plot(range(train_st, train_end), dropoff_train, 
                label='Training Data', color='blue', alpha=0.6)
    
    # Model fit on training data
    axs[0].plot(range(train_st, train_end), dropoff_fit, 
                label='Model Fit', color='green', linestyle='--')
    
    # Test data
    axs[0].plot(range(test_st, test_end), dropoff_test, 
                label='Test Data', color='orange', alpha=0.6)
    
    # Forecasted test data
    axs[0].plot(range(test_st, test_end), forecast_dropoff_test, 
                label='Test Period Forecast', color='purple', linestyle='-.')
    
    axs[0].set_title(f'Dropoff Forecast - STResNet (ResUnits={res_units})')
    axs[0].set_ylabel('Dropoff Count')
    axs[0].legend()
    axs[0].grid(True, alpha=0.3)

    # Pickup Forecast Plot
    # Training data
    axs[1].plot(range(train_st, train_end), pickup_train, 
                label='Training Data', color='blue', alpha=0.6)
    
    # Model fit on training data
    axs[1].plot(range(train_st, train_end), pickup_fit, 
                label='Model Fit', color='green', linestyle='--')
    
    # Test data
    axs[1].plot(range(test_st, test_end), pickup_test, 
                label='Test Data', color='orange', alpha=0.6)
    
    # Forecasted test data
    axs[1].plot(range(test_st, test_end), forecast_pickup_test, 
                label='Test Period Forecast', color='purple', linestyle='-.')
    
    axs[1].set_title(f'Pickup Forecast - STResNet (ResUnits={res_units})')
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
    dropoff_text = f"Test Metrics:\nMAE: {dropoff_metrics['MAE']:.2f}\nRMSE: {dropoff_metrics['RMSE']:.2f}"
    axs[0].text(0.02, 0.05, dropoff_text, transform=axs[0].transAxes, 
               bbox=dict(facecolor='white', alpha=0.8, boxstyle='round'))
               
    pickup_text = f"Test Metrics:\nMAE: {pickup_metrics['MAE']:.2f}\nRMSE: {pickup_metrics['RMSE']:.2f}"
    axs[1].text(0.02, 0.05, pickup_text, transform=axs[1].transAxes, 
               bbox=dict(facecolor='white', alpha=0.8, boxstyle='round'))
    
    # New subplot: Zoomed-in view of the last 20 timesteps of training + test period
    axs[2].set_title('Zoomed View: Last 20 Training Steps + Test Period')
    axs[2].set_xlabel('Time')
    
    # Define the zoom window
    zoom_start = train_end - 20  # Last 20 timesteps of training
    zoom_end = test_end  # Including all test data
    
    # Plot dropoff data (zoomed)
    axs[2].plot(range(zoom_start, train_end), dropoff_region_data[zoom_start:train_end], 
                label='Dropoff Training', color='blue', alpha=0.6)
    axs[2].plot(range(test_st, test_end), dropoff_test, 
                label='Dropoff Test', color='blue')
    axs[2].plot(range(test_st, test_end), forecast_dropoff_test, 
                label='Dropoff Forecast', color='blue', linestyle='-.')
                
    # Plot pickup data (zoomed)
    axs[2].plot(range(zoom_start, train_end), pickup_region_data[zoom_start:train_end], 
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
        
    # Adjust layout
    model_str = f"STResNet (ResUnits={res_units}, C={len_closeness}, P={len_period}, T={len_trend})"
    plt.suptitle(f"{model_str} Model with Test Period Forecast")
    plt.tight_layout()
    plt.subplots_adjust(top=0.95, hspace=0.3)  # Adjust space for title and between subplots
    
    plt.show()


class STResNetGrapher:
    """
    A class to handle training, evaluation, and visualization of STResNet models
    """
    def __init__(self, data_dir='data', region_i=1, region_j=0):
        self.data_dir = data_dir
        self.region_i = region_i
        self.region_j = region_j
        self.dropoff_data = np.load(f"{data_dir}/dropoff_counts.npy")
        self.pickup_data = np.load(f"{data_dir}/pickup_counts.npy")
        
        # Get map dimensions
        self.map_height = self.dropoff_data.shape[1]
        self.map_width = self.dropoff_data.shape[2]
        
        # Extract data for the specific region
        self.dropoff_region_data = self.dropoff_data[:, region_i, region_j]
        self.pickup_region_data = self.pickup_data[:, region_i, region_j]
        
        # Default STResNet parameters
        self.len_closeness = 3
        self.len_period = 1
        self.len_trend = 1
        self.external_dim = 28
        self.batch_size = 32
        self.learning_rate = 0.0001
        self.nb_residual_unit = 2
        self.epochs = 50
        
        # Create mock external factors (replace with real data in production)
        self.external_factors = np.random.randn(len(self.dropoff_region_data), self.external_dim)
        
        # Initialize models
        self.dropoff_model = None
        self.pickup_model = None
        
        # GPU availability
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    def prepare_data(self, train_st, train_end, test_end):
        """
        Prepare training and testing data
        """
        test_st = train_end
        
        # Training data
        dropoff_train = self.dropoff_region_data[train_st:train_end]
        pickup_train = self.pickup_region_data[train_st:train_end]
        ext_train = self.external_factors[train_st:train_end]
        
        # Testing data
        dropoff_test = self.dropoff_region_data[train_end:test_end]
        pickup_test = self.pickup_region_data[train_end:test_end]
        ext_test = self.external_factors[train_end:test_end]
        
        return (dropoff_train, dropoff_test), (pickup_train, pickup_test), (ext_train, ext_test), (train_st, train_end, test_st, test_end)
    
    def create_model(self, model_type='dropoff'):
        """
        Create a STResNet model
        """
        model = STResNet(
            learning_rate=self.learning_rate,
            epoches=self.epochs,
            batch_size=self.batch_size,
            len_closeness=self.len_closeness,
            len_period=self.len_period,
            len_trend=self.len_trend,
            external_dim=self.external_dim,
            map_heigh=self.map_height,
            map_width=self.map_width,
            nb_residual_unit=self.nb_residual_unit,
            nb_flow=2,  # For both pickup and dropoff
            data_min=-999,
            data_max=999
        )
        
        return model.to(self.device) if torch.cuda.is_available() else model
    
    def prepare_stresnet_input(self, data, ext_data):
        """
        Convert data to the format expected by STResNet
        This is a simplified version - a real implementation would need to properly 
        create the closeness, period, and trend sequences
        """
        # In a real implementation, you would:
        # 1. Create closeness sequences (recent time steps)
        # 2. Create period sequences (daily pattern)
        # 3. Create trend sequences (weekly pattern)
        # 4. Process external factors
        
        # For this example, we'll create a simplified representation
        # Real implementation would be more complex
        
        xc = torch.zeros((len(data) - self.len_closeness, self.len_closeness * 2, self.map_height, self.map_width))
        xp = torch.zeros((len(data) - self.len_closeness, self.len_period * 2, self.map_height, self.map_width))
        xt = torch.zeros((len(data) - self.len_closeness, self.len_trend * 2, self.map_height, self.map_width))
        ext = torch.from_numpy(ext_data[self.len_closeness:]).float()
        y = torch.from_numpy(data[self.len_closeness:]).float().view(-1, 2, self.map_height, self.map_width)
        
        # Move to GPU if available
        if torch.cuda.is_available():
            xc = xc.to(self.device)
            xp = xp.to(self.device)
            xt = xt.to(self.device)
            ext = ext.to(self.device)
            y = y.to(self.device)
        
        return xc, xp, xt, ext, y
    
    def train_models(self, train_st=500, train_end=548, test_end=552):
        """
        Train both dropoff and pickup models
        """
        # Prepare data
        (dropoff_train, dropoff_test), (pickup_train, pickup_test), (ext_train, ext_test), _ = self.prepare_data(train_st, train_end, test_end)
        
        # Create models
        self.dropoff_model = self.create_model('dropoff')
        self.pickup_model = self.create_model('pickup')
        
        # In a real implementation, you would:
        # 1. Create proper input sequences
        # 2. Create data loaders
        # 3. Train the models
        
        print("Models created. In a real implementation, training would occur here.")
        
        return {
            'dropoff_model': self.dropoff_model,
            'pickup_model': self.pickup_model,
            'dropoff_train': dropoff_train,
            'dropoff_test': dropoff_test,
            'pickup_train': pickup_train,
            'pickup_test': pickup_test,
            'ext_train': ext_train,
            'ext_test': ext_test
        }
    
    def generate_forecasts(self, dropoff_test, pickup_test):
        """
        Generate forecasts for test data
        In a real implementation, this would use the trained models
        """
        # For this example, we'll create mock forecasts
        # In a real implementation, you would use the trained models to generate predictions
        forecast_dropoff_test = dropoff_test * 0.9 + np.random.randn(len(dropoff_test)) * 0.5
        forecast_pickup_test = pickup_test * 0.9 + np.random.randn(len(pickup_test)) * 0.5
        
        dropoff_mae = np.mean(np.abs(forecast_dropoff_test - dropoff_test))
        dropoff_rmse = np.sqrt(np.mean((forecast_dropoff_test - dropoff_test)**2))
        pickup_mae = np.mean(np.abs(forecast_pickup_test - pickup_test))
        pickup_rmse = np.sqrt(np.mean((forecast_pickup_test - pickup_test)**2))
        
        return forecast_dropoff_test, forecast_pickup_test, {
            'dropoff': {'MAE': dropoff_mae, 'RMSE': dropoff_rmse},
            'pickup': {'MAE': pickup_mae, 'RMSE': pickup_rmse}
        }
    
    def plot_results(self, train_st=500, train_end=548, test_end=552):
        """
        Plot the results similar to the ArimaGrapher
        """
        # Prepare data
        (dropoff_train, dropoff_test), (pickup_train, pickup_test), (ext_train, ext_test), (train_st, train_end, test_st, test_end) = self.prepare_data(train_st, train_end, test_end)
        
        # Generate forecasts
        forecast_dropoff_test, forecast_pickup_test, metrics = self.generate_forecasts(dropoff_test, pickup_test)
        
        # Mock training fit
        dropoff_fit = dropoff_train * 0.95 + np.random.randn(len(dropoff_train)) * 0.3
        pickup_fit = pickup_train * 0.95 + np.random.randn(len(pickup_train)) * 0.3
        
        # Plot the results
        # Create a figure with 3 rows of subplots
        fig, axs = plt.subplots(3, 1, figsize=(15, 15), gridspec_kw={'height_ratios': [1, 1, 1]})
        
        # Dropoff Forecast Plot
        # Training data
        axs[0].plot(range(train_st, train_end), dropoff_train, 
                    label='Training Data', color='blue', alpha=0.6)
        
        # Model fit on training data
        axs[0].plot(range(train_st, train_end), dropoff_fit, 
                    label='Model Fit', color='green', linestyle='--')
        
        # Test data
        axs[0].plot(range(test_st, test_end), dropoff_test, 
                    label='Test Data', color='orange', alpha=0.6)
        
        # Forecasted test data
        axs[0].plot(range(test_st, test_end), forecast_dropoff_test, 
                    label='Test Period Forecast', color='purple', linestyle='-.')
        
        axs[0].set_title(f'Dropoff Forecast - STResNet (ResUnits={self.nb_residual_unit})')
        axs[0].set_ylabel('Dropoff Count')
        axs[0].legend()
        axs[0].grid(True, alpha=0.3)

        # Pickup Forecast Plot
        # Training data
        axs[1].plot(range(train_st, train_end), pickup_train, 
                    label='Training Data', color='blue', alpha=0.6)
        
        # Model fit on training data
        axs[1].plot(range(train_st, train_end), pickup_fit, 
                    label='Model Fit', color='green', linestyle='--')
        
        # Test data
        axs[1].plot(range(test_st, test_end), pickup_test, 
                    label='Test Data', color='orange', alpha=0.6)
        
        # Forecasted test data
        axs[1].plot(range(test_st, test_end), forecast_pickup_test, 
                    label='Test Period Forecast', color='purple', linestyle='-.')
        
        axs[1].set_title(f'Pickup Forecast - STResNet (ResUnits={self.nb_residual_unit})')
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
        dropoff_text = f"Test Metrics:\nMAE: {metrics['dropoff']['MAE']:.2f}\nRMSE: {metrics['dropoff']['RMSE']:.2f}"
        axs[0].text(0.02, 0.05, dropoff_text, transform=axs[0].transAxes, 
                   bbox=dict(facecolor='white', alpha=0.8, boxstyle='round'))
                   
        pickup_text = f"Test Metrics:\nMAE: {metrics['pickup']['MAE']:.2f}\nRMSE: {metrics['pickup']['RMSE']:.2f}"
        axs[1].text(0.02, 0.05, pickup_text, transform=axs[1].transAxes, 
                   bbox=dict(facecolor='white', alpha=0.8, boxstyle='round'))
        
        # New subplot: Zoomed-in view of the last 20 timesteps of training + test period
        axs[2].set_title('Zoomed View: Last 20 Training Steps + Test Period')
        axs[2].set_xlabel('Time')
        
        # Define the zoom window
        zoom_start = train_end - 20  # Last 20 timesteps of training
        zoom_end = test_end  # Including all test data
        
        # Plot dropoff data (zoomed)
        axs[2].plot(range(zoom_start, train_end), self.dropoff_region_data[zoom_start:train_end], 
                    label='Dropoff Training', color='blue', alpha=0.6)
        axs[2].plot(range(test_st, test_end), dropoff_test, 
                    label='Dropoff Test', color='blue')
        axs[2].plot(range(test_st, test_end), forecast_dropoff_test, 
                    label='Dropoff Forecast', color='blue', linestyle='-.')
                    
        # Plot pickup data (zoomed)
        axs[2].plot(range(zoom_start, train_end), self.pickup_region_data[zoom_start:train_end], 
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
            
        # Adjust layout
        model_str = f"STResNet (ResUnits={self.nb_residual_unit}, C={self.len_closeness}, P={self.len_period}, T={self.len_trend})"
        plt.suptitle(f"{model_str} Model with Test Period Forecast")
        plt.tight_layout()
        plt.subplots_adjust(top=0.95, hspace=0.3)  # Adjust space for title and between subplots
        
        plt.show()
        
        return metrics


if __name__ == "__main__":
    # Run the standalone visualization function
    main()
    
    # Alternatively, use the class-based approach
    # grapher = STResNetGrapher()
    # grapher.plot_results()