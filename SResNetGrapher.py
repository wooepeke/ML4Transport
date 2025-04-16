import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
from model import STResNet  # Import the STResNet model
import os

def load_data(path, is_pickup=True):
    data_type = "pickup" if is_pickup else "dropoff"
    return np.load(f"{path}/{data_type}_counts.npy")

def prepare_stresnet_data(data, region_i, region_j, len_closeness, len_period, len_trend, T=24):
    # Extract data for the specific region
    region_data = data[:, region_i, region_j]
    return region_data

def generate_sequences(data, seq_len, batch_size=32):
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
    
    forecast_hours = 24
    training_length = 2400
    train_st = 200
    train_end = test_st = training_length + train_st
    test_end = test_st + forecast_hours
    test_period_length = test_end - test_st
    
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
    
    # Mock forecasts for visualization
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
    fig, axs = plt.subplots(3, 1, figsize=(15, 15), gridspec_kw={'height_ratios': [1, 1, 1]})
    
    # --- Plotting in SARIMA style ---
    fig, axs = plt.subplots(3, 1, figsize=(10, 10), gridspec_kw={'height_ratios': [1, 1, 1]})
    
    zoom_len = 264

    # --- Dropoff Plot ---
    axs[0].plot(range(train_st, train_end)[-zoom_len:], dropoff_train[-zoom_len:], label='Training Data', color='blue')
    axs[0].plot(range(train_st, train_end)[-zoom_len:], dropoff_fit[-zoom_len:], label='Model Fit', color='green', linestyle='--')
    axs[0].plot(range(test_st, test_end)[-zoom_len:], dropoff_test[-zoom_len:], label='Test Data', color='orange')
    axs[0].plot(range(test_st, test_end)[-zoom_len:], forecast_dropoff_test[-zoom_len:], label='Forecast', color='purple', linestyle='-.')
    axs[0].set_title(f'Dropoff Forecast {zoom_len}/{training_length + test_period_length} timesteps - STResNet')
    axs[0].legend(loc='upper left')
    axs[0].grid(True, alpha=0.3)

    # --- Pickup Plot ---
    axs[1].plot(range(train_st, train_end)[-zoom_len:], pickup_train[-zoom_len:], label='Training Data', color='blue')
    axs[1].plot(range(train_st, train_end)[-zoom_len:], pickup_fit[-zoom_len:], label='Model Fit', color='green', linestyle='--')
    axs[1].plot(range(test_st, test_end)[-zoom_len:], pickup_test[-zoom_len:], label='Test Data', color='orange')
    axs[1].plot(range(test_st, test_end)[-zoom_len:], forecast_pickup_test[-zoom_len:], label='Forecast', color='purple', linestyle='-.')
    axs[1].set_title(f'Pickup Forecast {zoom_len}/{training_length + test_period_length} timesteps - STResNet')
    axs[1].legend(loc='upper left')
    axs[1].grid(True, alpha=0.3)

    # --- Zoomed-in subplot ---
    axs[2].set_title('Zoomed View: Last 20 Training Steps + Test Period')
    axs[2].set_xlabel('Time')
    
    zoom_start = train_end - 20

    axs[2].plot(range(zoom_start, train_end), dropoff_region_data[zoom_start:train_end], label='Dropoff Training', color='blue', alpha=0.6)
    axs[2].plot(range(test_st, test_end), dropoff_test, label='Dropoff Test', color='blue', alpha=0.8)
    axs[2].plot(range(test_st, test_end), forecast_dropoff_test, label='Dropoff Forecast', color='blue', linestyle='-.', alpha=0.8)
    axs[2].plot(range(zoom_start, train_end), pickup_region_data[zoom_start:train_end], label='Pickup Training', color='green', alpha=0.6)
    axs[2].plot(range(test_st, test_end), pickup_test, label='Pickup Test', color='green', alpha=0.8)
    axs[2].plot(range(test_st, test_end), forecast_pickup_test, label='Pickup Forecast', color='green', linestyle='-.', alpha=0.8)

    axs[2].fill_between(range(test_st, test_end), dropoff_test, forecast_dropoff_test, color='blue', alpha=0.2, label='Dropoff Error')
    axs[2].fill_between(range(test_st, test_end), pickup_test, forecast_pickup_test, color='green', alpha=0.2, label='Pickup Error')

    axs[2].axvline(x=train_end, color='red', linestyle='--', alpha=0.5)
    axs[2].annotate('End of Training', xy=(train_end, axs[2].get_ylim()[1]*0.9), xytext=(train_end+0.5, axs[2].get_ylim()[1]*0.9),
                   arrowprops=dict(facecolor='red', shrink=0.05, width=1, headwidth=6), fontsize=9, horizontalalignment='left')
    axs[2].grid(True, alpha=0.3)
    axs[2].legend(loc='upper left')

    plt.tight_layout()
    plt.subplots_adjust(top=0.95)
    plt.show()



class STResNetGrapher:
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
    

if __name__ == "__main__":
    # Run the standalone visualization function
    main()
    