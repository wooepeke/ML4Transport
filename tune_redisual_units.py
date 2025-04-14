import torch
from torch.utils.data import TensorDataset, DataLoader
import numpy as np
import gc
import random
import logging
from tqdm import tqdm
from data.data_preparation import get_sequential_data
from model.STResNet import STResNet
import matplotlib.pyplot as plt

# Set random seed for reproducibility
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_seed(42)

# Enable logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s - %(message)s')

# Check for GPU
gpu_available = torch.cuda.is_available()
device = torch.device("cuda:0" if gpu_available else "cpu")
logging.info(f"Using device: {device}")

# Constants
nb_epoch = 20  # Reduce for faster tuning (increase later for final training)
batch_size = 32
T = 24
len_closeness = 3
len_period = 1
len_trend = 1
map_height, map_width = 32, 32
nb_flow = 2
lr = 0.0002
residual_unit_range = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14]

# Load data
logging.info("Loading data...")
X_train, Y_train, X_test, Y_test, mmn, external_dim = get_sequential_data(len_closeness, len_period, len_trend)

train_set = TensorDataset(
    torch.Tensor(X_train[0]),
    torch.Tensor(X_train[1]),
    torch.Tensor(X_train[2]),
    torch.Tensor(X_train[3]),
    torch.Tensor(Y_train)
)
train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)

X_test_torch = [torch.Tensor(x).to(device) for x in X_test]
Y_test_torch = torch.Tensor(Y_test).to(device)

results = {}

# Hyperparameter tuning loop with tqdm
for nb_residual_unit in tqdm(residual_unit_range, desc="Tuning Residual Units"):
    logging.info(f"\nTraining with nb_residual_unit = {nb_residual_unit}")

    model = STResNet(
        learning_rate=lr,
        epoches=nb_epoch,
        batch_size=batch_size,
        len_closeness=len_closeness,
        len_period=len_period,
        len_trend=len_trend,
        external_dim=external_dim,
        map_heigh=map_height,
        map_width=map_width,
        nb_flow=nb_flow,
        nb_residual_unit=nb_residual_unit,
        data_min=mmn._min,
        data_max=mmn._max
    )

    if gpu_available:
        model = model.to(device)

    model.train_model(train_loader, X_test_torch, Y_test_torch)
    model.load_model("best")
    result = model.evaluate(X_test_torch, Y_test_torch)

    results[nb_residual_unit] = result

    # Clean up to free memory
    del model
    torch.cuda.empty_cache()
    gc.collect()

# Outputing results
print("\nHyperparameter tuning results:")
for nb_residual_unit, result in results.items():
    print(f"Residual Units: {nb_residual_unit} -> Eval: {result}")

# Extracting RMSE and MAE from the results
res_units = list(results.keys())
rmse_values = [results[k][0].item() for k in res_units]
mae_values = [results[k][1].item() for k in res_units]

# Plotting
plt.figure(figsize=(10, 5))
plt.plot(res_units, rmse_values, marker='o', label='RMSE')
plt.plot(res_units, mae_values, marker='s', label='MAE')
plt.xlabel('Number of Residual Units')
plt.ylabel('Error')
plt.title('RMSE and MAE vs. Number of Residual Units')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()