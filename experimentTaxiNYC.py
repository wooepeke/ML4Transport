import sys
import logging
from torch.utils.data import TensorDataset, DataLoader
from data.data_preparation import get_sequential_data
from model.STResNet import STResNet
import torch
import numpy as np

gpu_available = torch.cuda.is_available()
if gpu_available:
    gpu = torch.device("cuda:0")

if __name__ == '__main__':
    logging.basicConfig(level=logging.DEBUG, format=' %(levelname)s - %(message)s')
    nb_epoch = 100    # number of epoch at training stage
    batch_size = 32    # batch size
    T = 24    # number of time intervals in one day
    len_closeness = 3    # length of closeness dependent sequence
    len_period = 1    # length of period dependent sequence
    len_trend = 1    # length of trend dependent sequence
    map_height, map_width = 32, 32    # grid size
    nb_flow = 2    # pickup counts & dropoff counts
    lr = 0.0002    # learning rate
    nb_residual_unit = 4    # number of residual unit

    X_train, Y_train, X_test, Y_test, mmn, external_dim = get_sequential_data(len_closeness, len_period, len_trend)

    for i in range(4):
        print(np.min(X_train[i]))
        print(np.max(X_train[i]))
    print(np.min(Y_train))
    print(np.max(Y_train))

    train_set = TensorDataset(torch.Tensor(X_train[0]), torch.Tensor(X_train[1]), torch.Tensor(X_train[2]),
                              torch.Tensor(X_train[3]), torch.Tensor(Y_train))
    # corresponding to closeness, period, trend, external, y
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)

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
        model = model.to(gpu)
    X_test_torch = [torch.Tensor(x) for x in X_test]
    model.train_model(train_loader, X_test_torch, torch.Tensor(Y_test))
    model.load_model("best")
    model.evaluate(X_test_torch, torch.Tensor(Y_test))
