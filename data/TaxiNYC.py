import numpy as np
import h5py

np.random.seed(1337)  # for reproducibility


class MinMaxNormalization(object):
    """
        MinMax Normalization --> [-1, 1]
        x = (x - min) / (max - min).
        x = x * 2 - 1
    """

    def __init__(self):
        pass

    def fit(self, X):
        self._min = X.min()
        self._max = X.max()
        print("min:", self._min, "max:", self._max)

    def transform(self, X):
        X = 1. * (X - self._min) / (self._max - self._min)
        X = X * 2. - 1.
        return X

    def fit_transform(self, X):
        self.fit(X)
        return self.transform(X)

    def inverse_transform(self, X):
        X = (X + 1.) / 2.
        X = 1. * X * (self._max - self._min) + self._min
        return X


def f():
    # fl = h5py.File('BJ14_M32x32_T30_InOut.h5')
    # pickup_data = fl['data']
    pickup_data = np.load("data/p_map.npy")
    dropoff_data = np.load("data/d_map.npy")
    data2d = np.stack([pickup_data, dropoff_data], axis=1)
    mmn = MinMaxNormalization()
    T = 24
    train_st = 500
    train_ed = 3700
    test_st = 3700
    test_ed = 4500
    mmn.fit(data2d[train_st:train_ed])
    data_set = mmn.transform(data2d[:test_ed])
    Y_train = data_set[train_st:train_ed]
    Y_test = data_set[test_st:test_ed]
    Xc = []
    Xp = []
    Xt = []
    for i in range(train_st, train_ed):
        # Xc_unit = np.array([data_set[i-3], data_set[i-2], data_set[i-1]])
        Xc_unit = np.vstack((data_set[i - 3], data_set[i - 2], data_set[i - 1]))
        Xc.append(Xc_unit)
        Xp_unit = data_set[i-T]
        # Xp.append([Xp_unit])
        Xp.append(Xp_unit)
        Xt_unit = data_set[i-T*7]
        # Xt.append([Xt_unit])
        Xt.append(Xt_unit)
    Xc = np.array(Xc)
    Xp = np.array(Xp)
    Xt = np.array(Xt)
    ext = np.zeros([train_ed-train_st, 28])
    X_train = [Xc, Xp, Xt, ext]

    Xc = []
    Xp = []
    Xt = []
    for i in range(test_st, test_ed):
        # Xc_unit = np.array([data_set[i-3], data_set[i-2], data_set[i-1]])
        Xc_unit = np.vstack((data_set[i - 3], data_set[i - 2], data_set[i - 1]))
        Xc.append(Xc_unit)
        Xp_unit = data_set[i - T]
        # Xp.append([Xp_unit])
        Xp.append(Xp_unit)
        Xt_unit = data_set[i - T * 7]
        # Xt.append([Xt_unit])
        Xt.append(Xt_unit)
    Xc = np.array(Xc)
    Xp = np.array(Xp)
    Xt = np.array(Xt)
    ext = np.zeros([test_ed - test_st, 28])
    X_test = [Xc, Xp, Xt, ext]

    ext_dim = 28

    return X_train, Y_train, X_test, Y_test, mmn, ext_dim




f()

