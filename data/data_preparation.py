import pyarrow.parquet as pq
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
import os


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


def clean_and_convert_to_heatmap():
    for i in range(1, 6):
        # Load *.parquet file (Jan 2010 - May 2010)
        filename = 'yellow_tripdata_2010-0' + str(i) + '.parquet'
        filename_pickup_save = str(i) + 'pickup_counts.npy'
        filename_dropoff_save = str(i) + 'dropoff_counts.npy'
        table = pq.read_table(filename)
        df = table.to_pandas()

        # ---------Filtering outliers start---------
        # Filtering outliers based on trip duration (< 300 mins), fare amount ($0-$300), and trip distance (0-100 miles)
        df['pickup_datetime'] = pd.to_datetime(df['pickup_datetime'])
        df['dropoff_datetime'] = pd.to_datetime(df['dropoff_datetime'])
        df['trip_duration'] = (df['dropoff_datetime'] - df['pickup_datetime']).dt.total_seconds() / 60
        df = df[(df['trip_distance'] > 0) & (df['trip_duration'] > 0) & (df['fare_amount'] > 0)
                & (df['trip_distance'] < 100) & (df['fare_amount'] < 300) & (df['trip_duration'] < 300)]

        # Filtering outliers based on relationship between trip distance, trip duration, and fare amount
        model = LinearRegression()
        X = df[['trip_distance', 'trip_duration']]
        y = df['fare_amount']
        model.fit(X, y)
        df['fare_pred'] = model.predict(X)
        df['residual'] = df['fare_amount'] - df['fare_pred']
        # The residue of real and predicted fare amount is assumed to follow the normal distribution
        # Outliers are filtered according to 3 std criterion
        threshold = 3 * df['residual'].std()
        df = df[np.abs(df['residual']) <= threshold]
        # ---------Filtering outliers end---------

        # bounding box for Manhattan district
        lat_min, lat_max = 40.7000, 40.8800
        lon_min, lon_max = -74.0200, -73.9100

        # divide the grids
        grid_size = 32
        lat_step = (lat_max - lat_min) / grid_size
        lon_step = (lon_max - lon_min) / grid_size

        # Convert pickup records to heat map
        pickup_df = df[(df['pickup_latitude'] >= lat_min) & (df['pickup_latitude'] <= lat_max) &
                       (df['pickup_longitude'] >= lon_min) & (df['pickup_longitude'] <= lon_max)].copy()
        pickup_df['grid_x'] = (((pickup_df['pickup_longitude'] - lon_min) / lon_step).astype(int)).clip(0,
                                                                                                        grid_size - 1)
        pickup_df['grid_y'] = (((pickup_df['pickup_latitude'] - lat_min) / lat_step).astype(int)).clip(0, grid_size - 1)
        pickup_df['hour'] = pickup_df['pickup_datetime'].dt.floor('h')

        start_hour = pickup_df['hour'].min()
        end_hour = pickup_df['hour'].max()
        all_hours = pd.date_range(start=start_hour, end=end_hour, freq='h')
        n = len(all_hours)

        grouped = pickup_df.groupby(['hour', 'grid_y', 'grid_x']).size()

        full_index = pd.MultiIndex.from_product(
            [all_hours, range(grid_size), range(grid_size)],
            names=['hour', 'grid_y', 'grid_x']
        )
        grouped = grouped.reindex(full_index, fill_value=0)

        pickup_counts = grouped.unstack(level=['grid_y', 'grid_x']).values

        pickup_counts = pickup_counts.reshape(n, grid_size, grid_size)
        # print("pickup_counts shape:", pickup_counts.shape)
        np.save(filename_pickup_save, pickup_counts)

        # Convert dropoff records to heat map
        dropoff_df = df[(df['dropoff_latitude'] >= lat_min) & (df['dropoff_latitude'] <= lat_max) &
                        (df['dropoff_longitude'] >= lon_min) & (df['dropoff_longitude'] <= lon_max)].copy()
        dropoff_df['grid_x'] = (((dropoff_df['dropoff_longitude'] - lon_min) / lon_step).astype(int)).clip(0,
                                                                                                           grid_size - 1)
        dropoff_df['grid_y'] = (((dropoff_df['dropoff_latitude'] - lat_min) / lat_step).astype(int)).clip(0,
                                                                                                          grid_size - 1)
        dropoff_df['hour'] = dropoff_df['dropoff_datetime'].dt.floor('h')

        if i != 5:
            start_hour = dropoff_df['hour'].min()
            end_hour = dropoff_df['hour'].max()
            all_hours = pd.date_range(start=start_hour, end=end_hour, freq='h')
            n = len(all_hours)
            print(start_hour)
            print(end_hour)

        grouped_dropoff = dropoff_df.groupby(['hour', 'grid_y', 'grid_x']).size()
        full_index_dropoff = pd.MultiIndex.from_product(
            [all_hours, range(grid_size), range(grid_size)],
            names=['hour', 'grid_y', 'grid_x']
        )
        grouped_dropoff = grouped_dropoff.reindex(full_index_dropoff, fill_value=0)
        dropoff_counts = grouped_dropoff.unstack(level=['grid_y', 'grid_x']).values
        dropoff_counts = dropoff_counts.reshape(n, grid_size, grid_size)
        # print("dropoff_counts shape:", dropoff_counts.shape)
        np.save(filename_dropoff_save, dropoff_counts)


def get_concatenate_data():
    clean_and_convert_to_heatmap()

    temp1 = []
    for i in range(1, 6):
        filename1 = str(i) + 'pickup_counts.npy'
        arr = np.load(filename1)
        temp1.append(arr)

    temp2 = []
    for i in range(1, 6):
        filename = str(i) + 'dropoff_counts.npy'
        arr = np.load(filename)
        temp2.append(arr)

    for i in range(1, len(temp2)):
        diff = temp2[i-1].shape[0] - temp1[i-1].shape[0]
        temp2[i][:diff] = temp2[i-1][-diff:] + temp2[i][:diff]
        temp2[i-1] = temp2[i-1][:temp1[i-1].shape[0]]
    result1 = np.concatenate(temp1, axis=0)
    np.save('pickup_counts.npy', result1)
    result2 = np.concatenate(temp2, axis=0)
    np.save('dropoff_counts.npy', result2)


def get_sequential_data(len_closeness, len_period, len_trend):
    if os.path.exists("data/pickup_counts.npy") is False or os.path.exists("data/dropoff_counts.npy") is False:
        get_concatenate_data()

    # stack pickup & dropoff records together
    pickup_data = np.load("data/pickup_counts.npy")
    dropoff_data = np.load("data/dropoff_counts.npy")
    dropoff_data = dropoff_data[0:pickup_data.shape[0]]
    data2d = np.stack([pickup_data, dropoff_data], axis=1)

    mmn = MinMaxNormalization()
    T = 24  # number of time intervals in one day
    train_st = 200
    train_ed = 2600
    test_st = 2600
    test_ed = 3400
    mmn.fit(data2d[train_st:train_ed])
    data_set = mmn.transform(data2d[:test_ed])
    Y_train = data_set[train_st:train_ed]
    Y_test = data_set[test_st:test_ed]

    # Construct train data
    Xc = []
    Xp = []
    Xt = []
    for i in range(train_st, train_ed):
        # closeness
        Xc_unit = np.vstack([data_set[i - j] for j in range(len_closeness, 0, -1)])
        Xc.append(Xc_unit)
        # period
        if len_period > 0:
            Xp_unit = np.vstack([data_set[i - T * k] for k in range(len_period, 0, -1)])
        else:
            Xp_unit = None
        Xp.append(Xp_unit)
        # trend
        if len_trend > 0:
            Xt_unit = np.vstack([data_set[i - T * 7 * k] for k in range(len_trend, 0, -1)])
        else:
            Xt_unit = None
        Xt.append(Xt_unit)

    Xc = np.array(Xc)
    Xp = np.array(Xp) if len_period > 0 else None
    Xt = np.array(Xt) if len_trend > 0 else None
    ext = np.zeros([train_ed - train_st, 28])
    X_train = [Xc, Xp, Xt, ext]

    # construct test data
    Xc = []
    Xp = []
    Xt = []
    for i in range(test_st, test_ed):
        Xc_unit = np.vstack([data_set[i - j] for j in range(len_closeness, 0, -1)])
        Xc.append(Xc_unit)
        if len_period > 0:
            Xp_unit = np.vstack([data_set[i - T * k] for k in range(len_period, 0, -1)])
        else:
            Xp_unit = None
        Xp.append(Xp_unit)
        if len_trend > 0:
            Xt_unit = np.vstack([data_set[i - T * 7 * k] for k in range(len_trend, 0, -1)])
        else:
            Xt_unit = None
        Xt.append(Xt_unit)

    Xc = np.array(Xc)
    Xp = np.array(Xp) if len_period > 0 else None
    Xt = np.array(Xt) if len_trend > 0 else None
    ext = np.zeros([test_ed - test_st, 28])
    X_test = [Xc, Xp, Xt, ext]

    ext_dim = 28

    return X_train, Y_train, X_test, Y_test, mmn, ext_dim
