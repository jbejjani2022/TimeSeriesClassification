import numpy as np
import pandas_datareader as pdr
import matplotlib.pyplot as plt
from config import INDICATOR, START_DATE, END_DATE, NUM_LAGS, SPLIT


class DataLoader:
    def __init__(self, indicator=INDICATOR, start_date=START_DATE,
                 end_date=END_DATE,num_lags=NUM_LAGS, train_test_split=SPLIT):
        self.indicator = indicator
        self.start_date = start_date
        self.end_date = end_date
        self.num_lags = num_lags
        self.train_test_split = train_test_split


    def split(self, data):
        # preprocess data for training
        x, y = [], []
        for i in range(len(data) - self.num_lags):
            x.append(data[i : i + self.num_lags])
            y.append(data[i + self.num_lags])
        x, y = np.array(x), np.array(y)
        # split data into train and test sets
        split_index = int(self.train_test_split * len(x))
        x_train, y_train = x[:split_index], y[:split_index]
        x_test, y_test = x[split_index:], y[split_index:]

        return x_train, y_train, x_test, y_test


    def load(self):
        # fetch SP500 time series data from the Federal Reserve Economic Data (FRED) service
        data = pdr.get_data_fred(self.indicator, start = self.start_date, end = self.end_date)
        data = data.dropna()
        # take the difference between each closing value to make the series stationary
        stationary_data = data.diff().dropna()
        # flatten the data array
        stationary_data = np.array(stationary_data).flatten()
        # classify positive returns as 1 and negative returns as -1, i.e. market 'up' and market 'down'
        ups_downs = np.where(stationary_data > 0, 1, -1)

        return ups_downs
    