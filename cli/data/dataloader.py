import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import pandas_datareader as pdr
from config import INDICATOR, START_DATE, END_DATE, NUM_LAGS, SPLIT, DATA_PATH


class DataLoader:
    def __init__(self, task='c', indicator=INDICATOR, start_date=START_DATE,
                 end_date=END_DATE,num_lags=NUM_LAGS, train_test_split=SPLIT):
        self.task = task
        self.indicator = indicator
        self.start_date = start_date
        self.end_date = end_date
        self.num_lags = num_lags
        self.train_test_split = train_test_split


    def split(self, df):
        # preprocess data for training
        x, y = [], []
        for i in range(len(df) - self.num_lags):
            x.append(df[i : i + self.num_lags, 0])
            if self.task == 'c': 
                y.append(1 if df[i + self.num_lags, 0] > 0 else -1)
            elif self.task == 'r':
                y.append(df[i + self.num_lags, 0])
            else:
                raise ValueError("Invalid argument: task must be classification or regression")
           
        x, y = np.array(x), np.array(y)
        # split data into train and test sets
        split_index = int(self.train_test_split * len(x))
        x_train, y_train = x[:split_index], y[:split_index]
        x_test, y_test = x[split_index:], y[split_index:]

        return x_train, y_train, x_test, y_test


    def load(self, compress=False, scale=True):
        # fetch SP500 time series data from the Federal Reserve Economic Data (FRED) service
        # data = pdr.get_data_fred(self.indicator, start = self.start_date, end = self.end_date)
        df = pd.read_csv(DATA_PATH)
        closing_prices = df['Close']
        
        if scale:
            scaler = MinMaxScaler()
            scaled_data = closing_prices.values.reshape(closing_prices.shape[0], 1)
            scaled_data = scaler.fit_transform(scaled_data)
            closing_prices = scaled_data
        if self.task == 'r':
            return closing_prices
        elif self.task == 'c':
            # take the difference between each closing value to make the series stationary
            stationary_data = closing_prices.diff().dropna()
            # flatten the data array
            stationary_data = np.array(stationary_data).flatten()
            if compress:
                # classify positive returns as 1 and negative returns as -1, i.e. market 'up' and market 'down'
                stationary_data = np.where(stationary_data > 0, 1, -1)
            return stationary_data
        else:
            raise ValueError("Invalid argument: task must be classification or regression")
            

if __name__ == '__main__':
    DL = DataLoader()
    data = DL.load()
    x_train, y_train, x_test, y_test = DL.split(data)