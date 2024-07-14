from tensorflow import keras
from tensorflow.keras.layers import Dense, LSTM, Dropout, Flatten
from tensorflow.keras import Sequential
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from config import DATA_PATH, NUM_LAGS, SPLIT

class LSTMForecaster:
    def __init__(self, units=64, optimizer='adam', loss='mse'):
        self.model = Sequential()
        # self.model.add(LSTM(128,return_sequences=True))
        self.model.add(LSTM(units)) # model.add(LSTM(64,return_sequences=False))
        # self.model.add(Dense(25))
        self.model.add(Dense(1))
        self.model.compile(optimizer=optimizer, loss=loss)
        
    def summarize(self):
        print(self.model.summary())
        
    def train(self, x_train, y_train, validation_data, epochs=25, batch_size=8):
        # do not shuffle data during training because order matters in time series
        history = self.model.fit(x_train, y_train, epochs=epochs, batch_size=batch_size, 
                       validation_data=validation_data, shuffle=False)
        return history
    
    def predict(self, x):
        return self.model.predict(x)
    
    def evaluate(self, y_predicted, y_true, scale):
        if scale:
            scaler = MinMaxScaler()
            pred = [scaler.inverse_transform(y) for y in y_predicted]
            act = [scaler.inverse_transform(y) for y in y_true]
        else:
            pred = y_predicted
            act = y_true
        result_df = pd.DataFrame({'pred':list(np.reshape(pred, (-1))),'act':list(np.reshape(act, (-1)))})
        return result_df
        
        