import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, cohen_kappa_score
import pandas_datareader as pdr
import matplotlib.pyplot as plt


def data_preprocessing(data, num_lags, train_test_split):
    # Prepare the data for training
    x = []
    y = []
    for i in range(len(data) - num_lags):
        x.append(data[i:i + num_lags])
        y.append(data[i+ num_lags])
    # Convert the data to numpy arrays
    x = np.array(x)
    y = np.array(y)
    # Split the data into training and testing sets
    split_index = int(train_test_split * len(x))
    x_train = x[:split_index]
    y_train = y[:split_index]
    x_test = x[split_index:]
    y_test = y[split_index:]
    
    return x_train, y_train, x_test, y_test 

start_date = '1960-01-01'
end_date   = '2023-09-01'

# Set the time index if it's not already set
data = (pdr.get_data_fred('SP500', start = start_date, end = end_date).dropna())

# Perform differencing to make the data stationary
data_diff = data.diff().dropna()

# Categorizing positive returns as 1 and negative returns as -1
data_diff = np.reshape(np.array(data_diff), (-1))
data_diff = np.where(data_diff > 0, 1, -1)

x_train, y_train, x_test, y_test = data_preprocessing(data_diff, 50, 0.80)

# Create and train a Random Forest Classifier
model = RandomForestClassifier(n_estimators = 100, random_state = 0)
model.fit(x_train, y_train)

# Make predictions
y_predicted = model.predict(x_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_predicted)
kappa = cohen_kappa_score(y_test, y_predicted)
print(f"Accuracy: {accuracy:.2f}")
print(f"Kappa: {kappa:.2f}")