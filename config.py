# data is from https://www.kaggle.com/datasets/henryhan117/sp-500-historical-data
DATA_PATH = 'data/SP500.csv'
# the symbol representing the financial data series to be retrieved from FRED
INDICATOR = 'SP500'
# start and end dates for indicator data
START_DATE = '1960-01-01'
END_DATE = '2023-09-01'

# dataloading parameters
NUM_LAGS = 50
SPLIT = 0.7

# random forest parameters
N_ESTIMATORS = 80
RANDOM_STATE = 0