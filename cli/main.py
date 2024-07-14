import argparse
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error, mean_absolute_percentage_error
from data.dataloader import DataLoader
from models.LSTM import LSTMForecaster
from models.random_forest import RFClassifier
from models.evaluate import evaluate


# dataloading, training, and evaluation pipeline
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', choices=['LSTM', 'RF'], required=True, help='select model type: RF, LSTM')
    parser.add_argument('--task', choices=['c', 'r'], required=True, help='select task: classification, regression')
    parser.add_argument('--scale', action='store_true', help='apply minmaxscaler to dataset')
    args = parser.parse_args()
    
    DL = DataLoader(task=args.task)
    data = DL.load(scale=args.scale)
    x_train, y_train, x_test, y_test = DL.split(data)

    if args.model == 'RF':
        if args.task == 'r':
            raise ValueError("Invalid argument: please select classification task for RF")
        model = RFClassifier
        model.fit(x_train, y_train)
        y_predicted = model.predict(x_test)
        accuracy, score = evaluate(y_test, y_predicted)
        print(f"Accuracy: {accuracy:.2f}")
        print(f"Kappa: {score:.2f}")
    elif args.model == 'LSTM':
        model = LSTMForecaster()
        x_train = x_train.reshape((x_train.shape[0], x_train.shape[1], 1))
        x_test = x_test.reshape((x_test.shape[0], x_test.shape[1], 1))
        validation_set = (x_test, y_test)
        history = model.train(x_train, y_train, validation_set)
        # plt.plot(history.history['loss'])
        # plt.plot(history.history['val_loss'])
        # plt.show()
        y_predicted = model.predict(x_test)
        result_df = model.evaluate(y_predicted, y_test, scale=args.scale)
        print(result_df.head())
        print(result_df.shape)
        test = result_df['act']
        forecast = result_df['pred']
        if args.task == 'r':
            print('LSTM model MSE:', mean_squared_error(test, forecast))
            print('LSTM model MAE:', mean_absolute_error(test, forecast))
            print('LSTM model MAPE:', mean_absolute_percentage_error(test, forecast))
        elif args.task == 'c':
            accuracy, score = evaluate(y_test, y_predicted)
            print(f"Accuracy: {accuracy:.2f}")
            print(f"Kappa: {score:.2f}")

    
if __name__ == '__main__':
    main()
