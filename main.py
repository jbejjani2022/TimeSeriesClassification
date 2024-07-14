# dataloading, training, and evaluation pipeline
from data.data_loader import DataLoader
from models.random_forest import RFClassifier
from models.evaluate import evaluate


def main():
    DL = DataLoader()
    data = DL.load()
    x_train, y_train, x_test, y_test = DL.split(data)
    model = RFClassifier
    model.fit(x_train, y_train)
    y_predicted = model.predict(x_test)
    accuracy, score = evaluate(y_test, y_predicted)
    print(f"Accuracy: {accuracy:.2f}")
    print(f"Kappa: {score:.2f}")
    
    
if __name__ == '__main__':
    main()