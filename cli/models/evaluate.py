from sklearn.metrics import accuracy_score, cohen_kappa_score

def evaluate(labels, predicted):
    accuracy = accuracy_score(labels, predicted)
    kappa = cohen_kappa_score(labels, predicted)
    return accuracy, kappa
