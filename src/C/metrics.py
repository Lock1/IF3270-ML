import numpy as np

def confusion_matrix(y_actual, y_pred):
    """
        Generate a confusion matrix

        parameters:
            y_actual : actual target class
            y_pred : predicted target class
    """
    n_output = len(np.unique(y_actual))

    result = np.zeros((n_output, n_output))

    for i in range (len(y_actual)):
       result[y_actual[i]][y_pred[i]] += 1 

    return np.int_(result)

def accuracy(y_actual, y_pred):
    return "Accuracy"

def precision(y_actual, y_pred):
    return "Precision"

def recall(y_actual, y_pred):
    return "Recall"

def f1(y_actual, y_pred):
    return "F1"