import numpy as np

def confusion_matrix(y_actual, y_pred):
    """
        Generate a confusion matrix

        parameters:
            y_actual : actual target class
            y_pred : predicted target class
    """
    y_length = len(y_actual)
    n_output = len(np.unique(y_actual))
    result = np.zeros((n_output, n_output))

    # Generate confusion matrix
    for i in range (y_length):
       result[y_actual[i]][y_pred[i]] += 1 

    return np.int_(result)

def prediction_stats(confusion_matrix):
    """
        Generate true positive, true negative, false positive, false negative

        parameters:
            confusion_matrix: ndarray
    """
    tp = np.diag(confusion_matrix)
    fp = confusion_matrix.sum(axis=0) - np.diag(confusion_matrix)
    fn = confusion_matrix.sum(axis=1) - np.diag(confusion_matrix)
    tn = confusion_matrix.sum() - (tp + fp + fn)

    return tp, fp, fn, tn

def accuracy(y_actual, y_pred):
    matrix = confusion_matrix(y_actual, y_pred)
    tp, fp, fn, tn = prediction_stats(matrix)
    
    with np.errstate(divide='ignore'):
        result = (tp+fn)/(tp+fp+fn+tn)
        result[(tp+fp+fn+tn) == 0] = 0

    return result

def precision(y_actual, y_pred):
    matrix = confusion_matrix(y_actual, y_pred)
    tp, fp, fn, tn = prediction_stats(matrix)

    with np.errstate(divide='ignore', invalid='ignore'):
        result = (tp)/(tp+fp)
        result[(tp+fp) == 0] = 0

    return result

def recall(y_actual, y_pred):
    matrix = confusion_matrix(y_actual, y_pred)
    tp, fp, fn, tn = prediction_stats(matrix)

    with np.errstate(divide='ignore', invalid='ignore'):
        result = (tp)/(tp+fn)
        result[(tp+fn) == 0] = 0

    return result

def f1(y_actual, y_pred):
    _precision = precision(y_actual, y_pred)
    _recall = recall(y_actual, y_pred)

    with np.errstate(divide='ignore', invalid='ignore'):
        result = 2 * (_precision * _recall)/(_precision + _recall)
        result[(_precision + _recall) == 0] = 0

    return result