import numpy as np

def _get_weights(y_actual):
    _, counts = np.unique(y_actual, return_counts=True)

    return counts

def _precision_array(tp, fp):
    with np.errstate(divide='ignore', invalid='ignore'):
        result = (tp)/(tp+fp)
        result[(tp+fp) == 0] = 0

    return result

def _recall_array(tp, fn):
    with np.errstate(divide='ignore', invalid='ignore'):
        result = (tp)/(tp+fn)
        result[(tp+fn) == 0] = 0

    return result

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
    sample_weights = _get_weights(y_actual)
    matrix = confusion_matrix(y_actual, y_pred)
    tp, fp, fn, tn = prediction_stats(matrix)
    
    with np.errstate(divide='ignore'):
        result = (tp+fn)/(tp+fp+fn+tn)
        result[(tp+fp+fn+tn) == 0] = 0

    return np.average(result, weights=sample_weights)

def precision(y_actual, y_pred):
    sample_weights = _get_weights(y_actual)
    matrix = confusion_matrix(y_actual, y_pred)
    tp, fp, fn, tn = prediction_stats(matrix)

    result = _precision_array(tp, fp)

    return np.average(result, weights=sample_weights)

def recall(y_actual, y_pred):
    sample_weights = _get_weights(y_actual)
    matrix = confusion_matrix(y_actual, y_pred)
    tp, fp, fn, tn = prediction_stats(matrix)

    result = _recall_array(tp, fn)

    return np.average(result, weights=sample_weights)

def f1(y_actual, y_pred):
    sample_weights = _get_weights(y_actual)
    matrix = confusion_matrix(y_actual, y_pred)
    tp, fp, fn, tn = prediction_stats(matrix)

    _precision = _precision_array(tp, fp)
    _recall = _recall_array(tp, fn)

    with np.errstate(divide='ignore', invalid='ignore'):
        result = 2 * (_precision * _recall)/(_precision + _recall)
        result[(_precision + _recall) == 0] = 0

    return np.average(result, weights=sample_weights)