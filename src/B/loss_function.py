import numpy as np

def sse(y_true : float, y_pred : float, derivative=False) -> float:
  """sum of squared error"""
  if(derivative):
    return output_prediction - target
  return np.sum((y_true - y_pred)**2) / 2

def cross_entropy(y_true : float, y_pred : float, derivative=False) -> float:
  """cross entropy"""
  if(derivative):
    #  TODO
    pass
  return -np.sum(y_true * np.log(y_pred))
