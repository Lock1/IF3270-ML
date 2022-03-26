import numpy as np

def sse(y_true : float, y_pred : float) -> float:
  """sum of squared error"""
  return np.sum((y_true - y_pred)**2) / 2

def cross_entropy(y_true : float, y_pred : float) -> float:
  """cross entropy"""
  return -np.sum(y_true * np.log(y_pred))