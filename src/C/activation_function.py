import numpy as np

def linear(X : float, derivative : bool = False) -> float:
  """Fungsi aktivasi linear"""
  if derivative:
    return 1
  return X

def sigmoid(X : float, derivative : bool = False) -> float:
  """Fungsi aktivasi sigmoid"""
  X = np.clip(X,-500,500)
  if derivative:
    return sigmoid(X) * (1-sigmoid(X))
  return 1 / (1 + np.exp(-X))

def ReLU(X : float, derivative : bool = False) -> float:
  """Fungsi aktivasi ReLU"""
  if derivative:
    value = X.copy()
    value[value < 0] = 0
    value[value >= 0] = 1
    return value
  return np.maximum(0, X)

def softmax(X : float, derivative : bool = False) -> float:
  """Fungsi aktivasi softmax"""
  if derivative:
    return np.exp(X) / np.sum(np.exp(X) * (1 - np.exp(X) / np.sum(np.exp(X))))
  return np.exp(X) / np.sum(np.exp(X))
