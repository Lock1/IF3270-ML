import numpy as np

def linear(X):
  """Fungsi aktivasi linear"""
  return X

def sigmoid(X):
  """Fungsi aktivasi sigmoid"""
  return 1 / (1 + np.exp(-X))

def ReLU(X):
  """Fungsi aktivasi ReLU"""
  return np.maximum(0, X)

def softmax(X):
  """Fungsi aktivasi softmax"""
  return np.exp(X) / np.sum(np.exp(X))
