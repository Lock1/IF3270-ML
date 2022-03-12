import numpy as np

def sum_of_squared_error(tk, ok):
  """Fungsi loss sum of squared errors"""
  return 1/2*(np.square(tk-ok))

def cross_entropy(pk):
  """Fungsi loss cross entropy"""
  return -np.log(pk)
