from activation_function import sigmoid

def dLinear():
  """Linear function derivative"""
  return 1

def dSigmoid(X):
  """Sigmoid function derivative"""
  return (sigmoid(X)*(1-sigmoid(X)))

def dReLU(X):
  """ReLU function derivative"""
  return 0 if X > 0 else 1
