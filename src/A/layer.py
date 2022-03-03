import numpy as np
from activation_function import linear, sigmoid, ReLU, softmax

class Layer:
    def __init__(self, n_neuron, weights=None, biases=None, activation_function='linear'):
        activations = {
            'linear': linear,
            'sigmoid': sigmoid,
            'relu': ReLU,
            'softmax': softmax
        }

        if(n_neuron < 1):
            raise ValueError("Neuron must be greater than 0")
        if(activation_function not in ('linear', 'sigmoid', 'relu', 'softmax')):
            raise ValueError("Activation function must be one of 'linear', 'sigmoid', 'relu', 'softmax'")
        else:
            self.n_neuron = n_neuron
            self.activation_function = activations[activation_function]
            self.weights = weights
            self.biases = biases
            self.activation_value = None

    def forward_pass(self, input):
        self.input = input
        self.activation_value = self.activation_function(input)
        return self.activation_value
