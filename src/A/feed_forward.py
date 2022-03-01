import numpy as np
from layer import Layer

class FeedForwardNeuralNetwork:
    def __init__(self):
        self.layers = []
        self.n_layers = 0
    
    def add_layer(self, n_neuron, activation_function='linear'): 
        self.layers.append(Layer(n_neuron=n_neuron, activation_function=activation_function))
        self.n_layers += 1

    def predict(self, X):
        for i in range(self.n_layers):
            if(i == 0):
                self.layers[i].weights = np.random.randn(X.shape[1], self.layers[i].n_neuron)
            else:
                self.layers[i].weights = np.random.randn(self.layers[i-1].n_neuron, self.layers[i].n_neuron)
            
            self.layers[i].biases = np.zeros(self.layers[i].n_neuron)

        for i in range(self.n_layers):
            layer = self.layers[i]
            if(i==0):
                val = np.dot(X, layer.weights) + layer.biases
            else:
                val = np.dot(self.layers[i-1].activation_value, layer.weights) + layer.biases

            layer.forward_pass(val)

    def info(self):
        print("Number of layers: {}".format(self.n_layers))
        for i in range(self.n_layers):
            print("Layer {}".format(i))
            print("Number of neurons: {}".format(self.layers[i].n_neuron))
            print("Activation function: {}".format(self.layers[i].activation_function))
            print("Weights: \n{}".format(self.layers[i].weights))
            print("Biases: \n{}".format(self.layers[i].biases))
            print("Activation value: \n{}".format(self.layers[i].activation_value))
            print("="*20)
