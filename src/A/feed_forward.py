import numpy as np
from layer import Layer

class FeedForwardNeuralNetwork:
    def __init__(self):
        self.layers = []
        self.n_layers = 0
        self.prediction = None
    
    def add_layer(self, n_neuron, activation_function='linear', weights=None, biases=None):
        self.layers.append(Layer(n_neuron=n_neuron, weights=weights, biases=biases, activation_function=activation_function))
        self.n_layers += 1

    def predict(self, X):
        for i in range(self.n_layers):
            layer = self.layers[i]
            if(i==0):
                val = np.dot(X, layer.weights) + layer.biases
            else:
                val = np.dot(self.layers[i-1].activation_value, layer.weights) + layer.biases

            layer.forward_pass(val)

        self.prediction = np.copy(self.layers[-1].activation_value)
        self.prediction = self.prediction.reshape(self.prediction.shape[0], 1)

        for i in range(len(self.prediction)):
            if(self.prediction[i] > 0.5):
                self.prediction[i] = 1
            else:
                self.prediction[i] = 0

        return self.prediction

    def info(self):
        print("Number of layers: {}".format(self.n_layers))
        for i in range(self.n_layers):
            print("Layer {}".format(i+1))
            print("Number of neurons: {}".format(self.layers[i].n_neuron))
            print("Activation function: {}".format(self.layers[i].activation_function))
            print("Weights: \n{}".format(self.layers[i].weights))
            print("Biases: \n{}".format(self.layers[i].biases))
            print("Activation value: \n{}".format(self.layers[i].activation_value))
            print("="*20)

        print("Prediction: {}".format(self.prediction))
