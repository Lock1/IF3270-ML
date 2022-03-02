import numpy as np
from layer import Layer

class FeedForwardNeuralNetwork:
    def __init__(self):
        self.layers = []
        self.n_layers = 0
        self.prediction = None
    
    def add_layer(self, n_neuron, activation_function='linear', weights=None, biases=None):
        self.layers.append(Layer(n_neuron=n_neuron, activation_function=activation_function))
        self.layers[-1].weights = weights
        self.layers[-1].biases = biases
        self.n_layers += 1

    def predict(self, X):
        for i in range(self.n_layers):
            layer = self.layers[i]
            if(i==0):
                val = np.dot(X, layer.weights) + layer.biases
            else:
                val = np.dot(self.layers[i-1].activation_value, layer.weights) + layer.biases

            layer.forward_pass(val)

        prediction = self.layers[-1].activation_value
        prediction = prediction.reshape(prediction.shape[0], 1)

        for i in range(len(prediction)):
            if(prediction[i] > 0.5):
                prediction[i] = 1
            else:
                prediction[i] = 0

        self.prediction = prediction
        return self.prediction

    def info(self):
        print("Number of layers: {}".format(self.n_layers))
        for i in range(self.n_layers):
            print("Layer {}".format(i))
            print("Number of neurons: {}".format(self.layers[i].n_neuron))
            print("Activation function: {}".format(self.layers[i].activation_function))
            print("Weights: {}".format(self.layers[i].weights))
            print("Biases: {}".format(self.layers[i].biases))
            print("Activation value: {}".format(self.layers[i].activation_value))
            print("="*20)

        print("Prediction: {}".format(self.prediction))
