from feed_forward import FeedForwardNeuralNetwork
import numpy as np
import json

def generate_model(filename):
    model = open(filename)
    model_data = json.load(model)
    model.close()
    
    ffnn = FeedForwardNeuralNetwork()
    for layer in model_data:
        ffnn.add_layer(n_neuron=int(layer['n_neuron']), activation_function=layer['activation_function'], weights=np.array(layer['weights']), biases=np.array(layer['biases']))

    return ffnn