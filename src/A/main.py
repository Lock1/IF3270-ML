from feed_forward import FeedForwardNeuralNetwork
import numpy as np

ffnn = FeedForwardNeuralNetwork()
ffnn.add_layer(n_neuron=2, activation_function='sigmoid')
ffnn.add_layer(n_neuron=1, activation_function='sigmoid')
ffnn.predict(np.array([[0, 0], [0, 1], [1, 0], [1, 1]]))
ffnn.info()