from neural_net import NeuralNetwork
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_iris

dataset = load_iris()

X = dataset.data
Y = dataset.target

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)

NN = NeuralNetwork(max_iter=100, batch_size=3, epoch=100, error_threshold=0.1, learning_rate=0.1, random_state=69420)
NN.add_layer(n_neuron=4)
NN.add_layer(n_neuron=2, activation_function='softmax')
NN.add_layer(n_neuron=2, activation_function='relu')
NN.add_layer(n_neuron=2, activation_function='relu')
NN.add_layer(n_neuron=3)

NN.fit(X, Y)
pred = NN.predict(X)
NN.info()