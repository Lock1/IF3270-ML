from neural_net import NeuralNetwork
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_iris

dataset = load_iris()

X = dataset.data
Y = dataset.target

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)

NN = NeuralNetwork(max_iter=100, batch_size=2, epoch=3, error_threshold=0.01, learning_rate=0.01)
NN.add_layer(n_neuron=4)
NN.add_layer(n_neuron=2, activation_function='relu')
NN.add_layer(n_neuron=2, activation_function='relu')
NN.add_layer(n_neuron=1, activation_function='softmax')

NN.fit(X_train, Y_train)
pred = NN.predict(X_test)
print(pred)