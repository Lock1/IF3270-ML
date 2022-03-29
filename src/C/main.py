from neural_net import NeuralNetwork
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_iris
from sklearn.metrics import accuracy_score, f1_score
from metrics import confusion_matrix

dataset = load_iris()

X = dataset.data
Y = dataset.target

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)

NN = NeuralNetwork(max_iter=200, batch_size=10, error_threshold=0.1, learning_rate=0.1, random_state=42069)
NN.add_layer(n_neuron=4)
NN.add_layer(n_neuron=4, activation_function='relu')
NN.add_layer(n_neuron=4, activation_function='relu')
NN.add_layer(n_neuron=3)

NN.fit(X, Y)
y_pred = NN.predict(X)
NN.info()

accuracy = accuracy_score(Y, y_pred)
f1 = f1_score(Y, y_pred, average='weighted')

print(confusion_matrix(Y, y_pred))
print(f'Accuracy score = {accuracy}')
print(f'F1 Score = {f1}')