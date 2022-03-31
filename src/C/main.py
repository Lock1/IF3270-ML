from neural_net import NeuralNetwork
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_iris
from metrics import confusion_matrix, prediction_stats, accuracy, precision, recall, f1
# from save_load import saveModel, loadModel

dataset = load_iris()

X = dataset.data
Y = dataset.target

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.1)

NN = NeuralNetwork(max_iter=500, batch_size=10, error_threshold=0.01, learning_rate=0.01, random_state=42069)
NN.add_layer(n_neuron=4)
NN.add_layer(n_neuron=3, activation_function='relu')
NN.add_layer(n_neuron=3, activation_function='relu')
NN.add_layer(n_neuron=2, activation_function='sigmoid')
NN.add_layer(n_neuron=3)

NN.fit(X_train, Y_train)
y_pred = NN.predict(X_train)
conf_matrix = confusion_matrix(Y_test, y_pred)

print(f'Confusion Matrix:\n {conf_matrix}')
print(f'Precision = {precision(Y_test, y_pred)}')
print(f'Accuracy score = {accuracy(Y_test, y_pred)}')
print(f'Recall Score = {recall(Y_test, y_pred)}')
print(f'F1 Score = {f1(Y_test,y_pred)}')