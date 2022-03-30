from neural_net import NeuralNetwork
import numpy as np
from sklearn.model_selection import KFold
from sklearn.datasets import load_iris
from metrics import confusion_matrix, prediction_stats, accuracy, precision, recall, f1

dataset = load_iris()

X = dataset.data
Y = dataset.target

kf = KFold(n_splits=10, shuffle=True, random_state=1)

NN = NeuralNetwork(max_iter=200, batch_size=10, error_threshold=0.1, learning_rate=0.1, random_state=42069)
NN.add_layer(n_neuron=4)
NN.add_layer(n_neuron=4, activation_function='relu')
NN.add_layer(n_neuron=4, activation_function='relu')
NN.add_layer(n_neuron=3)

i = 1
for train_idx, test_idx in kf.split(X):
    print(f'''------------------- FOLD {i} -------------------''')
    X_train, X_test = X[train_idx], X[test_idx]
    Y_train, Y_test = Y[train_idx], Y[test_idx]
    NN.fit(X_train, Y_train)
    y_pred = NN.predict(X_test)
    
    conf_matrix = confusion_matrix(Y_test, y_pred)
    print(f'Confusion Matrix:\n {conf_matrix}')
    print(f'Precision = {precision(Y_test, y_pred)}')
    print(f'Accuracy score = {accuracy(Y_test, y_pred)}')
    print(f'Recall Score = {recall(Y_test, y_pred)}')
    print(f'F1 Score = {f1(Y_test,y_pred)}\n')
    i += 1