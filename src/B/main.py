from neural_net import NeuralNetwork
import numpy as np

NN = NeuralNetwork(max_iter=100, batch_size=2, epoch=3, error_threshold=0.01, learning_rate=0.01)
NN.add_layer(n_neuron=2)
NN.add_layer(n_neuron=2, activation_function='relu')
NN.add_layer(n_neuron=2, activation_function='relu')
NN.add_layer(n_neuron=1, activation_function='softmax')

x = np.array([[2.1234, 4.1234], [2.134, 3.341], [8.452345, 2.34], [1.23, 3.45], [2.1234, 4.1234], [2.134, 3.341], [8.452345, 2.34], [1.23, 3.45]])
y = np.array([[0], [0], [1], [1], [0], [0], [1], [1]])
NN.fit(x, y)
pred = NN.predict(x)
print(pred)