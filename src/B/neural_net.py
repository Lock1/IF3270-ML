import numpy as np
from layer import Layer
from loss_function import sse, cross_entropy

class NeuralNetwork:
    def __init__(self, max_iter : int, batch_size : int, error_threshold : float, learning_rate : float = 0.01, random_state=None):
        self.layers          = []
        self.n_layers        = 0
        self.prediction      = None
        self.max_iter        = max_iter
        self.iteration       = None
        self.batch_size      = batch_size
        self.error_threshold = error_threshold
        self.learning_rate   = learning_rate
        np.random.seed(random_state)
    
    def add_layer(self, n_neuron : int, activation_function : str = 'linear'):
        """ 
            method to add a layer to nn

            parameters:
                n_neuron : number of neuron in layer
                activation_function : activation function of layer
        """
        self.layers.append(Layer(n_neuron=n_neuron, activation_function=activation_function))
        
        # initialize weights and biases for hidden layer and output layer
        if(self.n_layers!=0):
            self.layers[-1].weights = np.random.randn(self.layers[-2].n_neuron, self.layers[-1].n_neuron) * 0.001

        self.n_layers += 1

    def get_error(self, activation_function_name, target, output_prediction, derivative=False):
        """
            calculate loss value for output layer

            parameters:
                activation_function_name : activation function name (linear, softmax, etc)
                output_prediction : predicted output value
                derivative : if true, return derivative of loss function
        """
        if(activation_function_name == "softmax"):
            return cross_entropy(output_prediction, derivative)
        else:
            return sse(target, output_prediction, derivative)

    def chain_rule(self, expected_target, reversed_layer):
        """
            chain rule for backpropagation 

            parameters:
                expected_target : expected target value
                reversed_layer : layer index to start backpropagation
        """
        # derivasi nilai error terhadap nilai output
        dError_dOutput = self.get_error(self.layers[-1].activation_function_name, expected_target, self.layers[-1].output, derivative=True)
        value = dError_dOutput

        for i in range(reversed_layer):
            # derivasi nilai output terhadap nilai input pada hidden layer
            dOutput_dInput = self.layers[-1-i].activation_function(self.layers[-1-i].input, derivative=True)

            # dError/dOutput * dOutput/dInput = dError/dInput
            value = value * dOutput_dInput
            
            value = np.dot(value, self.layers[-1-i].weights.T)

        dOutput_dInput = self.layers[-1 - reversed_layer].activation_function(self.layers[-1 - reversed_layer].input, derivative=True)
        value = value * dOutput_dInput

        dInput_dWeights = self.layers[-2 - reversed_layer].output
        return np.dot(dInput_dWeights.T, value)
        

    def forward_pass(self, input) -> float:
        """ 
            forward pass
            
            parameters:
                input : input value
        """

        # firts layer is input layer (input layer has no weights and biases)
        self.layers[0].forward_pass(input)

        # forward propagation for hidden layer and output layer
        for i in range(1, self.n_layers):
            self.layers[i].z = np.dot(self.layers[i-1].output, self.layers[i].weights) + self.layers[i].biases
            self.layers[i].forward_pass(self.layers[i].z)

        prediction = self.layers[-1].output.copy()
        prediction = np.argmax(prediction, axis=1)
        prediction = np.reshape(prediction, (prediction.shape[0],1))
        return prediction

    def back_propagation(self, expected_target):
        """
            back propagation

            parameters:

        """
        for i in range(1, self.n_layers):
            grad = self.chain_rule(expected_target, self.n_layers-1-i)
            self.layers[i].weights = self.layers[i].weights - (self.learning_rate * grad)
            self.layers[i].biases  = self.layers[i].biases - (self.learning_rate * np.mean(grad))

    def fit(self, X, y):
        """
            fit the model

            parameters:
                X : input value
                y : target value
        """
        epoch = len(X)//self.batch_size

        self.iteration  = 0
        error = 99
        
        while (error > self.error_threshold):
            total_error = 0
            i = 0
            while((self.iteration < self.max_iter) & (i<epoch) & (error > self.error_threshold)):
                input = X[i*self.batch_size:(i+1)*self.batch_size]
                target = np.array(y[i*self.batch_size:(i+1)*self.batch_size]).T
                target = np.reshape(target, (target.shape[0], 1))

                prediction = self.forward_pass(input)
                self.back_propagation(prediction)
                e = np.mean(sse(target, prediction)).mean()
                total_error += e
                i += 1
                self.iteration += 1
            error = total_error
        
    def predict(self, X):
        prediction = self.forward_pass(X).flatten()
        return prediction

    def info(self):
        print("Jumlah layer: {}".format(self.n_layers))
        print("Jumlah iterasi: {}".format(self.iteration))
        print("Jumlah fitur: {}".format(self.layers[0].input.shape[1]))
        print("Jumlah output: {}".format(self.layers[-1].output.shape[1]))

        # Print array weight untuk setiap layer
        for i in range(1, self.n_layers):
            print("Layer {} weights".format(i))
            print(self.layers[i].weights)
            print("")
            print("Layer {} biases".format(i))
            print(self.layers[i].biases)
            print("")
            