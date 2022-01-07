import numpy as np
from random import random

class MLP:

    def __init__(self, num_inputs, num_hidden, num_outputs):

        self.num_inputs = num_inputs
        self.num_hidden = num_hidden
        self.num_outputs = num_outputs

        layers = [self.num_inputs] + self.num_hidden + [self.num_outputs]

        # initiate random weights
        self.weights = []
        for i in range(len(layers) - 1):
            w = np.random.rand(layers[i], layers[i+1])
            self.weights.append(w)
        
        self.activations = []
        for i in range(len(layers)):
            a = np.zeros(layers[i])
            self.activations.append(a)
        
        self.derivatives = []
        for i in range(len(layers) - 1):
            d = np.zeros((layers[i], layers[i+1]))
            self.derivatives.append(d)



    def forward_propagate(self, inputs):

        activations = inputs
        self.activations[0] = inputs

        for i, w in enumerate(self.weights):
            # calculate net input
            net_inputs = np.dot(activations, w)

            # calculate the activations
            activations = self._segmoid(net_inputs)
            self.activations[i+1] = activations

        
        return activations

    
    def back_propagate(self, error, verbose=False):

        for i in reversed(range(len(self.derivatives))):
            activations = self.activations[i+1]
            delta = error * self._segmoid_derivative(activations)
            delta_reshaped = delta.reshape(delta.shape[0], -1).T
            current_activations = self.activations[i]
            current_activations_reshaped = current_activations.reshape(current_activations.shape[0], -1)
            self.derivatives[i] = np.dot(current_activations_reshaped, delta_reshaped)
            error = np.dot(delta, self.weights[i].T)

            if verbose:
                print("Derivatives for W{}:{}".format(i, self.derivatives[i]))

        return error

    def gradient_descent(self, learning_rate):
        for i in range(len(self.weights)):
            weights = self.weights[i]
            derivatives = self.derivatives[i]
            weights += derivatives * learning_rate

    def train(self, inputs, targets, epochs, learning_rate):
        
        for i in range(epochs):
            sum_error = 0
            for (input, target) in zip(inputs, targets):

                # forward propagation
                output = self.forward_propagate(input)
                
                # calculate error
                error = target - output

                # back propagate
                self.back_propagate(error)

                # apply gradiant descent 
                self.gradient_descent(learning_rate)

                sum_error += self._mse(target, output)

                # report error
            print("Error: {} at epoch {}".format(sum_error / len(inputs), i))


    

    def _segmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def _segmoid_derivative(self, x):
        return x * (1.0 - x)

    def _mse(self, target, output):
        return np.average((target-output)**2)

if __name__ == "__main__":

    # create an MLP
    mlp = MLP(2, [5], 1)

    inputs = np.array([[random() / 2 for _ in range(2)] for _ in range(10000)])
    targets = np.array([[i[0] + i[1]] for i in inputs])

    # train mlp
    mlp.train(inputs, targets, 50, 0.1)

    # create some inputs
    inputs = np.array([0.1, 0.3])
    target = np.array([0.3])
    print(mlp.forward_propagate(inputs))

