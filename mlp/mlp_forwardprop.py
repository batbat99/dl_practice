import numpy as np

class MLP:

    def __init__(self, num_inputs, num_hidden, num_outputs):

        self.num_inputs = num_inputs
        self.num_hidden = num_hidden
        self.num_outputs = num_outputs

        layers = [self.num_inputs] + self.num_hidden + [self.num_outputs]

        # initiate random weights
        self.weights = []
        for i in range(len(layers)-1):
            w = np.random.rand(layers[i], layers[i+1])
            self.weights.append(w)


    def forward_propagate(self, inputs):

        activations = inputs

        for w in self.weights:
            # calculate net input
            net_inputs = np.dot(activations, w)

            # calculate the activations
            activations = self._segmoid(net_inputs)
        
        return activations
    

    def _segmoid(self, x):
        return 1 / (1 + np.exp(-x))

if __name__ == "__main__":

    # create an MLP
    mlp = MLP(3, [3, 5], 2)

    # create some inputs
    inputs = np.random.rand(mlp.num_inputs)

    # perform forward prop
    outputs = mlp.forward_propagate(inputs)

    # print the results
    print("the network input is: {}".format(inputs))
    print("the network output is: {}".format(outputs))
