import math

def sigmoid(x):
    y = 1/(1+math.exp(-x))
    return y


class Neuron:
    def __init__(self, inputs, weights, function):
        self.inputs = inputs
        self.weights = weights
        self.activation_function = function
        self.y = self.activate()

    def activate(self):
        # find net input
        h = 0
        for x, w in zip(self.inputs, self.weights):
            h += x*w
        
        # perform activation
        return self.activation_function(h)

if __name__ == "__main__":
    inputs = [.5, .3, .2]
    weights = [.4, .7, .2]
    n = Neuron(inputs, weights, sigmoid)
    output = n.y
    print(output)