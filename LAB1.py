import numpy as np


## ZADANIE 1

print("Zadanie 1")

def neuron(input_size, weights, bias):
    return np.dot(input_size, weights) + bias


input_size = np.array([0.5, 0.75, 0.1])
weights = np.array([0.5, 0.75, 0.1])
bias = 0.5
print(neuron(input_size, weights, bias))


## ZADANIE 2

print("Zadanie 2")

def neural_network(input_size, weights, bias):
    neurons = weights.shape[0]
    output = np.zeros(neurons)
    for n in range(neurons):
        output[n] = neuron(input_size, weights[n], bias)
    return output


input_size = np.array([0.5, 0.75, 0.1])
weights = np.array([[0.1, 0.1, -0.3], [0.1, 0.2, 0.0], [0.0, 0.7, 0.1], [0.2, 0.4, 0.0], [-0.3, 0.5, 0.1]])

print(neural_network(input_size, weights, 0))

## ZADANIE 3

print("Zadanie 3")


def deep_neural_network(input_size, weights, bias):
    hidden_layer = neural_network(input_size, weights, bias)
    hidden_weights = np.array([[0.7, 0.9, -0.4, 0.8, 0.1], [0.8, 0.5, 0.3, 0.1, 0.0], [-0.3, 0.9, 0.3, 0.1, -0.2]])
    output_layer = neural_network(hidden_layer, hidden_weights, bias)
    return output_layer


input = np.array([0.5, 0.75, 0.1])
weights = np.array([[0.1, 0.1, -0.3], [0.1, 0.2, 0.0], [0.0, 0.7, 0.1], [0.2, 0.4, 0.0], [-0.3, 0.5, 0.1]])
out = deep_neural_network(input, weights, 0)
print(out)


## Zadanie 4

print("Zadanie 4")

class Layer:
    def __init__(self, output_size, input_size):
        self.input_size = input_size
        self.output_size = output_size
        self.weights = np.random.rand(output_size, input_size)


class NeuralNetwork:

    def __init__(self, output_size, input_size):
        self.input_size = input_size
        self.output_size = output_size
        self.layers = []
        self.layers.append(Layer(output_size, input_size))

    def add_layer(self, n):
        if len(self.layers) == 1:
            base_layer = self.layers.pop()
            self.layers.append(Layer(n, base_layer.input_size))
            self.layers.append(Layer(base_layer.output_size, n))
        else:
            last_layer = self.layers.pop()
            self.layers.append(Layer(n, last_layer.input_size))
            self.layers.append(Layer(self.output_size, n))

    def predict(self, input):
        output = input
        for l in self.layers:
            output = neural_network(output, l.weights, 0)
        return output

    def load_weights(self):
        weights = np.loadtxt('weights.txt', dtype=float)
        return weights

network = NeuralNetwork(5, 3)
input = np.array([0.5, 0.75, 0.1])
print(network.predict(input))

network.add_layer(6)
print(network.predict(input))

network.add_layer(7)
print(network.predict(input))

print(network.load_weights())