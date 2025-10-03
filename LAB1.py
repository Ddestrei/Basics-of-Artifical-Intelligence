import numpy as np


## ZADANIE 1

def neuron(input, weights, bias):
    return np.dot(input, weights) + bias


input = np.array([0.5, 0.75, 0.1])
weights = np.array([0.5, 0.75, 0.1])
print(weights)
print(input)
bias = 0.5
print(neuron(input, weights, bias))


## ZADANIE 2

def neural_network(input, weights, bias):
    neurons = weights.shape[0]
    output = np.zeros(neurons)
    for n in range(neurons):
        output[n] = neuron(input,weights[n],bias)
    return output



input = np.array([0.5, 0.75, 0.1])
weights = np.array([[0.1, 0.1, -0.3],[ 0.1, 0.2, 0.0],[0.0,0.7,0.1],[0.2,0.4,0.0],[-0.3,0.5,0.1]])
print(weights)

neural_network(input,weights,0)

## ZADANIE 3

print("Zadanie 3")

def deep_neural_network(input, weights, bias):
    hidden_layer = neural_network(input, weights, bias)
    hidden_weights = np.array([[0.7,0.9,-0.4,0.8,0.1],[0.8,0.5,0.3,0.1,0.0],[-0.3,0.9,0.3,0.1,-0.2]])
    print(hidden_layer)
    print(hidden_weights)
    output_layer = neural_network(hidden_layer, hidden_weights, bias)
    return output_layer

input = np.array([0.5, 0.75, 0.1])
weights = np.array([[0.1, 0.1, -0.3],[ 0.1, 0.2, 0.0],[0.0,0.7,0.1],[0.2,0.4,0.0],[-0.3,0.5,0.1]])
out = deep_neural_network(input,weights,0)
print(out)

## Zadanie 4

class NeuralNetwork:

    def __init__(self, input_size, output_size):
        self.input_size = input_size
        self.output_size = output_size
        self.weights = []
        self.weights.append(np.random.rand(input_size, output_size))

    def add_layer(self, n, weight_range):

        pass
    def predict(self, input):
        pass
    def load_weights(self):
        pass

