import numpy as np


class Layer:
    def __init__(self, output_size, input_size):
        self.input_size = input_size
        self.output_size = output_size
        self.weights = np.random.rand(output_size, input_size)



class NeuralNetwork:
    def __init__(self, output_size, input_size):
        self.input_size = input_size
        self.output_size = output_size
        self.hidden_layers = []
        self.output_layer = Layer(output_size, input_size)


    def add_layer(self, n):
        if len(self.hidden_layers) == 0:
            self.hidden_layers.append(Layer(n, self.output_layer.input_size))
            self.output_layer = Layer(self.output_layer.output_size, n)
        else:
            self.hidden_layers.append(Layer(n, self.output_layer.input_size))
            self.output_layer = Layer(self.output_layer.output_size, n)

    def train(self, input_data, output_data, n_epoch):
        for j in range(n_epoch):
            for i in range(input_data.shape[1]):


    def load_weights(self):
        weights = np.loadtxt('weights.txt', dtype=float)
        return weights

    def relu(values):
        return [max(0, v) for v in values]