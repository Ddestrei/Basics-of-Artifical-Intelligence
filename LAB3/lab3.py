import numpy as np


class Layer:
    def __init__(self, output_size, input_size):
        self.input_size = input_size
        self.output_size = output_size
        self.weights = np.random.rand(output_size, input_size)


def relu(values):
    return np.array([max(0, v) for v in values])


def relu_deriv(values):
    return np.where(values > 0, 1, 0)


class NeuralNetwork:
    def __init__(self, output_size, input_size, alfa):
        self.input_size = input_size
        self.output_size = output_size
        self.hidden_layers = []
        self.output_layer = Layer(output_size, input_size)
        self.alfa = alfa

    def add_layer(self, n):
        if len(self.hidden_layers) == 0:
            self.hidden_layers.append(Layer(n, self.output_layer.input_size))
            self.output_layer = Layer(self.output_layer.output_size, n)
        else:
            self.hidden_layers.append(Layer(n, self.output_layer.input_size))
            self.output_layer = Layer(self.output_layer.output_size, n)

    def count_hidden(self, layer, in_data):
        layer_hidden = np.zeros(layer.weights.shape[0])
        for n in range(layer.weights.shape[0]):
            layer_hidden[n] = np.dot(in_data, layer.weights.shape[0])
        layer_hidden = relu(layer_hidden)
        return layer_hidden

    def train(self, input_data, output_data, n_epoch):
        self.hidden_layers[0].weights = np.array(
            [[0.1, 0.1, -0.3], [0.1, 0.2, 0.0], [0.0, 0.7, 0.1], [0.2, 0.4, 0.0], [-0.3, 0.5, 0.1]])
        self.output_layer.weights = np.array(
            [[0.7, 0.9, -0.4, 0.8, 0.1], [0.8, 0.5, 0.3, 0.1, 0.0], [-0.3, 0.9, 0.3, 0.1, -0.2]])
        # print(self.hidden_layers[0].weights)
        for j in range(n_epoch):
            for i in range(input_data.shape[1]):
                # print(input_data[:, 0])
                layer_hidden_1 = np.zeros(self.hidden_layers[0].weights.shape[0])
                for n in range(self.hidden_layers[0].weights.shape[0]):
                    layer_hidden_1[n] = np.dot(input_data[:, i], self.hidden_layers[0].weights[n])
                # print(layer_hidden_1)
                layer_hidden_1 = relu(layer_hidden_1)
                # print(layer_hidden_1)
                layer_output = np.dot(self.output_layer.weights, layer_hidden_1)
                print(layer_output)
                layer_output_delta = 2 * 1 / self.output_layer.weights.shape[0] * (layer_output - output_data[:, i])
                # print(layer_output_delta)
                layer_hidden_1_delta = np.dot(np.transpose(self.output_layer.weights), layer_output_delta)
                # print(layer_hidden_1_delta)
                # print(relu_deriv(layer_hidden_1))
                layer_hidden_1_delta = layer_hidden_1_delta * relu_deriv(layer_hidden_1)
                # print(layer_hidden_1_delta)
                layer_output_weight_delta = np.outer(layer_output_delta, layer_hidden_1)
                # print(layer_output_weight_delta)
                layer_hidden_1_weight_delta = np.outer(layer_hidden_1_delta, np.transpose(input_data[:, i]))
                # print(layer_hidden_1_weight_delta)

                # DO ZADANIA 1 trzeba zrobić bez aktualizacji wag

                self.hidden_layers[0].weights -= self.alfa * layer_hidden_1_weight_delta
                # print(self.hidden_layers[0].weights)
                self.output_layer.weights -= self.alfa * layer_output_weight_delta
                # print(self.output_layer.weights)

    def load_weights(self):
        weights = np.loadtxt('weights.txt', dtype=float)
        return weights


network = NeuralNetwork(1, 1, 0.01)
network.add_layer(3)

print(network.output_layer.weights.shape)

input_data = np.array([[0.5, 0.1, 0.2, 0.8], [0.75, 0.3, 0.1, 0.9], [0.1, 0.7, 0.6, 0.2]])
output_data = np.array([[0.1, 0.5, 0.1, 0.7], [1.0, 0.2, 0.3, 0.6], [0.1, -0.5, 0.2, 0.2]])

network.train(input_data, output_data, 1)
