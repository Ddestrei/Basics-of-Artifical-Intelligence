import numpy as np

from load_mnist_images import load_labels, load_images

np.random.seed(42)


def relu(values):
    return np.maximum(0, values)


def relu_deriv(values):
    return np.where(values > 0, 1, 0)


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def sigmoid_deriv(x):
    s = sigmoid(x)
    return s * (1 - s)


def tanh(x):
    return np.tanh(x)


def tanh_deriv(x):
    return 1 - np.tanh(x) ** 2


def softmax(x):
    # stabilna wersja numerycznie (chroni przed overflowem)
    x = np.array(x)
    e_x = np.exp(x - np.max(x, axis=-1, keepdims=True))
    return e_x / np.sum(e_x, axis=-1, keepdims=True)


def softmax_deriv(x):
    s = softmax(x)
    return s * (1 - s)


class Layer:
    def __init__(self, output_size, input_size, alfa, dropout_percent, activation, activation_deriv):
        self.input_size = input_size
        self.output_size = output_size
        self.weights = np.random.uniform(low=-0.1, high=0.1, size=(output_size, input_size))
        self.alfa = alfa
        self.next_layer = None
        self.dropout_percent = dropout_percent
        self.activation = activation
        self.activation_deriv = activation_deriv

    def return_last(self):
        if self.next_layer is None:
            return self
        else:
            return self.next_layer.return_last()

    def mini_batch_GD(self, input_d, goal, batch_size):
        for i in range(0, len(input_d), batch_size):
            batch_input = input_d[:, i:i + batch_size]
            batch_goal = goal[:, i:i + batch_size]
            self.learn_batch(batch_input, batch_goal, batch_size)

    def learn_batch(self, input, goal, batch_size):
        output = self.weights @ input  # zamiast pętli
        if self.next_layer is not None:
            delta = self.next_layer.learn_batch(self.activation(self.dropout_batch(output)), goal, batch_size)
        else:
            delta = (2 / self.weights.shape[0]) * (output - goal) / batch_size
        to_return = (self.weights.T @ delta) * self.activation_deriv(input)
        self.weights -= self.alfa * (delta @ np.transpose(input))
        return to_return

    def dropout_batch(self, layer):
        layer = np.array(layer)
        rows, cols = layer.shape
        mask = np.random.binomial(1, 1 - self.dropout_percent, size=(rows, cols))
        layer = layer * mask
        layer /= (1 - self.dropout_percent)
        return layer

    def test(self, input, goal):
        output = self.weights @ input
        if self.next_layer is not None:
            return self.next_layer.test(self.activation(output), goal)
        else:
            output = (output == output.max())
            if (output == goal).all():
                return True
            else:
                return False


class NeuralNetwork:
    def __init__(self, output_size, input_size, alfa, dropout_percent, activation, activation_deriv):
        self.output_size = output_size
        self.input_size = input_size
        self.alfa = alfa
        self.first_layer = Layer(output_size, input_size, self.alfa, dropout_percent, activation, activation_deriv)
        self.dropout_percent = dropout_percent
        self.activation = activation
        self.activation_deriv = activation_deriv

    def add_layer(self, n):
        if self.first_layer.next_layer is None:
            old_layer = self.first_layer
            self.first_layer = Layer(n, old_layer.input_size, self.alfa, self.dropout_percent, self.activation,
                                     self.activation_deriv)
            self.first_layer.next_layer = Layer(old_layer.output_size, n, self.alfa, self.dropout_percent,
                                                self.activation, self.activation_deriv)
        else:
            last_layer = self.first_layer.return_last()
            last_layer_out = last_layer.output_size
            last_layer_in = last_layer.input_size
            last_layer = Layer(n, last_layer_in, self.alfa, self.dropout_percent, self.activation,
                               self.activation_deriv)
            last_layer.next_layer = Layer(last_layer_out, n, self.alfa, self.dropout_percent, self.activation,
                                          self.activation_deriv)

    def fit_batch(self, input_data, output_data, n_epoch, batch_size):
        for j in range(n_epoch):
            self.first_layer.mini_batch_GD(input_data, output_data, batch_size)

    def test(self, input_data, output_data):
        true = 0
        for i in range(input_data.shape[1]):
            true += self.first_layer.test(input_data[:, i], output_data[:, i])
        return true / input_data.shape[1]

    def load_weights(self):
        weights = np.loadtxt('weights.txt', dtype=float)
        return weights

    def save_weights(self, weights):
        np.savetxt('weights.txt', dtype=float)


# Zadanie 1

test_labels = load_labels('MNIST_ORG/t10k-labels.idx1-ubyte')
test_images = load_images('MNIST_ORG/t10k-images.idx3-ubyte').reshape(10000, -1)

train_labels = load_labels('MNIST_ORG/train-labels.idx1-ubyte')
train_images = load_images('MNIST_ORG/train-images.idx3-ubyte').reshape(60000, -1)

# --- Normalize pixel values ---
train_images = train_images / 255.0
test_images = test_images / 255.0

# --- One-hot encode labels (clean version) ---
num_classes = 10

# Reshape labels to (N, 1)
train_labels = train_labels.reshape(-1, 1)
test_labels = test_labels.reshape(-1, 1)

# Create one-hot encoded versions
train_labels_new = np.eye(num_classes)[train_labels.flatten()]
test_labels_new = np.eye(num_classes)[test_labels.flatten()]

# test_net = NeuralNetwork(output_size=3, input_size=3, alfa=0.1, dropout_percent=0.5)
# test_net.add_layer(5)
# test_net.first_layer.weights = np.array(
#     [[0.1, 0.1, -0.3], [0.1, 0.2, 0.0], [0.0, 0.7, 0.1], [0.2, 0.4, 0.0], [-0.3, 0.5, 0.1]])
# test_net.first_layer.next_layer.weights = np.array(
#     [[0.7, 0.9, -0.4, 0.8, 0.1], [0.8, 0.5, 0.3, 0.1, 0.0], [-0.3, 0.9, 0.3, 0.1, -0.2]])
#
# input_data = np.array([[0.5, 0.1, 0.2, 0.8], [0.75, 0.3, 0.1, 0.9], [0.1, 0.7, 0.6, 0.2]])
# output_data = np.array([[0.1, 0.5, 0.1, 0.7], [1.0, 0.2, 0.3, 0.6], [0.1, -0.5, 0.2, 0.2]])
#
# test_net.fit(input_data, output_data, 1)
#

## wagi są aktualizowane po każdej serii
#
# mnist_network3 = NeuralNetwork(output_size=10, input_size=784, alfa=0.01, dropout_percent=0.0, activation=relu,
#                                activation_deriv=relu_deriv)
# mnist_network3.add_layer(40)
# mnist_network3.fit_batch(np.transpose(train_images[:60000]), np.transpose(train_labels_new[:10000]), 10, 1)
# print(mnist_network3.test(np.transpose(test_images), np.transpose(test_labels_new)))
#
# mnist_network4 = NeuralNetwork(output_size=10, input_size=784, alfa=0.1, dropout_percent=0.5, activation=relu,
#                                activation_deriv=relu_deriv)
# mnist_network4.add_layer(40)
# mnist_network4.fit_batch(np.transpose(train_images[:1000]), np.transpose(train_labels_new[:1000]), 350, 100)
# print(mnist_network4.test(np.transpose(test_images), np.transpose(test_labels_new)))
#
# mnist_network5 = NeuralNetwork(output_size=10, input_size=784, alfa=0.1, dropout_percent=0.5, activation=relu,
#                                activation_deriv=relu_deriv)
# mnist_network5.add_layer(100)
# mnist_network5.fit_batch(np.transpose(train_images[:10000]), np.transpose(train_labels_new[:10000]), 350, 100)
# print(mnist_network5.test(np.transpose(test_images), np.transpose(test_labels_new)))
#
# mnist_network6 = NeuralNetwork(output_size=10, input_size=784, alfa=0.1, dropout_percent=0.5, activation=relu,
#                                activation_deriv=relu_deriv)
# mnist_network6.add_layer(100)
# mnist_network6.fit_batch(np.transpose(train_images[:60000]), np.transpose(train_labels_new[:10000]), 350, 100)
# print(mnist_network6.test(np.transpose(test_images), np.transpose(test_labels_new)))
#
# mnist_network7 = NeuralNetwork(output_size=10, input_size=784, alfa=0.2, dropout_percent=0.5, activation=sigmoid,
#                                activation_deriv=sigmoid_deriv)
# mnist_network7.add_layer(100)
# mnist_network7.fit_batch(np.transpose(train_images[:60000]), np.transpose(train_labels_new[:10000]), 350, 100)
# print(mnist_network7.test(np.transpose(test_images), np.transpose(test_labels_new)))
#
# mnist_network8 = NeuralNetwork(output_size=10, input_size=784, alfa=0.2, dropout_percent=0.5, activation=tanh,
#                                activation_deriv=tanh_deriv)
# mnist_network8.add_layer(100)
# mnist_network8.fit_batch(np.transpose(train_images[:60000]), np.transpose(train_labels_new[:10000]), 350, 100)
# print(mnist_network8.test(np.transpose(test_images), np.transpose(test_labels_new)))
#
# mnist_network9 = NeuralNetwork(output_size=10, input_size=784, alfa=0.2, dropout_percent=0.5, activation=softmax,
#                                activation_deriv=softmax_deriv)
# mnist_network9.add_layer(100)
# mnist_network9.fit_batch(np.transpose(train_images[:60000]), np.transpose(train_labels_new[:10000]), 350, 100)
# print(mnist_network9.test(np.transpose(test_images), np.transpose(test_labels_new)))

best = NeuralNetwork(output_size=10, input_size=784, alfa=0.1, dropout_percent=0.1, activation=relu,
                     activation_deriv=relu_deriv)
best.add_layer(100)
best.fit_batch(np.transpose(train_images[:60000]), np.transpose(train_labels_new[:10000]), 350, 100)
print(best.test(np.transpose(test_images), np.transpose(test_labels_new)))
