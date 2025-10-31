import random
import matplotlib.pyplot as plt
import numpy as np

from load_mnist_images import load_labels, load_images

np.random.seed(42)


def relu(values):
    return np.array([max(0, v) for v in values])


def relu_deriv(values):
    return np.where(values > 0, 1, 0)


class Layer:
    def __init__(self, output_size, input_size, alfa, dropout_percent):
        self.input_size = input_size
        self.output_size = output_size
        self.weights = np.random.uniform(low=-0.1, high=0.1, size=(output_size, input_size))
        self.alfa = alfa
        self.next_layer = None
        self.dropout_percent = dropout_percent

    def return_last(self):
        if self.next_layer is None:
            return self
        else:
            return self.next_layer.return_last()

    def learn(self, input, goal):
        output = relu(np.dot(self.weights, input))  # zamiast pętli
        if self.next_layer is not None:
            delta = self.next_layer.learn(self.dropout(output), goal)
        else:
            delta = (2 / self.weights.shape[0]) * (output - goal)
        to_return = np.dot(self.weights.T, delta) * relu_deriv(input)
        self.weights -= self.alfa * np.outer(delta, np.transpose(input))
        return to_return

    def dropout(self, layer):
        n = layer.__len__()
        dropout = [0] * int(n * self.dropout_percent) + [1] * int((n * (1 - self.dropout_percent)))
        random.shuffle(dropout)
        dropout = np.array(dropout)
        layer = layer * dropout
        layer *= 1 / self.dropout_percent
        return layer

    def test(self, input, goal):
        output = np.zeros(self.weights.shape[0])
        for n in range(self.weights.shape[0]):
            output[n] = np.dot(input, self.weights[n])
        if self.next_layer is not None:
            return self.next_layer.test(output, goal)
        else:
            output = (output == output.max())
            if (output == goal).all():
                return True
            else:
                return False


class NeuralNetwork:
    def __init__(self, output_size, input_size, alfa, dropout_percent):
        self.output_size = output_size
        self.input_size = input_size
        self.alfa = alfa
        self.first_layer = Layer(output_size, input_size, self.alfa, dropout_percent)
        self.dropout_percent = dropout_percent

    def add_layer(self, n):
        if self.first_layer.next_layer is None:
            old_layer = self.first_layer
            self.first_layer = Layer(n, old_layer.input_size, self.alfa, self.dropout_percent)
            self.first_layer.next_layer = Layer(old_layer.output_size, n, self.alfa, self.dropout_percent)
        else:
            last_layer = self.first_layer.return_last()
            last_layer_out = last_layer.output_size
            last_layer_in = last_layer.input_size
            last_layer = Layer(n, last_layer_in, self.alfa, self.dropout_percent)
            last_layer.next_layer = Layer(last_layer_out, n, self.alfa, self.dropout_percent)

    def fit(self, input_data, output_data, n_epoch):
        for j in range(n_epoch):
            print("Epoch: ", j)
            for i in range(input_data.shape[1]):
                # print("Seria: ",i)
                self.first_layer.learn(input_data[:, i], output_data[:, i])

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

# mnist_network1 = NeuralNetwork(output_size=10, input_size=784, alfa=0.005, dropout_percent=0.5)
# mnist_network1.add_layer(40)
# mnist_network1.fit(np.transpose(train_images[:1000]), np.transpose(train_labels_new[:1000]), 350)
# print(mnist_network1.test(np.transpose(test_images), np.transpose(test_labels_new)))

mnist_network2 = NeuralNetwork(output_size=10, input_size=784, alfa=0.005, dropout_percent=0.5)
mnist_network2.add_layer(100)
mnist_network2.fit(np.transpose(train_images[:10000]), np.transpose(train_labels_new[:10000]), 350)
print(mnist_network2.test(np.transpose(test_images), np.transpose(test_labels_new)))

# mnist_network3 = NeuralNetwork(output_size=10, input_size=784, alfa=0.005, dropout_percent=0.5)
# mnist_network3.add_layer(100)
# mnist_network3.fit(np.transpose(train_images[:60000]), np.transpose(train_labels_new[:10000]), 350)
# print(mnist_network3.test(np.transpose(test_images), np.transpose(test_labels_new)))
