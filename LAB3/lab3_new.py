import matplotlib.pyplot as plt
import numpy as np

from load_mnist_images import load_labels, load_images

np.random.seed(42)


def relu(values):
    return np.array([max(0, v) for v in values])


def relu_deriv(values):
    return np.where(values > 0, 1, 0)


class Layer:
    def __init__(self, output_size, input_size, alfa):
        self.input_size = input_size
        self.output_size = output_size
        self.weights = np.random.uniform(low=-0.1, high=0.1, size=(output_size, input_size))
        self.alfa = alfa
        self.next_layer = None

    def return_last(self):
        if self.next_layer is None:
            return self
        else:
            return self.next_layer.return_last()

    def learn(self, input, goal):
        output = relu(np.dot(self.weights, input))
        if self.next_layer is not None:
            delta = self.next_layer.learn(output, goal)
        else:
            delta = (2 / self.weights.shape[0]) * (output - goal)
        to_return = np.dot(self.weights.T, delta) * relu_deriv(input)
        self.weights -= self.alfa * np.outer(delta, np.transpose(input))
        return to_return

    def test(self, input, goal):
        output = np.dot(self.weights, input)
        if self.next_layer is not None:
            return self.next_layer.test(output, goal)
        else:
            output = (output == output.max())
            if (output == goal).all():
                return True
            else:
                return False


class NeuralNetwork:
    def __init__(self, output_size, input_size, alfa):
        self.output_size = output_size
        self.input_size = input_size
        self.alfa = alfa
        self.first_layer = Layer(output_size, input_size, self.alfa)

    def add_layer(self, n):
        if self.first_layer.next_layer is None:
            old_layer = self.first_layer
            self.first_layer = Layer(n, old_layer.input_size, self.alfa)
            self.first_layer.next_layer = Layer(old_layer.output_size, n, self.alfa)
        else:
            last_layer = self.first_layer.return_last()
            last_layer_out = last_layer.output_size
            last_layer_in = last_layer.input_size
            last_layer = Layer(n, last_layer_in, self.alfa)
            last_layer.next_layer = Layer(last_layer_out, n, self.alfa)

    def fit(self, input_data, output_data, n_epoch, t_i, t_o):
        for j in range(n_epoch):
            print("Epoch: ", j)
            for i in range(input_data.shape[1]):
                if i % 100 == 0:
                    print(self.test(t_i, t_o))
                    pass
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


# ZADANIE 1 i 2

network = NeuralNetwork(1, 1, 0.01)
network.add_layer(3)

network.first_layer.weights = np.array(
    [[0.1, 0.1, -0.3], [0.1, 0.2, 0.0], [0.0, 0.7, 0.1], [0.2, 0.4, 0.0], [-0.3, 0.5, 0.1]])
network.first_layer.next_layer.weights = np.array(
    [[0.7, 0.9, -0.4, 0.8, 0.1], [0.8, 0.5, 0.3, 0.1, 0.0], [-0.3, 0.9, 0.3, 0.1, -0.2]])

input_data = np.array([[0.5, 0.1, 0.2, 0.8], [0.75, 0.3, 0.1, 0.9], [0.1, 0.7, 0.6, 0.2]])
output_data = np.array([[0.1, 0.5, 0.1, 0.7], [1.0, 0.2, 0.3, 0.6], [0.1, -0.5, 0.2, 0.2]])

# network.fit(input_data, output_data, 50)


# Zadanie 3
test_labels = load_labels('MNIST_ORG/t10k-labels.idx1-ubyte')

test_images = load_images('MNIST_ORG/t10k-images.idx3-ubyte')
test_images = test_images.reshape(10000, -1)

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

mnist_network = NeuralNetwork(output_size=10, input_size=784, alfa=0.01)
mnist_network.add_layer(40)

mnist_network.fit(np.transpose(train_images), np.transpose(train_labels_new), 2,np.transpose(test_images), np.transpose(test_labels_new))
print(mnist_network.test(np.transpose(test_images), np.transpose(test_labels_new)))

# Zadanie 4

training = np.loadtxt("training.txt")
training_input = training[:, :-1]
training_output_raw = training[:, -1:]
training_output = np.zeros((training_output_raw.shape[0], 4))
for i in range(len(training_output)):
    if training_output_raw[i][0] == 1.0:
        training_output[i] = np.array([1, 0, 0, 0])
    elif training_output_raw[i][0] == 2.0:
        training_output[i] = np.array([0, 1, 0, 0])
    elif training_output_raw[i][0] == 3.0:
        training_output[i] = np.array([0, 0, 1, 0])
    elif training_output_raw[i][0] == 4.0:
        training_output[i] = np.array([0, 0, 0, 1])

test = np.loadtxt("test.txt")
test_input = test[:, :-1]
test_output_raw = test[:, -1:]
test_output = np.zeros((test_output_raw.shape[0], 4))
for i in range(len(test_output)):
    if test_output_raw[i][0] == 1.0:
        test_output[i] = np.array([1, 0, 0, 0])
    elif test_output_raw[i][0] == 2.0:
        test_output[i] = np.array([0, 1, 0, 0])
    elif test_output_raw[i][0] == 3.0:
        test_output[i] = np.array([0, 0, 1, 0])
    elif test_output_raw[i][0] == 4.0:
        test_output[i] = np.array([0, 0, 0, 1])

training_input = np.transpose(training_input)
training_output = np.transpose(training_output)

test_input = np.transpose(test_input)
test_output = np.transpose(test_output)

network = NeuralNetwork(output_size=4, input_size=3, alfa=0.1)

network.add_layer(5)
# network.fit(training_input, training_output, 8)
# print(network.test(test_input, test_output))
