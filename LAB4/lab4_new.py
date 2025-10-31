import random
import matplotlib.pyplot as plt
import numpy as np

from LAB4.load_mnist_images import load_labels, load_images

# Zakładamy, że 'load_mnist_images' i funkcje w niej są dostępne
# from load_mnist_images import load_labels, load_images

np.random.seed(42)


# --- Funkcje aktywacji ---

def relu(values):
    # Wektoryzacja: max(0, v)
    return np.maximum(0, values)


def relu_deriv(values):
    # Wektoryzacja: 1 tam, gdzie > 0, 0 w pozostałych przypadkach
    return np.where(values > 0, 1.0, 0.0)


def softmax(values):
    """Zastosowanie Softmax do wektora wartości."""
    # Stabilizacja numeryczna: odejmujemy max(values)
    exp_values = np.exp(values - np.max(values, axis=0, keepdims=True))
    return exp_values / np.sum(exp_values, axis=0, keepdims=True)


# --- Klasa Warstwy ---

class Layer:
    def __init__(self, output_size, input_size, alfa, dropout_percent=0.0):
        self.input_size = input_size
        self.output_size = output_size
        # Wagi inicjowane lepiej dla ReLU (He initialization), np. sqrt(2/input_size)
        self.weights = np.random.randn(output_size, input_size) * np.sqrt(2.0 / input_size)
        self.biases = np.zeros((output_size, 1))  # Dodanie wektora biasów
        self.alfa = alfa
        self.next_layer = None
        self.prev_layer = None  # Dodatkowe pole do łatwiejszego budowania sieci
        self.dropout_percent = dropout_percent

        # Zmienne do przechowywania wartości podczas forward pass (dla backprop)
        self.input_cache = None
        self.z_cache = None  # Suma ważona przed aktywacją
        self.activation_cache = None
        self.dropout_mask = None  # Maska dropout

    def set_next(self, next_layer):
        """Łączy obecną warstwę z następną."""
        self.next_layer = next_layer
        next_layer.prev_layer = self

    def is_output_layer(self):
        """Sprawdza, czy warstwa jest ostatnią w sieci."""
        return self.next_layer is None

    def dropout(self, activation):
        """Zoptymalizowana implementacja Inverted Dropout."""
        if self.dropout_percent == 0.0 or self.is_output_layer():
            return activation

        keep_prob = 1.0 - self.dropout_percent
        # Tworzenie maski (ten sam wymiar co aktywacja)
        self.dropout_mask = (np.random.rand(*activation.shape) < keep_prob).astype(float)

        # Zastosowanie maski i skalowanie
        activated_layer = activation * self.dropout_mask
        activated_layer *= (1.0 / keep_prob)
        return activated_layer

    def forward(self, input_data, is_training=True):
        """Wektoryzowany Forward Propagation."""
        # Z = W * A_prev + B (W: output_size x input_size, A_prev: input_size x 1)
        Z = np.dot(self.weights, input_data) + self.biases

        if self.is_output_layer():
            # Warstwa wyjściowa używa Softmax dla klasyfikacji
            A = softmax(Z)
        else:
            # Warstwy ukryte używają ReLU
            A = relu(Z)

        if is_training:
            self.input_cache = input_data
            self.z_cache = Z
            self.activation_cache = A  # Aktywacje przed dropoutem, jeśli jest

            # Zastosowanie dropoutu tylko w fazie treningu i tylko na warstwach ukrytych
            if not self.is_output_layer():
                A = self.dropout(A)

        return A

    def backward(self, d_A):
        """Wektoryzowany Backpropagation."""

        if self.is_output_layer():
            # Delta na wyjściu (dla Softmax z Cross-Entropy Loss)
            # Domyślnie błąd już zawiera dZ (d_A == dZ)
            dZ = d_A
        else:
            # Warstwy ukryte

            # Jeśli była maska dropout (trening), musimy ją zastosować do d_A
            if self.dropout_mask is not None:
                keep_prob = 1.0 - self.dropout_percent
                d_A = d_A * self.dropout_mask / keep_prob

            # dZ = d_A * g'(Z) (g'(Z) to pochodna aktywacji)
            dZ = d_A * relu_deriv(self.z_cache)

        # dW = dZ * A_prev.T
        dW = np.dot(dZ, self.input_cache.T)

        # dB = dZ
        dB = np.sum(dZ, axis=1, keepdims=True)

        # d_A_prev = W.T * dZ (przekazanie błędu do poprzedniej warstwy)
        d_A_prev = np.dot(self.weights.T, dZ)

        # Aktualizacja wag i biasów
        self.weights -= self.alfa * dW
        self.biases -= self.alfa * dB

        return d_A_prev


# --- Klasa Sieci Neuronowej ---

class NeuralNetwork:
    def __init__(self, output_size, input_size, alfa, dropout_percent=0.0):
        self.output_size = output_size
        self.input_size = input_size
        self.alfa = alfa
        self.dropout_percent = dropout_percent
        self.layers = []  # Przechowuje warstwy ukryte i wyjściową

        # Inicjalizacja: Użyjemy prostej inicjalizacji z jedną warstwą
        # Dodamy warstwy w metodzie add_layer, a ostatnią wyjściową na końcu budowania.
        self.head = None  # Pierwsza warstwa (ukryta)
        self.tail = None  # Ostatnia warstwa (wyjściowa)
        self.is_built = False

    def add_layer(self, n_neurons):
        """Dodaje warstwę ukrytą do sieci."""
        if self.is_built:
            raise Exception("Nie można dodawać warstw po rozpoczęciu trenowania.")

        current_input_size = self.head.output_size if self.head else self.input_size
        new_layer = Layer(n_neurons, current_input_size, self.alfa, self.dropout_percent)

        if not self.head:
            self.head = new_layer
        else:
            self.tail.set_next(new_layer)

        self.tail = new_layer  # Ustawia tail na nowo dodaną warstwę

    def _finalize_network(self):
        """Dodaje warstwę wyjściową po dodaniu wszystkich warstw ukrytych."""
        if self.is_built:
            return

        # Warstwa wyjściowa. Jej input_size to output_size ostatniej dodanej warstwy
        final_input_size = self.tail.output_size if self.tail else self.input_size
        output_layer = Layer(self.output_size, final_input_size, self.alfa, dropout_percent=0.0)

        if not self.head:
            self.head = output_layer  # Sieć tylko z warstwą wyjściową (niezalecane)
            self.tail = output_layer
        else:
            self.tail.set_next(output_layer)
            self.tail = output_layer

        # Zbudowanie listy warstw w kolejności
        current = self.head
        while current:
            self.layers.append(current)
            current = current.next_layer

        self.is_built = True
        print(f"Sieć zbudowana. Warstwy: {[(layer.input_size, layer.output_size) for layer in self.layers]}")

    def fit(self, input_data, output_data, n_epoch):
        """Trening sieci."""
        if not self.is_built:
            self._finalize_network()

        # Zmiana kształtu danych: (features, samples) dla wektoryzacji
        X = input_data
        Y = output_data
        num_samples = X.shape[1]

        for j in range(n_epoch):
            print(f"Epoch: {j + 1}/{n_epoch}")

            # Wektoryzacja pętli (trenowanie na pojedynczych próbkach)
            for i in range(num_samples):
                input_sample = X[:, i].reshape(-1, 1)  # (features, 1)
                goal_sample = Y[:, i].reshape(-1, 1)  # (output_size, 1)

                # 1. Forward Pass
                current_activation = input_sample
                for layer in self.layers:
                    current_activation = layer.forward(current_activation, is_training=True)

                # 2. Backward Pass (Używamy pochodnej Cross-Entropy Loss + Softmax)
                # dZ_L = A_L - Y
                d_A = current_activation - goal_sample

                # Propagacja wsteczna przez warstwy
                next_d_A = d_A
                for layer in reversed(self.layers):
                    next_d_A = layer.backward(next_d_A)

    def predict(self, input_data):
        """Funkcja predykcyjna (test) dla wektora wejściowego."""
        if not self.is_built:
            raise Exception("Sieć nie została jeszcze zbudowana ani wytrenowana.")

        # Warunek, aby obsłużyć pojedyncze próbki i zbiory testowe
        if input_data.ndim == 1:
            input_data = input_data.reshape(-1, 1)

        current_activation = input_data
        for layer in self.layers:
            # W fazie testowania dropout jest wyłączony (is_training=False)
            current_activation = layer.forward(current_activation, is_training=False)

        return current_activation

    def test(self, input_data, output_data):
        """Obliczenie dokładności (accuracy)."""
        if not self.is_built:
            self._finalize_network()

        X = input_data
        Y = output_data
        num_samples = X.shape[1]

        # Obliczanie predykcji dla całego zbioru (można zoptymalizować do batchy)
        true = 0
        for i in range(num_samples):
            input_sample = X[:, i].reshape(-1, 1)
            goal_sample = Y[:, i].reshape(-1, 1)

            prediction = self.predict(input_sample)

            # Wybór klasy o najwyższym prawdopodobieństwie
            predicted_class = np.argmax(prediction)
            true_class = np.argmax(goal_sample)

            if predicted_class == true_class:
                true += 1

        return true / num_samples

    # Funkcje load_weights i save_weights zostawione bez zmian, ale
    # z optymalizacyjnymi uwagami o użyciu np.savez do zapisu wielu macierzy wag.

    def load_weights(self):
        # Wymagałoby przebudowy, aby wczytać wagi i biasy dla każdej warstwy
        # np.load('weights.npz') i wczytanie 'w_0', 'b_0', 'w_1', 'b_1', itd.
        pass

    def save_weights(self, weights):
        # np.savez('weights.npz', w_0=self.layers[0].weights, b_0=self.layers[0].biases, ...)
        pass

# --- Użycie kodu (pozostało niezmienione, ale uruchamia się na zoptymalizowanej sieci) ---
# ...
# mnist_network2 = NeuralNetwork(output_size=10, input_size=784, alfa=0.005, dropout_percent=0.5)
# mnist_network2.add_layer(100)
# # UWAGA: Jeśli sieć nie ma ręcznie dodanej warstwy wyjściowej, należy ją dodać
# # w optymalizowanej wersji jest ona automatycznie dodawana w fit().
#
# # Reszta kodu pozostaje bez zmian


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
