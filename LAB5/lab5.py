import numpy as np
from scipy.signal import convolve2d
from load_mnist_images import load_labels, load_images

np.random.seed(42)

def konwolucja2D(image, filter, step=1, padding=0):
    # Wymiary obrazu i filtra
    h_image, w_image = image.shape
    h_filter, w_filter = filter.shape

    # Dodanie paddingu zerowego wokół obrazu
    if padding > 0:
        image = np.pad(image, ((padding, padding), (padding, padding)), mode='constant')

    # Nowe wymiary po paddingu
    h_image_p, w_image_p = image.shape

    # Obliczenie wymiarów wyjściowych
    h_output = (h_image_p - h_filter) // step + 1
    w_output = (w_image_p - w_filter) // step + 1

    # Inicjalizacja wyniku
    output = np.zeros((h_output, w_output))

    # Obrócenie filtra o 180° (klasyczna konwolucja)
    filter_rotation = np.flipud(np.fliplr(filter))

    # Właściwy splot
    for y in range(0, h_output):
        for x in range(0, w_output):
            element = image[y * step:y * step + h_filter, x * step:x * step + w_filter]
            output[y, x] = np.sum(element * filter_rotation)

    return output


image = np.array([
    [1, 1, 1, 0, 0],
    [0, 1, 1, 1, 0],
    [0, 0, 1, 1, 1],
    [0, 0, 1, 1, 0],
    [0, 1, 1, 0, 0]
])

# Filtr 3x3 (np. wykrywacz krawędzi)
filter = np.array([
    [1, 0, 1],
    [0, 1, 0],
    [1, 0, 1]
])

output = konwolucja2D(image, filter, step=1, padding=0)
print(output)

print(np.array([[0.1, 0.2, -0.1],[ -0.1, 0.1, 0.9], [0.1, 0.4, 0.1]]).flatten())

def relu(x):
    return np.maximum(0, x)

def relu_deriv(x):
    return (x > 0).astype(float)

class CNN_V1:
    def __init__(self):
        self.filter_size = 3
        self.n_filters = 2
        #self.filters = np.array([np.random.uniform(high=0.01, low=-0.01, size=(self.filter_size, self.filter_size)) for _ in range(self.n_filters)])
        self.filters = np.array([np.array([[0.1, 0.2, -0.1],[ -0.1, 0.1, 0.9], [0.1, 0.4, 0.1]]),np.array([[0.3, 1.1, -0.3],[ 0.1, 0.2, 0.0],[ 0.0, 1.3, 0.1]])])
        #self.output_weights = np.random.uniform(high=0.1, low=-0.1, size=(2, 4)) # size=(10, 10816)
        self.output_weights = np.array([[0.1, -0.2, 0.1, 0.3], [0.2, 0.1, 0.5, -0.3]])
        self.alfa = 0.01
        pass

    def seg_and_conv(self,image, filters):
        feature_maps = []
        for f in filters:
            f = f.T
            for x in range(0,image.shape[1]-f.shape[1]+1, 1):
                row = []
                for y in range(0,image.shape[0]-f.shape[0]+1, 1):
                    sum = 0
                    for f_x in range(0,f.shape[1]):
                        for f_y in range(0, f.shape[0]):
                            sum += image[y+f_y, x+f_x] * f[f_y, f_x]
                    row.append(sum)
                    print(sum)
                row = np.array(row)
                feature_maps.append(row.T)
        feature_maps = np.array(feature_maps)
        return feature_maps

    def conv_backward(self, d_out, image, filters):
        n_filters, kh, kw = filters.shape
        d_filters = np.zeros_like(filters)

        for f in range(n_filters):
            for i in range(image.shape[0] - kh + 1):
                for j in range(image.shape[1] - kw + 1):
                    region = image[i:i + kh, j:j + kw]
                    d_filters[f] += d_out[f, i, j] * region
        return d_filters

    def train(self, img, goal):
        kernel_layer = self.seg_and_conv(img,self.filters)
        kernel_layer = kernel_layer.T
        kernel_layer = relu(kernel_layer)
        layer_output = self.output_weights @ kernel_layer.flatten()
        layer_output_delta = 2 * 1 / layer_output.shape[0] * (layer_output - goal)
        kernel_layer_delta = self.output_weights.T @ layer_output_delta
        kernel_layer_delta_reshape = kernel_layer_delta.reshape(kernel_layer.shape)
        #kernel_layer_delta_reshape = relu_deriv(kernel_layer_delta_reshape)
        layer_output_weight_delta = np.outer(layer_output_delta, kernel_layer.flatten())
        kernel_layer_weight_delta = self.conv_backward(kernel_layer_delta_reshape,img,self.filters)
        self.filters -= self.alfa * kernel_layer_weight_delta
        self.output_weights -= self.alfa * layer_output_weight_delta

    def fit(self, input_data, output_data, n_epoch):
        for i in range(n_epoch):
            for j in range(input_data.shape[0]):
                if j % 100 == 0:
                    print(j)
                self.train(input_data[j,:,: ], output_data[j,:])

    def test(self, input_data, output_data):
        true = 0
        for i in range(input_data.shape[0]):
            kernel_layer = self.conv_layer(input_data[i,:,: ],self.filters)
            kernel_layer = relu(kernel_layer)
            layer_output = self.output_weights @ kernel_layer.flatten()
            layer_output = (layer_output == layer_output.max())
            if (layer_output == output_data[i,:]).all():
                true += 1

        return true / input_data.shape[1]


test_labels = load_labels('../MNIST_ORG/t10k-labels.idx1-ubyte')
test_images = load_images('../MNIST_ORG/t10k-images.idx3-ubyte')

train_labels = load_labels('../MNIST_ORG/train-labels.idx1-ubyte')
train_images = load_images('../MNIST_ORG/train-images.idx3-ubyte')

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
# cnn_v1_1 = CNN_V1()
# cnn_v1_1.fit(train_images[:100],train_labels_new[:100],1)
# print(cnn_v1_1.test(test_images,test_labels_new))


cnn = CNN_V1()
cnn.train(np.array([[8.5, 0.65, 1.2], [9.5, 0.8, 1.3], [9.9, 0.8, 0.5], [9.0, 0.9, 1.0]]), np.array([0, 1]))

