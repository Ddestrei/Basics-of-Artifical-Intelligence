import numpy as np

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
print(konwolucja2D(np.ones((28,28)),np.ones((3,3)),step=1,padding=0).shape)


def relu(x):
    return np.maximum(0, x)

def softmax(x):
    exps = np.exp(x - np.max(x))
    return exps / np.sum(exps)

class CNN_V1:
    def __init__(self):
        self.kernel_weights = np.random.uniform(high=0.01,low=0.01,size=(16,26*26))
        self.output_weights = np.random.uniform(high=0.01,low=0.01,size=(10,28))
        pass

    def train(self, img, goal):
        image_sections =