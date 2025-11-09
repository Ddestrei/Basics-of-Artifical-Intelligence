import numpy as np


def sliding_reshape(arr, window_rows, step):
    m, n = arr.shape
    num_windows = (m-window_rows) // step + 1
    new_rows = []

    for w in range(num_windows):
        row = []
        for col in range(n):
            fragment = arr[w:w + window_rows, col]
            row.extend(fragment)
        new_rows.append(row)

    return np.array(new_rows)

input = np.array([[8.5, 0.65, 1.2], [9.5, 0.8, 1.3], [9.9, 0.8, 0.5], [9.0, 0.9, 1.0]])
expected_output = np.array([0, 1])
kernel_1_weights = np.array([0.1, 0.2, -0.1, -0.1, 0.1, 0.9, 0.1, 0.4, 0.1])
kernel_2_weights = np.array([0.3, 1.1, -0.3, 0.1, 0.2, 0.0, 0.0, 1.3, 0.1])
kernel_weights = np.array([kernel_1_weights,kernel_2_weights])
W_y = np.array([[0.1, -0.2, 0.1, 0.3], [0.2, 0.1, 0.5, -0.3]])
image_sections = sliding_reshape(input,3,1)
kernel_layer = image_sections @ kernel_weights.T
output_layer = W_y @ kernel_layer.flatten()
delta_output_layer = 2 * 1 / output_layer.shape[0] * (output_layer - expected_output)
kernel_layer_delta = W_y.T @ delta_output_layer
kernel_layer_delta_reshape = kernel_layer_delta.reshape((2,2))
weight_delta_output_layer = np.array([[delta_output_layer[0]],[delta_output_layer[1]]]) * kernel_layer.flatten()
kernel_layer_weight_delta = kernel_layer_delta_reshape.T @ image_sections
kernel_weights -= 0.01 * kernel_layer_weight_delta
W_y -= 0.01 * weight_delta_output_layer
print(kernel_layer_weight_delta)
