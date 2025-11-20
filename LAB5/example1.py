import numpy as np


class Conv2D_2D:
    def __init__(self, num_filters, kernel_size):
        self.num_filters = num_filters
        self.kernel_size = kernel_size

        # Filtry: (F, KH, KW)
        self.filters = np.array([np.array([[0.1, 0.2, -0.1],[ -0.1, 0.1, 0.9], [0.1, 0.4, 0.1]]),np.array([[0.3, 1.1, -0.3],[ 0.1, 0.2, 0.0],[ 0.0, 1.3, 0.1]])])

    def forward(self, x):
        """
        x: wejście 2D (H, W)
        """
        self.x = x
        H, W = x.shape
        KH, KW = self.kernel_size, self.kernel_size

        out_h = H - KH + 1
        out_w = W - KW + 1

        self.output = np.zeros((self.num_filters, out_h, out_w))

        for f in range(self.num_filters):
            for i in range(out_h):
                for j in range(out_w):
                    region = x[i:i + KH, j:j + KW]
                    self.output[f, i, j] = np.sum(region * self.filters[f].T)

        return self.output

    def backward(self, d_out, lr=0.01):
        """
        d_out: gradient (F, out_h, out_w)
        """
        H, W = self.x.shape
        KH, KW = self.kernel_size, self.kernel_size

        dX = np.zeros_like(self.x)
        dFilters = np.zeros_like(self.filters)

        _, out_h, out_w = d_out.shape

        for f in range(self.num_filters):
            for i in range(out_h):
                for j in range(out_w):
                    region = self.x[i:i + KH, j:j + KW]

                    dFilters[f] += d_out[f, i, j] * region
                    dX[i:i + KH, j:j + KW] += d_out[f, i, j] * self.filters[f].T

        # Update: SGD
        self.filters -= lr * dFilters

        return dX, dFilters

#cnn.train(np.array([[8.5, 0.65, 1.2], [9.5, 0.8, 1.3], [9.9, 0.8, 0.5], [9.0, 0.9, 1.0]]), np.array([0, 1]))


# obraz 7x7 (szary)
img = np.array([[8.5, 0.65, 1.2], [9.5, 0.8, 1.3], [9.9, 0.8, 0.5], [9.0, 0.9, 1.0]])

conv = Conv2D_2D(num_filters=2, kernel_size=3)

out = conv.forward(img)

d_out = np.ones_like(out)
dX, dW = conv.backward(d_out)

print("Output shape:", out.shape)  # (3, 5, 5)
print("dX:", dX.shape)             # (7, 7)
print("dW:", dW.shape)             # (3, 3, 3)