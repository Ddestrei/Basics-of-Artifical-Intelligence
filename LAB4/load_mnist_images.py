import struct

import numpy as np


def load_labels(filepath):
    with open(filepath, 'rb') as f:
        labels = np.frombuffer(f.read(), dtype=np.uint8, offset=8)
    return labels


def load_images(filepath):
    with open(filepath, 'rb') as f:
        magic_number = struct.unpack('>I', f.read(4))[0]
        if magic_number != 2051:
            raise ValueError(f"Błędny magic number: {magic_number}. Oczekiwano 2051 (0x00000803).")

        num_images = struct.unpack('>I', f.read(4))[0]
        num_rows = struct.unpack('>I', f.read(4))[0]
        num_cols = struct.unpack('>I', f.read(4))[0]

        images = np.frombuffer(f.read(), dtype=np.uint8)

    images = images.reshape(num_images, num_rows, num_cols)

    return images
