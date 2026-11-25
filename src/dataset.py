import gzip
import struct
import urllib.request
import os
import numpy as np

BASE_URL = "https://storage.googleapis.com/cvdf-datasets/mnist/"

FILES = [
    "train-images-idx3-ubyte.gz",
    "train-labels-idx1-ubyte.gz",
    "t10k-images-idx3-ubyte.gz",
    "t10k-labels-idx1-ubyte.gz"
]

def download_dataset():
    os.makedirs("dataset", exist_ok=True)
    for filename in FILES:
        path = os.path.join("dataset", filename)
        if not os.path.exists(path):
            print("Downloading:", filename)
            urllib.request.urlretrieve(BASE_URL + filename, path)

def load_images(filename):
    with gzip.open(filename, "rb") as f:
        magic, count, rows, cols = struct.unpack(">IIII", f.read(16))
        images = np.frombuffer(f.read(), dtype=np.uint8)

    images = images.reshape(count, rows * cols)
    return images.astype(np.float32) / 255.0

def load_labels(filename):
    with gzip.open(filename, "rb") as f:
        magic, count = struct.unpack(">II", f.read(8))
        labels = np.frombuffer(f.read(), dtype=np.uint8)

    return labels.astype(np.int32)

if __name__ == "__main__":
    download_dataset()

    X_train = load_images("dataset/train-images-idx3-ubyte.gz")
    y_train = load_labels("dataset/train-labels-idx1-ubyte.gz")
    X_test = load_images("dataset/t10k-images-idx3-ubyte.gz")
    y_test = load_labels("dataset/t10k-labels-idx1-ubyte.gz")
    X_train.astype(np.float32).tofile("dataset/X_train.bin")
    X_test.astype(np.float32).tofile("dataset/X_test.bin")
    y_train.astype(np.int32).tofile("dataset/y_train.bin")
    y_test.astype(np.int32).tofile("dataset/y_test.bin")

    print("Training images:", X_train.shape)
    print("Training labels:", y_train.shape)
    print("Testing images:", X_test.shape)
    print("Testing labels:", y_test.shape)
    print("\nBinary files created successfully.")