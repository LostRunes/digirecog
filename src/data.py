"""Data loading and preprocessing for MNIST."""
import numpy as np
from tensorflow.keras.datasets import mnist


def load_mnist(val_size=5000, seed=42):
    """Return train/val/test splits, pixels scaled to [0, 1], shape (N, 28, 28)."""
    (x_tr, y_tr), (x_te, y_te) = mnist.load_data()
    x_tr = x_tr.astype("float32") / 255.0
    x_te = x_te.astype("float32") / 255.0
    rng = np.random.RandomState(seed)
    idx = rng.permutation(len(x_tr))
    va, tr = idx[:val_size], idx[val_size:]
    return (x_tr[tr], y_tr[tr]), (x_tr[va], y_tr[va]), (x_te, y_te)


def flatten(x):
    return x.reshape(len(x), -1)
