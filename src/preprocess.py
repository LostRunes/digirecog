"""Convert a real-world image / canvas drawing into an MNIST-style 28x28 input.

MNIST digits are white on black, scaled to fit a 20x20 box and centred by
centre-of-mass inside a 28x28 frame. Reproducing that is essential for
good accuracy on user-drawn digits.
"""
import cv2
import numpy as np
from scipy import ndimage


def to_mnist(img, invert=None):
    """img: 2-D uint8 array (grayscale). Returns float32 (28, 28) in [0, 1]."""
    img = img.astype("uint8")
    if invert is None:                      # dark ink on light paper -> invert
        invert = img.mean() > 127
    if invert:
        img = 255 - img
    img = cv2.GaussianBlur(img, (3, 3), 0)
    _, bw = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    ys, xs = np.where(bw > 0)
    if len(ys) == 0:
        return np.zeros((28, 28), "float32")
    crop = img[ys.min():ys.max() + 1, xs.min():xs.max() + 1]
    h, w = crop.shape
    scale = 20.0 / max(h, w)                # longest side -> 20 px
    crop = cv2.resize(crop, (max(1, round(w * scale)), max(1, round(h * scale))),
                      interpolation=cv2.INTER_AREA)
    canvas = np.zeros((28, 28), "uint8")
    y0, x0 = (28 - crop.shape[0]) // 2, (28 - crop.shape[1]) // 2
    canvas[y0:y0 + crop.shape[0], x0:x0 + crop.shape[1]] = crop
    cy, cx = ndimage.center_of_mass(canvas)  # centre by mass like MNIST
    canvas = ndimage.shift(canvas, (14 - cy, 14 - cx), order=1)
    return (canvas / 255.0).astype("float32")
