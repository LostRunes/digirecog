"""Usage: python predict.py path/to/digit.png"""
import os
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
import sys, cv2
from tensorflow.keras.models import load_model
from preprocess import to_mnist

img = cv2.imread(sys.argv[1], cv2.IMREAD_GRAYSCALE)
p = load_model(os.path.join(ROOT, "models", "cnn.keras")).predict(to_mnist(img)[None, ..., None], verbose=0)[0]
print(f"Predicted digit: {p.argmax()}  (confidence {p.max()*100:.2f}%)")
