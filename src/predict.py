"""Usage: python predict.py path/to/image.png [letters]
Add 'letters' to use the 36-class model (0-9, A-Z); default is the 10-class digit model."""
import os
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
import sys, cv2
from tensorflow.keras.models import load_model
from preprocess import to_mnist

letters = len(sys.argv) > 2 and sys.argv[2] == "letters"
labels = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ" if letters else "0123456789"
model = load_model(os.path.join(ROOT, "models", "cnn_alnum.keras" if letters else "cnn.keras"))
img = cv2.imread(sys.argv[1], cv2.IMREAD_GRAYSCALE)
p = model.predict(to_mnist(img)[None, ..., None], verbose=0)[0]
print(f"Predicted: {labels[p.argmax()]}  (confidence {p.max() * 100:.2f}%)")
