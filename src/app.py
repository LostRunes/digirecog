"""Draw a digit with the mouse and get a live prediction (Tkinter)."""
import os
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
import numpy as np, tkinter as tk
from PIL import Image, ImageDraw
from tensorflow.keras.models import load_model
from preprocess import to_mnist

model = load_model(os.path.join(ROOT, "models", "cnn.keras"))
S = 280
root = tk.Tk(); root.title("Handwritten Digit Recognition")
cv = tk.Canvas(root, width=S, height=S, bg="black", cursor="cross"); cv.grid(row=0, column=0, columnspan=2)
img = Image.new("L", (S, S), 0); dr = ImageDraw.Draw(img)
out = tk.Label(root, text="Draw a digit", font=("Segoe UI", 16)); out.grid(row=1, column=0, columnspan=2)
bars = tk.Label(root, text="", font=("Consolas", 9), justify="left"); bars.grid(row=2, column=0, columnspan=2)


def predict():
    x = to_mnist(np.array(img), invert=False)[None, ..., None]
    p = model.predict(x, verbose=0)[0]
    out.config(text=f"Prediction: {p.argmax()}   ({p.max()*100:.1f}%)")
    bars.config(text="\n".join(f"{d}: {'#' * int(p[d] * 30):<30} {p[d]*100:5.1f}%" for d in np.argsort(-p)[:3]))


def paint(e):
    r = 11
    cv.create_oval(e.x - r, e.y - r, e.x + r, e.y + r, fill="white", outline="white")
    dr.ellipse([e.x - r, e.y - r, e.x + r, e.y + r], fill=255)


def clear():
    cv.delete("all"); img.paste(0, (0, 0, S, S)); out.config(text="Draw a digit"); bars.config(text="")


cv.bind("<B1-Motion>", paint); cv.bind("<ButtonRelease-1>", lambda e: predict())
tk.Button(root, text="Clear", command=clear, width=14).grid(row=3, column=0, pady=6)
tk.Button(root, text="Quit", command=root.destroy, width=14).grid(row=3, column=1)
root.mainloop()
