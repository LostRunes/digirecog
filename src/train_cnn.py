"""Train the CNN with on-the-fly augmentation. Saves model, history and test predictions."""
import os
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
import json, numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models, callbacks
from data import load_mnist


def build():
    return models.Sequential([
        layers.Input((28, 28, 1)),
        layers.RandomRotation(0.06), layers.RandomTranslation(0.08, 0.08), layers.RandomZoom(0.1),
        layers.Conv2D(32, 3, padding="same"), layers.BatchNormalization(), layers.Activation("relu"),
        layers.Conv2D(32, 3, padding="same"), layers.BatchNormalization(), layers.Activation("relu"),
        layers.MaxPooling2D(), layers.Dropout(0.25),
        layers.Conv2D(64, 3, padding="same"), layers.BatchNormalization(), layers.Activation("relu"),
        layers.Conv2D(64, 3, padding="same"), layers.BatchNormalization(), layers.Activation("relu"),
        layers.MaxPooling2D(), layers.Dropout(0.25),
        layers.Flatten(), layers.Dense(128, activation="relu"), layers.Dropout(0.4),
        layers.Dense(10, activation="softmax"),
    ], name="digit_cnn")


if __name__ == "__main__":
    tf.keras.utils.set_random_seed(42)
    (xtr, ytr), (xva, yva), (xte, yte) = load_mnist()
    xtr, xva, xte = [a[..., None] for a in (xtr, xva, xte)]
    m = build()
    m.compile("adam", "sparse_categorical_crossentropy", metrics=["accuracy"])
    m.summary()
    cb = [callbacks.ReduceLROnPlateau(patience=2, factor=0.5, min_lr=1e-5, verbose=1),
          callbacks.EarlyStopping(patience=5, restore_best_weights=True)]
    h = m.fit(xtr, ytr, validation_data=(xva, yva), epochs=30, batch_size=128, callbacks=cb, verbose=2)
    loss, acc = m.evaluate(xte, yte, verbose=0)
    print(f"TEST accuracy: {acc:.4f}")
    m.save(os.path.join(ROOT, "models", "cnn.keras"))
    np.save(os.path.join(ROOT, "results", "cnn_test_probs.npy"), m.predict(xte, verbose=0))
    hist = {k: [float(v) for v in vs] for k, vs in h.history.items()}
    hist.update(test_acc=float(acc), test_loss=float(loss), params=int(m.count_params()))
    json.dump(hist, open(os.path.join(ROOT, "results", "cnn_history.json"), "w"))
