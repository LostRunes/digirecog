"""Train the 36-class (0-9, A-Z) CNN on EMNIST. Run prepare_emnist.py first."""
import os
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
import json, numpy as np
import tensorflow as tf
from tensorflow.keras import callbacks
from train_cnn import build

LABELS = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ"
EPOCHS = int(os.environ.get("EPOCHS", 12))

if __name__ == "__main__":
    tf.keras.utils.set_random_seed(42)
    d = np.load(os.path.join(ROOT, "data", "emnist36.npz"))
    x, y, xte, yte = d["x_train"], d["y_train"].astype("int64"), d["x_test"], d["y_test"].astype("int64")
    n_val = len(x) // 20                                    # 5 % validation (already shuffled)
    xva, yva, xtr, ytr = x[:n_val], y[:n_val], x[n_val:], y[n_val:]
    print("train", xtr.shape, "val", xva.shape, "test", xte.shape, flush=True)

    def ds(a, b, shuffle):                                  # uint8 -> float32 on the fly (low RAM)
        s = tf.data.Dataset.from_tensor_slices((a, b))
        if shuffle: s = s.shuffle(20000, seed=42)
        return s.batch(128).map(lambda i, l: (tf.cast(i, tf.float32)[..., None] / 255.0, l)).prefetch(2)

    m = build(36)
    m.compile("adam", "sparse_categorical_crossentropy", metrics=["accuracy"])
    cb = [callbacks.ReduceLROnPlateau(patience=1, factor=0.5, min_lr=1e-5, verbose=1),
          callbacks.EarlyStopping(patience=3, restore_best_weights=True)]
    h = m.fit(ds(xtr, ytr, True), validation_data=ds(xva, yva, False), epochs=EPOCHS, callbacks=cb, verbose=2)
    loss, acc = m.evaluate(ds(xte, yte, False), verbose=0)
    print(f"TEST accuracy: {acc:.4f}")
    m.save(os.path.join(ROOT, "models", "cnn_alnum.keras"))
    probs = m.predict(ds(xte, yte, False), verbose=0)
    np.save(os.path.join(ROOT, "results", "alnum_test_probs.npy"), probs.astype("float16"))
    np.save(os.path.join(ROOT, "results", "alnum_test_labels.npy"), yte.astype("uint8"))
    hist = {k: [float(v) for v in vs] for k, vs in h.history.items()}
    hist.update(test_acc=float(acc), test_loss=float(loss), params=int(m.count_params()),
                n_train=int(len(xtr)), n_val=int(len(xva)), n_test=int(len(xte)))
    json.dump(hist, open(os.path.join(ROOT, "results", "alnum_history.json"), "w"))
