"""Metrics + figures for the 36-class (0-9, A-Z) model. Run after train_alnum.py."""
import os
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
import json, numpy as np, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, f1_score

R = os.path.join(ROOT, "results") + os.sep
L = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ"
probs = np.load(R + "alnum_test_probs.npy").astype("float32"); y = np.load(R + "alnum_test_labels.npy").astype(int)
h = json.load(open(R + "alnum_history.json")); pred = probs.argmax(1)
plt.rcParams.update({"font.family": "DejaVu Sans", "axes.spines.top": False, "axes.spines.right": False})
BLUE, ORANGE, GREY = "#2f5d7c", "#c0623b", "#9aa7b3"

cm = confusion_matrix(y, pred, labels=range(36))
recall = cm.diagonal() / cm.sum(1)
f1 = f1_score(y, pred, average=None, labels=range(36))
is_digit = y < 10
off = cm.copy(); np.fill_diagonal(off, 0)
pairs = sorted(((int(off[i, j]), L[i], L[j]) for i in range(36) for j in range(36) if off[i, j]), reverse=True)[:8]
# how many errors are digit<->letter look-alikes (0/O, 1/I, 5/S, 2/Z, 8/B, 6/G, 9/Q ...)
cross = int(off[:10, 10:].sum() + off[10:, :10].sum())
res = dict(accuracy=float((pred == y).mean()), digit_accuracy=float((pred[is_digit] == y[is_digit]).mean()),
           letter_accuracy=float((pred[~is_digit] == y[~is_digit]).mean()), macro_f1=float(f1.mean()),
           errors=int((pred != y).sum()), n_test=int(len(y)), digit_letter_confusions=cross,
           top_pairs=[{"true": a, "pred": b, "count": n} for n, a, b in pairs],
           best=[L[i] for i in np.argsort(-f1)[:3]], worst=[L[i] for i in np.argsort(f1)[:3]],
           best_f1=[float(f1[i]) for i in np.argsort(-f1)[:3]], worst_f1=[float(f1[i]) for i in np.argsort(f1)[:3]])
json.dump(res, open(R + "alnum_report.json", "w"), indent=1)

# per-class F1
plt.figure(figsize=(9, 3.2)); plt.bar(range(36), f1 * 100, color=[BLUE] * 10 + [ORANGE] * 26)
plt.xticks(range(36), list(L), fontsize=8); plt.ylim(0, 100); plt.ylabel("F1-score (%)")
plt.tight_layout(); plt.savefig(R + "alnum_f1.png", dpi=200); plt.close()

# normalised confusion matrix
plt.figure(figsize=(6.2, 5.4)); plt.imshow(cm / cm.sum(1, keepdims=True), cmap="Blues", vmin=0, vmax=1)
plt.xticks(range(36), list(L), fontsize=6); plt.yticks(range(36), list(L), fontsize=6)
plt.xlabel("Predicted"); plt.ylabel("True"); plt.colorbar(fraction=0.046, label="fraction of class")
plt.tight_layout(); plt.savefig(R + "alnum_confusion.png", dpi=200); plt.close()

# training curves
e = range(1, len(h["loss"]) + 1)
fig, ax = plt.subplots(1, 2, figsize=(8, 3)); ax[0].plot(e, h["accuracy"], color=BLUE, label="Train"); ax[0].plot(e, h["val_accuracy"], color=ORANGE, label="Validation")
ax[0].set_title("Accuracy"); ax[0].set_xlabel("Epoch"); ax[0].legend(); ax[1].plot(e, h["loss"], color=BLUE); ax[1].plot(e, h["val_loss"], color=ORANGE)
ax[1].set_title("Loss"); ax[1].set_xlabel("Epoch"); plt.tight_layout(); plt.savefig(R + "alnum_curves.png", dpi=200); plt.close()

# sample predictions incl. mistakes
wrong = np.where(pred != y)[0]
d = np.load(os.path.join(ROOT, "data", "emnist36.npz"))["x_test"]
fig, ax = plt.subplots(2, 8, figsize=(8, 2.4))
for a, i in zip(ax.ravel(), np.random.RandomState(1).choice(wrong, 16, replace=False)):
    a.imshow(d[i], cmap="gray_r"); a.set_title(f"T:{L[y[i]]} P:{L[pred[i]]}", fontsize=8); a.axis("off")
plt.tight_layout(); plt.savefig(R + "alnum_errors.png", dpi=200); plt.close()
print(json.dumps({k: v for k, v in res.items() if k != "top_pairs"}, indent=1)); print(pairs)
