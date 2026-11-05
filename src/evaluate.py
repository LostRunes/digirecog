"""Create figures + metrics from saved results (confusion matrix, curves, errors...)."""
import os
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
import json, numpy as np, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt, seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
from data import load_mnist

R = os.path.join(ROOT, "results") + os.sep
(_, ytr), _, (xte, yte) = load_mnist()
probs = np.load(R + "cnn_test_probs.npy"); pred = probs.argmax(1)
h = json.load(open(R + "cnn_history.json")); cl = json.load(open(R + "classical_results.json"))
plt.rcParams.update({"font.family": "DejaVu Sans", "axes.spines.top": False, "axes.spines.right": False})
BLUE, ORANGE, GREY = "#2f5d7c", "#c0623b", "#9aa7b3"

# dataset samples
fig, ax = plt.subplots(2, 10, figsize=(12, 2.8))
for d in range(10):
    for r in range(2):
        ax[r, d].imshow(xte[np.where(yte == d)[0][r]], cmap="gray_r"); ax[r, d].axis("off")
plt.tight_layout(); plt.savefig(R + "samples.png", dpi=200); plt.close()

# class distribution
plt.figure(figsize=(6, 3.2)); c = np.bincount(ytr)
plt.bar(range(10), c, color=BLUE); plt.xticks(range(10)); plt.xlabel("Digit"); plt.ylabel("Training images")
plt.tight_layout(); plt.savefig(R + "class_dist.png", dpi=200); plt.close()

# training curves
fig, ax = plt.subplots(1, 2, figsize=(9, 3.4)); e = range(1, len(h["loss"]) + 1)
ax[0].plot(e, h["accuracy"], label="Train", color=BLUE); ax[0].plot(e, h["val_accuracy"], label="Validation", color=ORANGE)
ax[0].set_ylim(0.90, 1.0); ax[0].set_title("Accuracy"); ax[0].set_xlabel("Epoch"); ax[0].legend()
ax[1].plot(e, h["loss"], color=BLUE); ax[1].plot(e, h["val_loss"], color=ORANGE); ax[1].set_title("Loss"); ax[1].set_xlabel("Epoch")
plt.tight_layout(); plt.savefig(R + "training_curves.png", dpi=200); plt.close()

# confusion matrix
cm = confusion_matrix(yte, pred)
plt.figure(figsize=(5.4, 4.6)); sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False, annot_kws={"size": 8})
plt.xlabel("Predicted"); plt.ylabel("True"); plt.tight_layout(); plt.savefig(R + "confusion_matrix.png", dpi=200); plt.close()

# model comparison
names = list(cl) + ["CNN (proposed)"]; accs = [cl[n]["accuracy"] * 100 for n in cl] + [h["test_acc"] * 100]
plt.figure(figsize=(7, 3.6)); cols = [GREY] * len(cl) + [ORANGE]
b = plt.barh(names, accs, color=cols); plt.xlim(90, 100.4); plt.xlabel("Test accuracy (%)")
for bar, a in zip(b, accs): plt.text(a + 0.05, bar.get_y() + bar.get_height() / 2, f"{a:.2f}", va="center", fontsize=9)
plt.gca().invert_yaxis(); plt.tight_layout(); plt.savefig(R + "model_comparison.png", dpi=200); plt.close()

# misclassified
wrong = np.where(pred != yte)[0][:12]
fig, ax = plt.subplots(2, 6, figsize=(8, 3.2))
for a, i in zip(ax.ravel(), wrong):
    a.imshow(xte[i], cmap="gray_r"); a.set_title(f"True:{yte[i]} Pred:{pred[i]}", fontsize=9); a.axis("off")
plt.tight_layout(); plt.savefig(R + "misclassified.png", dpi=200); plt.close()

rep = classification_report(yte, pred, digits=4, output_dict=True)
json.dump({"report": rep, "errors": int((pred != yte).sum()), "confusion": cm.tolist()}, open(R + "cnn_report.json", "w"), indent=1)
print(classification_report(yte, pred, digits=4)); print("errors:", (pred != yte).sum())
