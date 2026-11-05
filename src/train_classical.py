"""Baseline classical ML models. Saves results/classical_results.json."""
import json, time, joblib
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, f1_score
from data import load_mnist, flatten

(xtr, ytr), (xva, yva), (xte, yte) = load_mnist()
xtr, xva, xte = flatten(xtr), flatten(xva), flatten(xte)

models = {
    "Logistic Regression": make_pipeline(StandardScaler(), LogisticRegression(max_iter=200)),
    "Random Forest": RandomForestClassifier(n_estimators=200, n_jobs=-1, random_state=0),
    "k-NN (k=3, PCA-50)": make_pipeline(PCA(50, random_state=0), KNeighborsClassifier(3, weights="distance")),
    "SVM RBF (PCA-50)": make_pipeline(PCA(50, random_state=0), SVC(C=10, gamma="scale")),
}
out = {}
for name, m in models.items():
    t = time.time(); m.fit(xtr, ytr); tt = time.time() - t
    t = time.time(); p = m.predict(xte); ti = time.time() - t
    out[name] = dict(accuracy=accuracy_score(yte, p), macro_f1=f1_score(yte, p, average="macro"),
                     train_s=tt, infer_s=ti, val_accuracy=accuracy_score(yva, m.predict(xva)))
    print(f"{name:22s} test acc {out[name]['accuracy']:.4f}  train {tt:.0f}s", flush=True)
    if name.startswith("SVM"):
        joblib.dump(m, "../models/svm.joblib")
json.dump(out, open("../results/classical_results.json", "w"), indent=2)
