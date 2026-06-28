"""10 — Multiclass classification + viz: Iris (real dataset, bundled with scikit-learn).

Logistic regression on all 4 features; PCA 2D scatter with class colors.
Kaggle-ready: swap the loader for a Kaggle CSV (see README).
"""
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report

SEED = 42
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "outputs")
os.makedirs(OUT, exist_ok=True)

data = load_iris()
X = StandardScaler().fit_transform(data.data)
y = data.target

cv = cross_val_score(LogisticRegression(max_iter=500), X, y, cv=5).mean()
Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.3, random_state=SEED, stratify=y)
clf = LogisticRegression(max_iter=500).fit(Xtr, ytr)
acc = accuracy_score(yte, clf.predict(Xte))
print(classification_report(yte, clf.predict(Xte), target_names=data.target_names, digits=3))

Z = PCA(n_components=2, random_state=SEED).fit_transform(X)
plt.figure(figsize=(6, 5))
for c, name in enumerate(data.target_names):
    m = y == c
    plt.scatter(Z[m, 0], Z[m, 1], label=name, s=25)
plt.legend()
plt.title(f"Iris — PCA projection (5-fold CV acc {cv:.3f})")
plt.xlabel("PC1")
plt.ylabel("PC2")
plt.tight_layout()
plt.savefig(os.path.join(OUT, "10_iris_pca.png"), dpi=110)
plt.close()

print(f"CV accuracy={cv:.3f}  holdout accuracy={acc:.3f}")
print("self-check:", "OK" if cv > 0.9 else "FAIL")
