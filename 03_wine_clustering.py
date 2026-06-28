"""03 — Clustering: Wine recognition (real dataset, bundled with scikit-learn).

Unsupervised KMeans; we score it against the (held-out) true cultivar labels with ARI.
Kaggle-ready: swap the loader for a downloaded Kaggle CSV (see README).
"""
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.datasets import load_wine
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, adjusted_rand_score

SEED = 42
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "outputs")
os.makedirs(OUT, exist_ok=True)

data = load_wine()
X = StandardScaler().fit_transform(data.data)
y_true = data.target

km = KMeans(n_clusters=3, n_init=10, random_state=SEED).fit(X)
labels = km.labels_
sil = silhouette_score(X, labels)
ari = adjusted_rand_score(y_true, labels)

Z = PCA(n_components=2, random_state=SEED).fit_transform(X)
plt.figure(figsize=(6, 5))
plt.scatter(Z[:, 0], Z[:, 1], c=labels, cmap="viridis", s=25)
plt.title(f"Wine KMeans (k=3) — silhouette {sil:.2f}, ARI {ari:.2f}")
plt.xlabel("PC1")
plt.ylabel("PC2")
plt.tight_layout()
plt.savefig(os.path.join(OUT, "03_wine_clusters.png"), dpi=110)
plt.close()

print(f"silhouette={sil:.3f}  ARI_vs_true={ari:.3f}")
print("self-check:", "OK" if sil > 0.2 and ari > 0.5 else "FAIL")
