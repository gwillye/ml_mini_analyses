"""01 — Classification: Breast Cancer Wisconsin (real dataset, bundled with scikit-learn).

Kaggle-ready: to use a Kaggle dataset instead, install the kaggle CLI, set credentials,
and uncomment the download line below, then point the loader at the CSV.
    # os.system("kaggle datasets download -d uciml/breast-cancer-wisconsin-data")
"""
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, classification_report

SEED = 42
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "outputs")
os.makedirs(OUT, exist_ok=True)

data = load_breast_cancer()
X, y = data.data, data.target
Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.25, random_state=SEED, stratify=y)

clf = RandomForestClassifier(n_estimators=300, random_state=SEED).fit(Xtr, ytr)
proba = clf.predict_proba(Xte)[:, 1]
auc = roc_auc_score(yte, proba)
print(classification_report(yte, clf.predict(Xte), digits=3))

imp = clf.feature_importances_
idx = np.argsort(imp)[-10:]
plt.figure(figsize=(7, 5))
plt.barh(np.array(data.feature_names)[idx], imp[idx], color="#4c78a8")
plt.title(f"Breast cancer — RF top features (ROC-AUC {auc:.3f})")
plt.tight_layout()
plt.savefig(os.path.join(OUT, "01_breast_cancer_features.png"), dpi=110)
plt.close()

print(f"ROC-AUC = {auc:.3f}")
print("self-check:", "OK" if auc > 0.95 else "FAIL")
