"""04 — Classification + Regression: Wine Quality (UCI red wine, downloaded, no key).

Predicts wine quality from physicochemical features: a regression (exact score) and a
classification (good >= 7). Kaggle-ready: swap the URL/loader for a Kaggle CSV (see README).
"""
import os
import io
import numpy as np
import pandas as pd
import requests
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error, roc_auc_score

SEED = 42
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "outputs")
os.makedirs(OUT, exist_ok=True)
URL = "https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv"

df = pd.read_csv(io.StringIO(requests.get(URL, timeout=30).text), sep=";")
X = df.drop(columns="quality")
y = df["quality"]
print(f"loaded {len(df)} rows, {X.shape[1]} features")

# regression
Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.25, random_state=SEED)
reg = RandomForestRegressor(n_estimators=300, random_state=SEED).fit(Xtr, ytr)
pred = reg.predict(Xte)
r2 = r2_score(yte, pred)
rmse = mean_squared_error(yte, pred) ** 0.5

# classification: "good" wine (quality >= 7)
yb = (y >= 7).astype(int)
Xtr, Xte, ytr, yte = train_test_split(X, yb, test_size=0.25, random_state=SEED, stratify=yb)
clf = RandomForestClassifier(n_estimators=300, random_state=SEED, class_weight="balanced").fit(Xtr, ytr)
auc = roc_auc_score(yte, clf.predict_proba(Xte)[:, 1])

imp = clf.feature_importances_
idx = np.argsort(imp)
plt.figure(figsize=(7, 5))
plt.barh(np.array(X.columns)[idx], imp[idx], color="#722f37")
plt.title(f"Wine quality — drivers (good-wine ROC-AUC {auc:.3f})")
plt.tight_layout()
plt.savefig(os.path.join(OUT, "04_wine_quality_features.png"), dpi=110)
plt.close()

print(f"regression R2={r2:.3f} RMSE={rmse:.3f} | good-wine ROC-AUC={auc:.3f}")
print("self-check:", "OK" if (r2 > 0.3 and auc > 0.8) else "FAIL")
