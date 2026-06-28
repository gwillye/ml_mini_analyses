"""02 — Regression: Diabetes progression (real dataset, bundled with scikit-learn).

Kaggle-ready: swap the loader for a downloaded Kaggle CSV (see README).
"""
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import load_diabetes
from sklearn.linear_model import Ridge
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error

SEED = 42
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "outputs")
os.makedirs(OUT, exist_ok=True)

X, y = load_diabetes(return_X_y=True)
Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.25, random_state=SEED)

results = {}
for name, model in [("Ridge", Ridge(alpha=1.0)),
                    ("GradientBoosting", GradientBoostingRegressor(random_state=SEED))]:
    model.fit(Xtr, ytr)
    pred = model.predict(Xte)
    results[name] = (r2_score(yte, pred), mean_squared_error(yte, pred) ** 0.5)

best = max(results, key=lambda k: results[k][0])
for n, (r2, rmse) in results.items():
    print(f"{n:18} R2={r2:.3f}  RMSE={rmse:.1f}")

pred = (Ridge(alpha=1.0).fit(Xtr, ytr)).predict(Xte)
plt.figure(figsize=(5, 5))
plt.scatter(yte, pred, alpha=0.6, color="#54a24b")
lims = [min(yte.min(), pred.min()), max(yte.max(), pred.max())]
plt.plot(lims, lims, "--", color="gray")
plt.xlabel("actual")
plt.ylabel("predicted")
plt.title("Diabetes — Ridge: actual vs predicted")
plt.tight_layout()
plt.savefig(os.path.join(OUT, "02_diabetes_pred.png"), dpi=110)
plt.close()

print(f"best = {best} (R2={results[best][0]:.3f})")
print("self-check:", "OK" if results[best][0] > 0.3 else "FAIL")
