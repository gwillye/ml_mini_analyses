# 04 — Wine Quality: Regression + Classification (Random Forest)

**Headline result: regression R² = 0.497 (RMSE 0.558 quality points); "good wine" classifier ROC-AUC = 0.947.**

## Problem
The **UCI Wine Quality (red)** dataset has **1,599** Portuguese *vinho verde* red wines, each
with 11 physicochemical measurements (fixed/volatile acidity, citric acid, residual sugar,
chlorides, free/total SO₂, density, pH, sulphates, alcohol) and a sensory **quality score
0–10** assigned by tasters (in practice 3–8).

Two business-relevant questions, framed two ways:
1. **Regression:** can we predict the *exact* tasting score from chemistry?
2. **Classification:** can we flag **"good" wines (quality ≥ 7)** — a more actionable yes/no
   for quality control? The good-wine class is rare (~14%), so this is also an imbalance test.

This matters for any product-quality or QC pipeline: predicting a subjective human score from
objective sensors, and ranking which chemical levers matter.

## Approach
- **Two Random Forests** (300 trees each, `random_state=42`):
  - `RandomForestRegressor` on the raw 0–10 score.
  - `RandomForestClassifier` with **`class_weight="balanced"`** on the binary `quality ≥ 7`
    target — the balancing is essential because only ~14% of wines are "good," and an
    unweighted classifier would lazily predict "not good" for everything.
- **Why Random Forest:** the feature–quality relationships are non-linear and interacting
  (e.g. alcohol matters more at certain acidity levels), trees handle that natively, need no
  scaling, and give a feature-importance ranking of the quality drivers.
- **Preprocessing:** none beyond the binary target derivation; no missing values.
- **Validation:** 75/25 split for each task; the classification split is **stratified** on the
  rare class. Metrics: **R² + RMSE** for regression; **ROC-AUC** for classification (the
  threshold-free metric that is honest under imbalance).

## Results & analysis
Running `py 04_wine_quality.py`:

```
loaded 1599 rows, 11 features
regression R2=0.497 RMSE=0.558 | good-wine ROC-AUC=0.947
```

The feature-importance bar chart (good-wine classifier) is in
[`outputs/04_wine_quality_features.png`](../outputs/04_wine_quality_features.png), where
**alcohol, sulphates, and volatile acidity** dominate the ranking.

**Interpretation.** Note the **gap between the two framings** — it is the whole story:
- **Regression R² = 0.497** with RMSE = **0.558**. The model explains ~50% of score variance
  and is typically within ~0.56 of the true score. Since quality is integer and tasters
  themselves disagree by ±1, an RMSE around half a point is essentially **at the noise floor of
  human subjectivity** — you cannot reliably predict the exact score because the *label itself*
  is noisy.
- **Classification ROC-AUC = 0.947.** Yet the *easier, more useful* question — "is this a good
  wine?" — is answered almost perfectly. Collapsing the noisy 0–10 score into a robust binary
  threshold removes most of the label noise and exposes a clean separable signal.

So the takeaway-in-action: **predicting the exact subjective score is hard (R²≈0.5) but
flagging good-vs-rest is easy (AUC≈0.95)** — choosing the right target framing matters more
than the model. And the drivers are interpretable and oenologically sensible: higher **alcohol**
and **sulphates** push quality up, higher **volatile acidity** (vinegar taint) pushes it down.

**Limitations.** (1) Red wine only — the white-wine file is separate and behaves differently.
(2) "Quality" is the median of ≥3 tasters; the model inherits their bias. (3) `class_weight`
handles imbalance during training, but for deployment you would still tune the operating
threshold to your tolerated false-positive rate. (4) Single split — k-fold would tighten both
estimates.

**Takeaway:** the same data and same model give a mediocre or an excellent result depending
**entirely on how you frame the target** — bin the noisy score and the problem becomes easy.

## How to run
```bash
pip install -r requirements.txt
py 04_wine_quality.py     # downloads the UCI CSV over HTTPS — no key required
```
Swap the `URL`/loader for a Kaggle CSV (Wine Quality is on Kaggle) to run offline (see README).
