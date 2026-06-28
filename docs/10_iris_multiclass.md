# 10 — Multiclass Classification + Viz: Iris (Logistic Regression + PCA)

**Headline result: 5-fold CV accuracy = 0.960 (hold-out accuracy 0.911) across 3 species.**

## Problem
The **Iris** dataset is the *Hello, World* of classification: **150 flowers**, **3 species**
(setosa, versicolor, virginica, 50 each — perfectly balanced), described by 4 measurements
(sepal length/width, petal length/width).

The question: **can a simple linear classifier separate the three species, and how separable are
they geometrically?** Iris is the canonical sanity-check pipeline — its value here is
demonstrating a *clean, correct* multiclass workflow (proper CV + a 2-D visualization that
explains the result) rather than a hard problem.

## Approach
- **Standardize** the 4 features (`StandardScaler`) so each contributes comparably to both the
  classifier and the PCA — distance/coefficient-based methods are scale-sensitive.
- **Model: multinomial Logistic Regression** (`max_iter=500`). A linear model is deliberately
  chosen: if a plain linear boundary already nails it, nothing more complex is warranted — and
  it keeps the decision interpretable.
- **PCA(2) for visualization.** The 4-D data is projected to 2 components purely to *see* the
  class structure; classification uses all 4 standardized features.
- **Validation: two complementary checks.**
  - **5-fold cross-validation** on all 150 rows — the headline metric, because on a 150-row
    dataset a single split is high-variance, and CV averages over five folds for a stable
    estimate.
  - A **stratified 70/30 hold-out** for a per-class precision/recall report and confusion view.
  Metric: **accuracy** (fair — classes are exactly balanced).

## Results & analysis
Running `py 10_iris_multiclass.py`:

```
              precision    recall  f1-score   support
      setosa      1.000     1.000     1.000        15
  versicolor      0.824     0.933     0.875        15
   virginica      0.923     0.800     0.857        15
    accuracy                          0.911        45
CV accuracy=0.960  holdout accuracy=0.911
```

The PCA projection coloured by species is in
[`outputs/10_iris_pca.png`](../outputs/10_iris_pca.png).

**Interpretation.** Report the **CV number (0.960)** as the headline, not the hold-out (0.911):
with only 45 test flowers, the hold-out swings by whole-percent steps per misclassification, so
it is noisy — the 5-fold average is the trustworthy estimate. The model is **excellent**, and the
*structure* of the result is the lesson:
- **setosa: perfect (F1 = 1.000).** The PCA plot shows why — setosa sits as a completely isolated
  cluster, **linearly separable** from the other two. No model could do worse than perfect here.
- **versicolor ↔ virginica: every error lives here.** They form two **adjacent, slightly
  overlapping** clouds in PCA space. The hold-out confusion is symmetric and tiny — versicolor
  recall 0.933, virginica recall 0.800 (i.e. a few virginica flowers near the boundary get called
  versicolor). This is a genuine *biological* overlap, not a model defect.

So the picture and the metrics agree: **one species is trivially separable, the other two share a
fuzzy boundary** — and that boundary region is the entire residual error.

**What worked:** a linear model is the *correct* level of complexity. CV 0.96 with full
interpretability means a heavier model (SVM/RF) would add nothing but opacity here — a useful
discipline reminder.

**Limitations.** (1) 150 rows — small enough that even CV carries a confidence interval; treat
0.96 as "excellent," not as four-significant-figure truth. (2) Iris is *too* easy to be evidence
of method quality on hard data; it validates the *pipeline*, not the modelling chops. (3) The
versicolor/virginica overlap is irreducible with these 4 features — it is a property of the
flowers, not something tuning can remove.

**Takeaway:** **match model complexity to the problem** — a standardized linear classifier hits
~96% CV on Iris with a fully interpretable, picture-backed result; the only errors are the one
genuinely overlapping species pair.

## How to run
```bash
pip install -r requirements.txt
py 10_iris_multiclass.py
```
Dataset ships with scikit-learn — **no download, no key**. Swap `load_iris()` for a Kaggle CSV
loader to reuse the standardize → CV → PCA-plot template (see top-level README).
