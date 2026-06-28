# 02 — Diabetes Progression Regression (Ridge vs Gradient Boosting)

**Headline result: Ridge R² = 0.438, RMSE = 55.7 (best of two models) on a held-out 25% test set.**

## Problem
The scikit-learn **Diabetes** dataset has 442 patients, each described by 10 baseline
features (age, sex, BMI, mean blood pressure, and six blood-serum measurements). All features
are mean-centred and scaled to unit L2-norm. The target is a **continuous disease-progression
score measured one year after baseline** (a quantitative index of disease severity, roughly
on a 25–346 scale).

The question: **how much of a patient's one-year disease progression can be predicted from
cheap baseline measurements, and is the relationship linear or does it need a non-linear
model?** This is a classic "small-n, modest-signal" medical regression — the interesting part
is honestly quantifying *how much* is predictable, not chasing a big number.

## Approach
We deliberately compare a **linear** and a **non-linear** model to answer the "is non-linearity
worth it?" question directly:
- **Ridge regression (α = 1.0)** — linear with L2 regularization. Chosen because the dataset
  is small (442 rows, 10 features) and the features are correlated; the L2 penalty shrinks
  coefficients and stabilizes the fit against that collinearity.
- **Gradient Boosting Regressor** (default hyper-parameters, `random_state=42`) — a flexible
  non-linear ensemble, included as a sanity check on whether tree-based interactions buy
  anything over a plain linear fit.
- **Preprocessing:** none — features are already centred and scaled by the dataset provider.
- **Validation:** 75/25 train-test split. We report **R²** (fraction of target variance
  explained) and **RMSE** (error in the target's own units). The script auto-selects the
  better model by test R² and plots actual-vs-predicted for Ridge.

## Results & analysis
Running `py 02_diabetes_regression.py`:

```
Ridge              R2=0.438  RMSE=55.7
GradientBoosting   R2=0.424  RMSE=56.4
best = Ridge (R2=0.438)
```

The actual-vs-predicted scatter is in
[`outputs/02_diabetes_pred.png`](../outputs/02_diabetes_pred.png): points cluster around the
diagonal but with wide vertical scatter, and the model visibly *compresses the range* — it
under-predicts the highest-progression patients and over-predicts the lowest.

**Interpretation.** R² = **0.438** means the baseline measurements explain roughly **44% of
the variance** in one-year progression — a genuinely useful but far-from-deterministic signal.
RMSE = **55.7** on a target ranging ~25–346 says a typical prediction is off by ~56 units, so
this is a model for *population-level risk stratification*, not individual prognosis.

The headline finding is the **near-tie**: the linear Ridge model (0.438) actually *edges out*
Gradient Boosting (0.424). That is the analytically interesting result — with only 442 rows,
the extra flexibility of a boosted ensemble does not pay off and even slightly overfits;
**the underlying signal is essentially linear at this sample size.** Occam wins.

**Limitations.** (1) The remaining ~56% of variance is genuinely unexplained by these 10
baseline features — likely reflecting real biological/lifestyle factors not measured, so this
is close to the achievable ceiling, not a tuning failure. (2) The range-compression seen in
the plot is the expected behaviour of a regularized linear model on a noisy target;
predictions for extreme patients should be treated with caution. (3) A single split; k-fold CV
would tighten the estimate.

**Takeaway:** when data is small and the signal is moderate, a **regularized linear model
matches a flexible ensemble** — added model complexity is not free, and here it bought
nothing. Report the honest 44%, don't oversell it.

## How to run
```bash
pip install -r requirements.txt
py 02_diabetes_regression.py
```
Dataset is bundled with scikit-learn — **no download, no key**. To use a Kaggle CSV, replace
`load_diabetes(...)` with a pandas load of the downloaded file (see top-level README).
