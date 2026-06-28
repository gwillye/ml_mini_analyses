# 09 — Credit-Risk Classification: German Credit / Statlog (Random Forest)

**Headline result: ROC-AUC = 0.804 for predicting bad credit (30% base bad-rate, 1,000 applicants).**

## Problem
The **Statlog German Credit** dataset (UCI, numeric variant) has **1,000 loan applicants**, each
described by **24 numeric features** (the numeric encoding of attributes like account status,
credit duration, amount, employment, age, existing credits…). Each applicant is labelled **good**
or **bad** credit risk. The classes are imbalanced: **30% are bad-credit** (the positive class
we care about).

The question is a core finance / risk problem (the Neoway domain): **can we score applicants by
default risk from their application features, and rank the riskiest reliably?** In lending, the
cost of a missed *bad* (a default) far exceeds a false alarm, so the model's job is **risk
ranking**, not just accuracy.

## Approach
- **Model: Random Forest** (400 trees, `random_state=42`, **`class_weight="balanced"`**). The
  balancing is essential — with a 30% positive rate, an unweighted model is rewarded for
  ignoring bad-credit cases; `class_weight="balanced"` re-weights the rare class so the model
  actually learns to flag defaults. RF is chosen for mixed-scale tabular features, native
  handling of non-linear interactions, and a feature-importance view.
- **Preprocessing:** the numeric Statlog variant is already encoded; we load it as a float
  matrix. The positive class is defined as **label == 2 (bad credit)**.
- **Validation:** **stratified** 75/25 split (preserves the 30% bad-rate in both halves).
  Metric: **ROC-AUC** — the right choice under imbalance because it measures *ranking* quality
  independent of any threshold, plus a full precision/recall report at the default cut.

## Results & analysis
Running `py 09_credit_risk_classification.py`:

```
loaded 1000 rows, 24 features, bad-rate 30.0%
              precision    recall  f1-score   support
           0      0.839     0.834     0.837       175   (good credit)
           1      0.618     0.627     0.623        75   (bad credit)
    accuracy                          0.772       250
ROC-AUC = 0.804
```

The top-10 feature-importance chart is in
[`outputs/09_credit_features.png`](../outputs/09_credit_features.png). Note features are shown
as `f0…f23` — the **numeric Statlog variant is anonymized**, so importances are positional, not
human-labelled (a known limitation called out below).

**Interpretation.** **ROC-AUC = 0.804** means that, given a random bad applicant and a random
good one, the model assigns the bad one a higher risk score **~80% of the time** — solid, and in
line with the literature for this hard dataset (German Credit has a low ceiling; real default
behaviour is only partly predictable from the application). This is a usable risk-ranking model.

The per-class report exposes the real challenge, and it is **not** the 0.772 accuracy (which is
misleading here — predicting "all good" would already score 70%). It's the **bad-credit recall
of 0.627**: at the default 0.5 threshold the model **still misses ~37% of true defaulters**. In
lending, those misses are the expensive errors. The fix is *not* a better model but a **threshold
move**: because AUC is a healthy 0.80, lowering the decision threshold raises bad-recall (catches
more defaulters) at the cost of rejecting some good applicants — and the **right** trade-off point
is set by the *cost ratio of a default vs. a lost good loan*, a business decision the AUC enables.

**What worked:** `class_weight="balanced"` is why bad-credit recall (0.627) is in a usable range
at all rather than collapsing toward zero — the single most important line for this imbalanced
target.

**Limitations.** (1) **Anonymized numeric features** → no human-readable driver story; the
categorical `german.data` variant would let us name the risk factors (e.g. checking-account
status, credit duration), which is what a credit team needs. (2) 1,000 rows is small for credit
modelling; estimates are noisy (single split — k-fold advised). (3) The default threshold is
arbitrary; deployment requires explicit cost-based threshold tuning and a calibration check
(reliability of the probabilities). (4) No fairness audit — credit models must be checked for
disparate impact before use.

**Takeaway:** the model **ranks default risk well (AUC 0.80)**, but the operational lever is the
**decision threshold, driven by the cost of a default** — accuracy is the wrong yardstick for a
30%-imbalanced lending problem.

## How to run
```bash
pip install -r requirements.txt
py 09_credit_risk_classification.py   # downloads the UCI Statlog numeric file — no key required
```
German Credit is also on Kaggle — point the loader at the Kaggle CSV (use the *categorical*
version if you want named features) to run offline (see top-level README).
