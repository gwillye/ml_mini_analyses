# 09 - Credit-Risk Classification: German Credit / Statlog (Random Forest)

Headline result: ROC-AUC = 0.804 for predicting bad credit on a 30% base bad-rate across 1,000 applicants.

## Problem

The Statlog German Credit dataset (UCI, numeric variant) has 1,000 loan applicants. Each one is described by 24 numeric features, which are the numeric encoding of attributes like account status, credit duration, amount, employment, and age. Every applicant is labelled as either good or bad credit risk, and the classes are imbalanced: 30% are bad-credit, and that is the positive class we actually care about.

This is a core finance and risk problem. Can we score applicants by default risk from their application features and rank the riskiest ones reliably? In lending, the cost of a missed bad case (a default) is far higher than a false alarm, so the model's real job is risk ranking, not just accuracy.

## Approach

The model is a Random Forest with 400 trees, `random_state=42`, and `class_weight="balanced"`. That balancing matters a lot here. With a 30% positive rate, an unweighted model gets rewarded for ignoring bad-credit cases, so `class_weight="balanced"` re-weights the rare class and pushes the model to actually learn to flag defaults. Random Forest is a good fit for mixed-scale tabular features, it handles non-linear interactions natively, and it gives a feature-importance view for free.

For preprocessing, the numeric Statlog variant is already encoded, so we load it as a float matrix. The positive class is defined as label == 2 (bad credit).

Validation uses a stratified 75/25 split, which preserves the 30% bad-rate in both halves. The headline metric is ROC-AUC, which is the right choice under imbalance because it measures ranking quality independent of any threshold. We also report a full precision/recall breakdown at the default cut.

## Results and analysis

Running `py 09_credit_risk_classification.py`:

```
loaded 1000 rows, 24 features, bad-rate 30.0%
              precision    recall  f1-score   support
           0      0.839     0.834     0.837       175   (good credit)
           1      0.618     0.627     0.623        75   (bad credit)
    accuracy                          0.772       250
ROC-AUC = 0.804
```

The top-10 feature-importance chart is in [`outputs/09_credit_features.png`](../outputs/09_credit_features.png). The features show up as `f0` through `f23` because the numeric Statlog variant is anonymized, so the importances are positional rather than human-labelled. That is a known limitation, called out below.

A ROC-AUC of 0.804 means that, given a random bad applicant and a random good one, the model assigns the bad one a higher risk score about 80% of the time. That is solid and in line with the literature for this hard dataset (German Credit has a low ceiling, since real default behaviour is only partly predictable from the application). This is a usable risk-ranking model.

The per-class report exposes the real challenge, and it is not the 0.772 accuracy, which is misleading here (predicting "all good" would already score 70%). The real number is the bad-credit recall of 0.627: at the default 0.5 threshold, the model still misses about 37% of true defaulters. In lending, those misses are the expensive errors. The fix is not a better model but a threshold move. Because AUC is a healthy 0.80, lowering the decision threshold raises bad-recall (catches more defaulters) at the cost of rejecting some good applicants, and the right trade-off point is set by the cost ratio of a default versus a lost good loan, which is a business decision the AUC enables.

What worked: `class_weight="balanced"` is the reason bad-credit recall (0.627) lands in a usable range at all, instead of collapsing toward zero. It is the single most important line for this imbalanced target.

A few limitations are worth stating plainly. First, the anonymized numeric features mean there is no human-readable driver story. The categorical `german.data` variant would let us name the risk factors (checking-account status, credit duration, and so on), which is exactly what a credit team needs. Second, 1,000 rows is small for credit modelling, so estimates are noisy (this is a single split, and k-fold is advised). Third, the default threshold is arbitrary, and deployment requires explicit cost-based threshold tuning plus a calibration check on the reliability of the probabilities. Fourth, there is no fairness audit, and credit models must be checked for disparate impact before use.

Takeaway: the model ranks default risk well (AUC 0.80), but the operational lever is the decision threshold, driven by the cost of a default. Accuracy is the wrong yardstick for a 30%-imbalanced lending problem.

## How to run

```bash
pip install -r requirements.txt
py 09_credit_risk_classification.py   # downloads the UCI Statlog numeric file, no key required
```

German Credit is also on Kaggle, so you can point the loader at the Kaggle CSV (use the categorical version if you want named features) to run offline. See the top-level README.
