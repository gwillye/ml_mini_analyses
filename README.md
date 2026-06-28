# Kaggle-Analytics — end-to-end analyses on public datasets

A portfolio of compact, **reproducible** data-science analyses across distinct problem types — classification, regression, clustering, NLP, time series and recommendation — each on a **real public dataset that needs no Kaggle API key**, and each structured so a Kaggle dataset can be dropped in later.

> On-brand for Data Science / marketing / finance / government use cases. Every script is **self-contained and verified by running** (prints a metric + self-check, saves a plot to `outputs/`).

## Analyses
| # | Problem | Dataset | Source | Status | Headline |
|---|---|---|---|---|---|
| 01 | Classification | Breast Cancer Wisconsin | scikit-learn | ✅ | RF ROC-AUC **0.995** |
| 02 | Regression | Diabetes progression | scikit-learn | ✅ | Ridge R² **0.44** |
| 03 | Clustering | Wine cultivars | scikit-learn | ✅ | KMeans ARI **0.90** |
| 04 | Classification + Regression | Wine Quality (red) | UCI (download) | ✅ | RF AUC **0.95**, R² **0.50** |
| 05 | NLP text classification | 20 Newsgroups (4 topics) | sklearn fetch | ✅ | TF-IDF+LogReg acc **0.89** |
| 06 | Time series | market / classic series | yfinance / URL | ⏳ | — |
| 07 | Recommendation | MovieLens 100k | GroupLens (download) | ⏳ | — |
| 08 | EDA / open gov data | Brazil (gov.br / IBGE) | download | ⏳ | — |
| 09 | Finance | B3 tickers | Yahoo Finance | ⏳ | — |
| 10 | Multiclass + viz | Iris | scikit-learn | ✅ | LogReg CV acc **0.96** |

## Kaggle-ready
Each script runs on a public dataset with **no key required**. To swap in a Kaggle dataset:
1. `pip install kaggle`; place `kaggle.json` (or set `KAGGLE_USERNAME` / `KAGGLE_KEY` in `.env`).
2. Uncomment the `kaggle datasets download -d <slug>` line at the top of the script.
3. Point the loader at the downloaded CSV.

## Run
```bash
pip install -r requirements.txt
python 01_breast_cancer_classification.py   # 02_..., 03_..., etc.
```
Outputs (plots) are written to `outputs/`.

## Stack
Python · scikit-learn · pandas · NumPy · matplotlib
