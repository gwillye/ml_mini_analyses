# Kaggle-Analytics — end-to-end analyses on public datasets

A portfolio of compact, **reproducible** data-science analyses across distinct problem types —
classification, regression, clustering, NLP, time series, recommendation and open-data EDA —
each on a **real public dataset that needs no Kaggle API key**, and each structured so a Kaggle
dataset can be dropped in later.

Every script is **self-contained and verified by running** (prints a metric + a self-check,
saves a plot to `outputs/`). **All metrics in this repo are produced by actually running the
scripts — none are invented.**

> Each analysis has a full write-up in [`docs/`](docs/) — a mini-article covering the
> **problem**, the **approach (and why)**, the **real results with interpretation**, and how to
> run it. Click a row's write-up link below to read it.

## Analyses

| # | Problem type | Dataset | Source | Headline result (real run) | Write-up |
|---|---|---|---|---|---|
| 01 | Classification | Breast Cancer Wisconsin | scikit-learn (bundled) | RF **ROC-AUC 0.995**, acc 0.958 | [docs/01](docs/01_breast_cancer_classification.md) |
| 02 | Regression | Diabetes progression | scikit-learn (bundled) | Ridge **R² 0.438** (beats GBoost) | [docs/02](docs/02_diabetes_regression.md) |
| 03 | Clustering (unsupervised) | Wine cultivars | scikit-learn (bundled) | KMeans **ARI 0.897**, silhouette 0.285 | [docs/03](docs/03_wine_clustering.md) |
| 04 | Regression + Classification | Wine Quality (red) | UCI (HTTPS download) | reg **R² 0.497**; good-wine **AUC 0.947** | [docs/04](docs/04_wine_quality.md) |
| 05 | NLP text classification | 20 Newsgroups (4 topics) | sklearn fetch (cached) | TF-IDF+LogReg **acc 0.888** | [docs/05](docs/05_newsgroups_text_classification.md) |
| 06 | Time series forecasting | Airline passengers | HTTPS download | log+Fourier **MAPE 11.6%** | [docs/06](docs/06_airline_timeseries.md) |
| 07 | Recommendation | MovieLens 100k | GroupLens (HTTPS download) | item-item CF **RMSE 0.918** vs 1.021 base (**−10.1%**) | [docs/07](docs/07_movielens_recommender.md) |
| 08 | EDA / open gov data | Brazil indicators | World Bank API | GDP/cap **CAGR 6.1%** (235 → 10,310 US$) | [docs/08](docs/08_worldbank_brazil_eda.md) |
| 09 | Credit risk (classification) | German Credit (Statlog) | UCI (HTTPS download) | RF **ROC-AUC 0.804** (30% bad-rate) | [docs/09](docs/09_credit_risk_classification.md) |
| 10 | Multiclass + viz | Iris | scikit-learn (bundled) | LogReg **5-fold CV acc 0.960** | [docs/10](docs/10_iris_multiclass.md) |

Each write-up references its plot in [`outputs/`](outputs/) (07 writes a text result instead of
a PNG).

## Why this layout

These are intentionally *small* analyses chosen to cover the **breadth** of applied data science
on **trustworthy, key-free public data**, and to demonstrate good practice over flashy numbers:
stratified / chronological splits, the right metric per problem (ROC-AUC under imbalance, MAPE
for forecasts, ARI for clustering, baseline comparison for recommenders), and honest
interpretation of limitations. The `docs/` mini-articles are where the actual **analysis of the
results** lives.

## Kaggle-ready

Every script runs on a public dataset with **no key required**. To swap in a Kaggle dataset:
1. `pip install kaggle`; place `kaggle.json` (or set `KAGGLE_USERNAME` / `KAGGLE_KEY` in `.env`).
2. Uncomment / add the `kaggle datasets download -d <slug>` line at the top of the script.
3. Point the loader at the downloaded CSV (each script isolates its data-loading step).

## Run

```bash
pip install -r requirements.txt
py 01_breast_cancer_classification.py   # then 02_..., 03_..., etc.  (use 'python' if 'py' is unavailable)
```

Outputs (plots / result files) are written to `outputs/`. Each analysis is also provided as a
**Colab-ready notebook** (`NN_*.ipynb`) — open in Jupyter / Google Colab and run top-to-bottom.

## Stack

Python · scikit-learn · pandas · NumPy · matplotlib · requests
