# 07 — Recommendation: MovieLens 100k (item-item Collaborative Filtering)

**Headline result: item-item CF RMSE = 0.918 vs item-mean baseline 1.021 — a 10.1% error reduction.**

## Problem
**MovieLens 100k** is the reference recommender dataset: **100,000 ratings** (1–5 stars) from
**943 users** on **1,682 movies**. The rating matrix is ~94% empty — most users have rated only
a handful of films.

The question: **given who-rated-what, can we predict the star rating a user would give a movie
they haven't seen, better than just guessing that movie's average?** Rating prediction is the
classic offline evaluation of a recommender, and the *honest* version of the question always
includes "...better than the trivial baseline," because beating the baseline is the entire job.

## Approach
- **Model: item-item collaborative filtering.** For a target (user *u*, item *i*), predict
  from the ratings *u* already gave to **items similar to *i***. Item-item (rather than
  user-user) CF is chosen because item similarities are more stable and the item side is
  smaller here — the standard production choice (it's essentially what early Amazon used).
- **Similarity: mean-centred cosine.** We subtract each item's mean rating before computing
  cosine similarity, which removes "this movie is just generally liked/disliked" bias and
  isolates *co-rating pattern* similarity. Missing entries are treated as 0 in the centred
  matrix (no contribution).
- **Prediction:** item-mean **+** a similarity-weighted sum of the user's deviations on the
  **top-30 most similar items** they've rated. Capping at K=30 neighbours filters out weak,
  noisy correlations and keeps it fast; predictions are clipped to the valid [1, 5] range.
- **Baseline: item-mean.** Every CF system must beat "just predict the movie's average rating"
  — this is the line in the sand.
- **Validation:** a random **80/20 hold-out** of the rating events (seed 42). Metric: **RMSE**
  in stars, reported for *both* CF and baseline so the comparison is explicit.

## Results & analysis
Running `py 07_movielens_recommender.py` (results also saved to
[`outputs/07_movielens_results.txt`](../outputs/07_movielens_results.txt)):

```
ratings=100000 users=943 items=1682
RMSE_cf=0.918 RMSE_base=1.021 improve=10.1%
```

*(A `RuntimeWarning: Mean of empty slice` prints during the run — it is benign: a handful of
items have no training ratings after the split, and the code immediately backfills their mean
with the global mean.)*

**Interpretation.** The number that matters is **not 0.918 in isolation — it's 0.918 vs
1.021.** Anyone can report an RMSE; the discipline is showing it beats the trivial baseline.
Collaborative filtering cuts the typical star error from ~1.02 to ~0.92, a **10.1% relative
improvement** — i.e. personalization genuinely adds signal beyond "popular movies are rated
highly." On a 1–5 scale, ~0.92 RMSE is a respectable result for a from-scratch neighbourhood
model and is in the expected range for classic item-item CF on this dataset.

**What this means in practice:** the 10% lift is exactly the kind of margin that, at the top of
a ranked recommendation list, reorders which titles a user sees — the gain is more valuable for
*ranking* than the raw RMSE delta suggests.

**Limitations.** (1) **Cold start** — brand-new items/users with no co-ratings fall back to the
mean (the empty-slice case); CF cannot help them. (2) Pure neighbourhood CF is beaten by
**matrix factorization (SVD)**, which typically reaches ~0.91–0.93 RMSE here while generalizing
better to sparse regions — the natural next iteration. (3) RMSE measures rating accuracy, not
top-N ranking quality (precision@k / NDCG would be the deployment metric). (4) A single random
split; the standard MovieLens protocol uses the provided 5-fold splits for comparability.

**Takeaway:** collaborative filtering **earns its place by beating the item-mean baseline by
~10%** — and the right way to report a recommender is always *against that baseline*, never as
a lone RMSE.

## How to run
```bash
pip install -r requirements.txt
py 07_movielens_recommender.py   # downloads ml-100k.zip from GroupLens — no key required
```
MovieLens is also on Kaggle — point the loader at the Kaggle `u.data`/`ratings.csv` to run
offline (see top-level README). *(This script writes a text result, not a PNG.)*
