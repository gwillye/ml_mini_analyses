# 07 - Recommendation: MovieLens 100k (item-item Collaborative Filtering)

Headline result: item-item CF RMSE = 0.918 versus item-mean baseline 1.021, a 10.1% error reduction.

## Problem

MovieLens 100k is the reference recommender dataset: 100,000 ratings (1 to 5 stars) from 943 users on 1,682 movies. The rating matrix is about 94% empty, since most users have rated only a handful of films.

The question: given who-rated-what, can we predict the star rating a user would give a movie they haven't seen, better than just guessing that movie's average? Rating prediction is the classic offline evaluation of a recommender, and the honest version of the question always includes "and better than the trivial baseline," because beating the baseline is the entire job.

## Approach

The model is item-item collaborative filtering. For a target (user u, item i), we predict from the ratings u already gave to items similar to i. Item-item CF (rather than user-user) is chosen because item similarities are more stable and the item side is smaller here. It is the standard production choice (essentially what early Amazon used).

For similarity we use mean-centred cosine. We subtract each item's mean rating before computing cosine similarity, which removes "this movie is just generally liked or disliked" bias and isolates co-rating pattern similarity. Missing entries are treated as 0 in the centred matrix, so they contribute nothing.

The prediction is the item-mean plus a similarity-weighted sum of the user's deviations on the top-30 most similar items they have rated. Capping at K=30 neighbours filters out weak, noisy correlations and keeps it fast, and predictions are clipped to the valid [1, 5] range.

The baseline is the item-mean. Every CF system has to beat "just predict the movie's average rating," and that is the line in the sand.

Validation is a random 80/20 hold-out of the rating events (seed 42). The metric is RMSE in stars, reported for both CF and the baseline so the comparison is explicit.

## Results and analysis

Running `py 07_movielens_recommender.py` (results also saved to [`outputs/07_movielens_results.txt`](../outputs/07_movielens_results.txt)):

```
ratings=100000 users=943 items=1682
RMSE_cf=0.918 RMSE_base=1.021 improve=10.1%
```

A `RuntimeWarning: Mean of empty slice` prints during the run. It is benign: a handful of items have no training ratings after the split, and the code immediately backfills their mean with the global mean.

The number that matters is not 0.918 in isolation, it is 0.918 versus 1.021. Anyone can report an RMSE, and the discipline is showing it beats the trivial baseline. Collaborative filtering cuts the typical star error from about 1.02 to about 0.92, a 10.1% relative improvement, which means personalization genuinely adds signal beyond "popular movies are rated highly." On a 1 to 5 scale, an RMSE around 0.92 is a respectable result for a from-scratch neighbourhood model and is in the expected range for classic item-item CF on this dataset.

What this means in practice: the 10% lift is exactly the kind of margin that, at the top of a ranked recommendation list, reorders which titles a user sees. The gain is more valuable for ranking than the raw RMSE delta suggests.

A few limitations. First, cold start: brand-new items or users with no co-ratings fall back to the mean (the empty-slice case), and CF cannot help them. Second, pure neighbourhood CF is beaten by matrix factorization (SVD), which typically reaches about 0.91 to 0.93 RMSE here while generalizing better to sparse regions, and that is the natural next iteration. Third, RMSE measures rating accuracy, not top-N ranking quality (precision@k or NDCG would be the deployment metric). Fourth, this is a single random split, and the standard MovieLens protocol uses the provided 5-fold splits for comparability.

Takeaway: collaborative filtering earns its place by beating the item-mean baseline by about 10%, and the right way to report a recommender is always against that baseline, never as a lone RMSE.

## How to run

```bash
pip install -r requirements.txt
py 07_movielens_recommender.py   # downloads ml-100k.zip from GroupLens, no key required
```

MovieLens is also on Kaggle, so you can point the loader at the Kaggle `u.data` or `ratings.csv` to run offline. See the top-level README. This script writes a text result, not a PNG.
