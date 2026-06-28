# 03 — Wine Cultivar Clustering (KMeans, unsupervised)

**Headline result: KMeans (k=3) recovers the true cultivars with ARI = 0.897; silhouette = 0.285.**

## Problem
The scikit-learn **Wine** dataset has 178 wines from **three cultivars** grown in the same
Italian region, each described by 13 physicochemical features (alcohol, malic acid, ash,
flavanoids, color intensity, proline, etc.). The true cultivar label exists, but we
**pretend it does not** and ask an *unsupervised* question:

**Without any labels, does the natural structure of the chemistry alone re-discover the three
cultivars?** This is the canonical test of whether clustering finds *meaningful* groups
rather than arbitrary ones — and we get to grade it because the ground truth is available.

## Approach
- **Standardization first (critical).** The 13 features live on wildly different scales
  (proline is in the hundreds; many acids are below 5). Without scaling, a Euclidean-distance
  method like KMeans would be dominated by proline alone. We apply `StandardScaler`
  (zero mean, unit variance) so every feature contributes comparably.
- **Model: KMeans, k=3** (`n_init=10`, `random_state=42`). We *fix* k=3 because the question
  is explicitly "do the clusters match the three known cultivars?" `n_init=10` runs ten random
  initializations and keeps the best inertia, guarding against bad seeds.
- **PCA(2) for visualization only** — the clustering happens in the full 13-D standardized
  space; PCA just projects to 2-D so we can see the separation.
- **Validation / scoring:**
  - **Silhouette score** — internal, label-free: how tight and well-separated the clusters
    are (range −1 to 1).
  - **Adjusted Rand Index (ARI)** — external: agreement between cluster assignments and the
    true cultivars, corrected for chance (0 = random, 1 = perfect).

## Results & analysis
Running `py 03_wine_clustering.py`:

```
silhouette=0.285  ARI_vs_true=0.897
```

The PCA scatter coloured by predicted cluster is in
[`outputs/03_wine_clusters.png`](../outputs/03_wine_clusters.png): three clearly separated
blobs along PC1/PC2, with only a thin boundary region of ambiguous points.

**Interpretation.** The standout number is **ARI = 0.897**. Adjusted for chance, that means
the purely unsupervised clustering **almost perfectly reproduces the three true cultivars** —
the wine chemistry is so cultivar-specific that the grouping structure *is* the cultivar
structure. A near-zero ARI would have meant "clusters are arbitrary"; 0.90 means "the chemistry
knows the grape."

The **silhouette of 0.285** is the instructive contrast. It is only "moderate," which at first
looks like a contradiction with the excellent ARI. It is not: silhouette measures *geometric
compactness* in 13-D, where clusters touch at their edges, while ARI measures *label agreement*.
The lesson is that **clusters can be correct without being geometrically crisp** — a few wines
sit near a boundary (lowering silhouette) yet are still assigned to the right cultivar (keeping
ARI high). Reporting both metrics is what makes this honest.

**What worked:** standardization. Skipping it would let proline swamp the distance metric and
collapse the result. The whole quality of this analysis hinges on that one preprocessing step.

**Limitations.** (1) We *gave* the algorithm k=3; a real unsupervised task must also *discover*
k (e.g. via the elbow / silhouette-over-k method). (2) ARI is only computable because labels
happen to exist here — it is a validation luxury, not something available in true production
clustering. (3) KMeans assumes roughly spherical, equal-variance clusters; it works here, but
would struggle on elongated or unequal-density groups.

**Takeaway:** with proper scaling, the wine chemistry is **self-organizing** — an unlabeled
method recovers the cultivars at ARI 0.90, and the silhouette/ARI gap is a clean reminder that
"compact" and "correct" are different things.

## How to run
```bash
pip install -r requirements.txt
py 03_wine_clustering.py
```
Dataset is bundled with scikit-learn — **no download, no key**. Swap `load_wine()` for a Kaggle
CSV loader to cluster your own data (see top-level README).
