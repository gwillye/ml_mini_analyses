# 10 - Multiclass Classification + Viz: Iris (Logistic Regression + PCA)

Headline result: 5-fold CV accuracy = 0.960 (hold-out accuracy 0.911) across 3 species.

## Problem

The Iris dataset is the "Hello, World" of classification: 150 flowers, 3 species (setosa, versicolor, virginica, 50 each, perfectly balanced), described by 4 measurements (sepal length/width, petal length/width).

The question: can a simple linear classifier separate the three species, and how separable are they geometrically? Iris is the canonical sanity-check pipeline. Its value here is demonstrating a clean, correct multiclass workflow (proper CV plus a 2-D visualization that explains the result) rather than a hard problem.

## Approach

We standardize the 4 features with `StandardScaler` so each contributes comparably to both the classifier and the PCA, since distance and coefficient-based methods are scale-sensitive.

The model is a multinomial Logistic Regression (`max_iter=500`). A linear model is deliberately chosen: if a plain linear boundary already nails it, nothing more complex is warranted, and it keeps the decision interpretable.

PCA(2) is used for visualization. The 4-D data is projected to 2 components purely to see the class structure, while classification uses all 4 standardized features.

Validation runs two complementary checks. The first is 5-fold cross-validation on all 150 rows, which is the headline metric, because on a 150-row dataset a single split is high-variance, so CV averages over five folds for a stable estimate. The second is a stratified 70/30 hold-out for a per-class precision/recall report and a confusion view. The metric is accuracy, which is fair because the classes are exactly balanced.

## Results and analysis

Running `py 10_iris_multiclass.py`:

```
              precision    recall  f1-score   support
      setosa      1.000     1.000     1.000        15
  versicolor      0.824     0.933     0.875        15
   virginica      0.923     0.800     0.857        15
    accuracy                          0.911        45
CV accuracy=0.960  holdout accuracy=0.911
```

The PCA projection coloured by species is in [`outputs/10_iris_pca.png`](../outputs/10_iris_pca.png).

Report the CV number (0.960) as the headline, not the hold-out (0.911). With only 45 test flowers, the hold-out swings by whole-percent steps per misclassification, so it is noisy, and the 5-fold average is the trustworthy estimate. The model is excellent, and the structure of the result is the lesson.

setosa is perfect (F1 = 1.000). The PCA plot shows why: setosa sits as a completely isolated cluster, linearly separable from the other two, so no model could do worse than perfect here. versicolor and virginica are where every error lives. They form two adjacent, slightly overlapping clouds in PCA space. The hold-out confusion is symmetric and tiny, with versicolor recall 0.933 and virginica recall 0.800 (a few virginica flowers near the boundary get called versicolor). This is a genuine biological overlap, not a model defect.

So the picture and the metrics agree: one species is trivially separable, and the other two share a fuzzy boundary, and that boundary region is the entire residual error.

What worked: a linear model is the correct level of complexity. CV 0.96 with full interpretability means a heavier model (SVM or RF) would add nothing but opacity here, which is a useful discipline reminder.

A few limitations. First, 150 rows is small enough that even CV carries a confidence interval, so treat 0.96 as "excellent," not as four-significant-figure truth. Second, Iris is too easy to be evidence of method quality on hard data, since it validates the pipeline, not the modelling chops. Third, the versicolor/virginica overlap is irreducible with these 4 features, since it is a property of the flowers, not something tuning can remove.

Takeaway: match model complexity to the problem. A standardized linear classifier hits about 96% CV on Iris with a fully interpretable, picture-backed result, and the only errors are the one genuinely overlapping species pair.

## How to run

```bash
pip install -r requirements.txt
py 10_iris_multiclass.py
```

The dataset ships with scikit-learn, so there is no download and no key. Swap `load_iris()` for a Kaggle CSV loader to reuse the standardize, CV, and PCA-plot template. See the top-level README.
