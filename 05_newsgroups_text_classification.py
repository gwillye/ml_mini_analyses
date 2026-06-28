"""05 — NLP text classification: 20 Newsgroups (downloaded & cached by scikit-learn).

TF-IDF + Logistic Regression over 4 topics. Kaggle-ready: point the loader at a Kaggle
text CSV (text + label columns) instead of fetch_20newsgroups (see README).
"""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

SEED = 42
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "outputs")
os.makedirs(OUT, exist_ok=True)

cats = ["rec.autos", "sci.med", "comp.graphics", "talk.politics.mideast"]
strip = ("headers", "footers", "quotes")
tr = fetch_20newsgroups(subset="train", categories=cats, remove=strip, random_state=SEED)
te = fetch_20newsgroups(subset="test", categories=cats, remove=strip, random_state=SEED)
print(f"train={len(tr.data)} test={len(te.data)} classes={len(cats)}")

pipe = Pipeline([
    ("tfidf", TfidfVectorizer(stop_words="english", min_df=2, ngram_range=(1, 2))),
    ("clf", LogisticRegression(max_iter=2000, C=10)),
])
pipe.fit(tr.data, tr.target)
pred = pipe.predict(te.data)
acc = accuracy_score(te.target, pred)
print(classification_report(te.target, pred, target_names=[c.split('.')[-1] for c in cats], digits=3))

cm = confusion_matrix(te.target, pred)
short = [c.split(".")[-1] for c in cats]
plt.figure(figsize=(6, 5))
plt.imshow(cm, cmap="Blues")
plt.xticks(range(len(cats)), short, rotation=30)
plt.yticks(range(len(cats)), short)
plt.xlabel("predicted")
plt.ylabel("true")
plt.title(f"20 Newsgroups — accuracy {acc:.3f}")
plt.colorbar(fraction=0.046)
plt.tight_layout()
plt.savefig(os.path.join(OUT, "05_newsgroups_confusion.png"), dpi=110)
plt.close()

print(f"accuracy = {acc:.3f}")
print("self-check:", "OK" if acc > 0.80 else "FAIL")
