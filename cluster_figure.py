"""2-D projections of the wallet feature space for the paper's Approach section.

Same top-30 XGB-gain features as isoforest.ipynb / cluster_explore.py, median-
imputed + standardized. Produces a 2x2 figure:
  row 1: PCA   colored by (true insider label) | (KMeans k=5 cluster)
  row 2: t-SNE colored by (true insider label) | (KMeans k=5 cluster)
The DBSCAN-pure clump shows up as the tight island; coloring by true label
shows insiders are NOT a single separable region -> unsupervised can't cluster
insiders as a class.
"""
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler

CACHE = Path("cache")
MODEL_DIR = CACHE / "models"
TOP_K = 30
LABEL_COL = "is_insider"
RANDOM_STATE = 42
K = 5

df = pd.read_parquet(CACHE / "training_data.parquet")
with open(MODEL_DIR / "xgb_insider_latest.meta.json") as f:
    feature_names = json.load(f)["features"]
booster = xgb.XGBClassifier()
booster.load_model(str(MODEL_DIR / "xgb_insider_latest.json"))
ranked = sorted(zip(feature_names, booster.feature_importances_),
                key=lambda kv: kv[1], reverse=True)
feature_cols = [n for n, _ in ranked[:TOP_K]]

y = df[LABEL_COL].astype(int).to_numpy()
X = StandardScaler().fit_transform(
    SimpleImputer(strategy="median").fit_transform(df[feature_cols].astype(float))
)

km_labels = KMeans(n_clusters=K, random_state=RANDOM_STATE, n_init=10).fit_predict(X)

pca_xy = PCA(n_components=2, random_state=RANDOM_STATE).fit_transform(X)
tsne_xy = TSNE(n_components=2, random_state=RANDOM_STATE,
               perplexity=30, init="pca").fit_transform(X)

fig, axes = plt.subplots(2, 2, figsize=(13, 11))

def scatter_label(ax, xy, title):
    neg = y == 0
    pos = y == 1
    ax.scatter(xy[neg, 0], xy[neg, 1], s=12, c="#bbbbbb", alpha=0.6,
               label="non-insider", linewidths=0)
    ax.scatter(xy[pos, 0], xy[pos, 1], s=26, c="#d62728", alpha=0.85,
               label="insider", linewidths=0)
    ax.set_title(title)
    ax.legend(loc="best", fontsize=9)

def scatter_cluster(ax, xy, title):
    for c in sorted(set(km_labels)):
        m = km_labels == c
        ax.scatter(xy[m, 0], xy[m, 1], s=14, alpha=0.7,
                   label=f"cluster {c} (n={int(m.sum())})", linewidths=0)
    ax.set_title(title)
    ax.legend(loc="best", fontsize=8)

scatter_label(axes[0, 0], pca_xy, "PCA — colored by TRUE label")
scatter_cluster(axes[0, 1], pca_xy, f"PCA — colored by KMeans (k={K})")
scatter_label(axes[1, 0], tsne_xy, "t-SNE — colored by TRUE label")
scatter_cluster(axes[1, 1], tsne_xy, f"t-SNE — colored by KMeans (k={K})")
for ax in axes.ravel():
    ax.set_xticks([]); ax.set_yticks([])

fig.suptitle(
    "Wallet feature space (top-30 features): insiders are mixed with non-insiders, "
    "not a single separable cluster", fontsize=13)
fig.tight_layout(rect=(0, 0, 1, 0.97))
out = "cluster_projection.png"
fig.savefig(out, dpi=150, bbox_inches="tight")
print(f"saved {out}")
