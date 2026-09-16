"""Isolation Forest visualization, parallel to cluster_figure.py.

UNSUPERVISED: IsolationForest fits on the feature distribution alone (labels
withheld); we consult labels only afterward. Mirrors isoforest.ipynb --
same top-30 XGB-gain features, contamination = insider prevalence. The
anomaly SCORE (higher = more outlier-like) is what the model actually
produces; we show whether high-anomaly regions coincide with true insiders.

Outputs:
  isoforest_projection.png  -- PCA/t-SNE colored by true label vs anomaly score
  isoforest_scores.png      -- anomaly-score distribution, insider vs non-insider
"""
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.decomposition import PCA
from sklearn.ensemble import IsolationForest
from sklearn.impute import SimpleImputer
from sklearn.manifold import TSNE
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.preprocessing import StandardScaler

CACHE = Path("cache")
MODEL_DIR = CACHE / "models"
TOP_K = 30
LABEL_COL = "is_insider"
RANDOM_STATE = 42

df = pd.read_parquet(CACHE / "training_data.parquet")
with open(MODEL_DIR / "xgb_insider_latest.meta.json") as f:
    feature_names = json.load(f)["features"]
booster = xgb.XGBClassifier()
booster.load_model(str(MODEL_DIR / "xgb_insider_latest.json"))
ranked = sorted(zip(feature_names, booster.feature_importances_),
                key=lambda kv: kv[1], reverse=True)
feature_cols = [n for n, _ in ranked[:TOP_K]]

y = df[LABEL_COL].astype(int).to_numpy()
prevalence = y.mean()

# IsolationForest is tree-based (scale-invariant): fit on imputed-but-unscaled
# features, matching isoforest.ipynb.
X_imp = SimpleImputer(strategy="median").fit_transform(df[feature_cols].astype(float))
iso = IsolationForest(n_estimators=300, contamination=prevalence,
                      random_state=RANDOM_STATE, n_jobs=-1).fit(X_imp)
anomaly = -iso.decision_function(X_imp)      # higher = more insider-like
flagged = iso.predict(X_imp) == -1           # predicted outlier at contamination

print(f"whole-set ROC-AUC={roc_auc_score(y, anomaly):.3f}  "
      f"PR-AUC={average_precision_score(y, anomaly):.3f}")

# 2-D coordinates: same standardized matrix as the clustering figure, so the
# two figures are directly comparable panel-to-panel.
X_std = StandardScaler().fit_transform(X_imp)
pca_xy = PCA(n_components=2, random_state=RANDOM_STATE).fit_transform(X_std)
tsne_xy = TSNE(n_components=2, random_state=RANDOM_STATE,
               perplexity=30, init="pca").fit_transform(X_std)

# ---------------- figure 1: projections ----------------
fig, axes = plt.subplots(2, 2, figsize=(13, 11))

def scatter_label(ax, xy, title):
    neg, pos = y == 0, y == 1
    ax.scatter(xy[neg, 0], xy[neg, 1], s=12, c="#bbbbbb", alpha=0.6,
               label="non-insider", linewidths=0)
    ax.scatter(xy[pos, 0], xy[pos, 1], s=26, c="#d62728", alpha=0.85,
               label="insider", linewidths=0)
    ax.set_title(title); ax.legend(loc="best", fontsize=9)

def scatter_score(ax, xy, title):
    order = np.argsort(anomaly)   # draw most-anomalous last (on top)
    sc = ax.scatter(xy[order, 0], xy[order, 1], s=16, c=anomaly[order],
                    cmap="viridis", alpha=0.85, linewidths=0)
    ax.set_title(title)
    plt.colorbar(sc, ax=ax, fraction=0.046, pad=0.04, label="anomaly score")

scatter_label(axes[0, 0], pca_xy, "PCA -- colored by TRUE label")
scatter_score(axes[0, 1], pca_xy, "PCA -- colored by IsoForest anomaly score")
scatter_label(axes[1, 0], tsne_xy, "t-SNE -- colored by TRUE label")
scatter_score(axes[1, 1], tsne_xy, "t-SNE -- colored by IsoForest anomaly score")
for ax in axes.ravel():
    ax.set_xticks([]); ax.set_yticks([])
fig.suptitle("Isolation Forest: high-anomaly regions only partly overlap true "
             "insiders (ROC-AUC 0.72)", fontsize=13)
fig.tight_layout(rect=(0, 0, 1, 0.97))
fig.savefig("isoforest_projection.png", dpi=150, bbox_inches="tight")
print("saved isoforest_projection.png")

# ---------------- figure 2: score distribution ----------------
fig2, ax = plt.subplots(figsize=(8, 5))
bins = np.linspace(anomaly.min(), anomaly.max(), 40)
ax.hist(anomaly[y == 0], bins=bins, alpha=0.6, color="#bbbbbb",
        density=True, label="non-insider")
ax.hist(anomaly[y == 1], bins=bins, alpha=0.6, color="#d62728",
        density=True, label="insider")
thr = np.sort(anomaly)[::-1][int(prevalence * len(anomaly))]
ax.axvline(thr, color="k", ls="--", lw=1,
           label=f"flag threshold (contamination={prevalence:.2f})")
ax.set_xlabel("IsoForest anomaly score (higher = more outlier-like)")
ax.set_ylabel("density")
ax.set_title("Anomaly-score distributions overlap heavily: "
             "insiders are not the only outliers")
ax.legend()
fig2.tight_layout()
fig2.savefig("isoforest_scores.png", dpi=150, bbox_inches="tight")
print("saved isoforest_scores.png")
