"""Unsupervised clustering probe: do KMeans / DBSCAN groups concentrate insiders?

Mirrors isoforest.ipynb: same training_data.parquet, same top-30 features by
XGB gain. UNSUPERVISED — labels (is_insider) are withheld during fitting and
used ONLY afterward to measure per-cluster insider prevalence. Unlike the
tree-based Isolation Forest, KMeans/DBSCAN are distance-based, so we median-
impute and StandardScale the features first.
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.cluster import DBSCAN, KMeans
from sklearn.impute import SimpleImputer
from sklearn.metrics import silhouette_score
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler

CACHE = Path("cache")
TRAIN_PARQUET = CACHE / "training_data.parquet"
MODEL_DIR = CACHE / "models"
EXISTING_MODEL = MODEL_DIR / "xgb_insider_latest.json"
EXISTING_META = MODEL_DIR / "xgb_insider_latest.meta.json"
TOP_K = 30
LABEL_COL = "is_insider"
RANDOM_STATE = 42

df = pd.read_parquet(TRAIN_PARQUET)
n_pos = int((df[LABEL_COL] == 1).sum())
n_neg = int((df[LABEL_COL] == 0).sum())
base_rate = n_pos / (n_pos + n_neg)
print(f"data: shape={df.shape}  insiders={n_pos}  negatives={n_neg}  "
      f"base insider rate={base_rate:.3f}")

# --- reproduce top-30 features by gain from the 319-feat XGB model ---
with open(EXISTING_META) as f:
    feature_names = json.load(f)["features"]
booster = xgb.XGBClassifier()
booster.load_model(str(EXISTING_MODEL))
ranked = sorted(zip(feature_names, booster.feature_importances_),
                key=lambda kv: kv[1], reverse=True)
feature_cols = [name for name, _ in ranked[:TOP_K]]

X_raw = df[feature_cols].astype(float)
y = df[LABEL_COL].astype(int).to_numpy()

# distance-based algorithms need imputation + scaling (trees did not)
X = StandardScaler().fit_transform(
    SimpleImputer(strategy="median").fit_transform(X_raw)
)
print(f"feature matrix: {X.shape} (median-imputed, standardized)\n")


def profile_clusters(labels, name):
    """Print per-cluster size + insider prevalence. labels==-1 is DBSCAN noise."""
    print(f"=== {name} ===")
    uniq = sorted(set(labels))
    print(f"{'cluster':>8} {'size':>6} {'insiders':>9} {'rate':>7} {'lift':>6}")
    best = None
    for c in uniq:
        mask = labels == c
        size = int(mask.sum())
        ins = int(y[mask].sum())
        rate = ins / size if size else 0.0
        lift = rate / base_rate if base_rate else 0.0
        tag = " (noise)" if c == -1 else ""
        print(f"{c:>8} {size:>6} {ins:>9} {rate:>7.3f} {lift:>5.2f}x{tag}")
        if c != -1 and (best is None or rate > best[1]):
            best = (c, rate, size, ins)
    # how concentrated are insiders overall? entropy-free simple check:
    if best:
        print(f"  -> purest cluster #{best[0]}: {best[3]}/{best[2]} insiders "
              f"({best[1]:.3f}, {best[1]/base_rate:.2f}x base rate)")
    # fraction of ALL insiders that land in any single cluster
    frac_in_purest = best[3] / n_pos if best else 0.0
    print(f"  -> that cluster captures {frac_in_purest:.1%} of all insiders\n")


# ---------------- KMeans: sweep k, report silhouette + purity ----------------
print("################ KMeans ################\n")
for k in (2, 3, 4, 5, 6, 8, 10):
    km = KMeans(n_clusters=k, random_state=RANDOM_STATE, n_init=10)
    labels = km.fit_predict(X)
    sil = silhouette_score(X, labels)
    print(f"[k={k}] silhouette={sil:.3f}")
    profile_clusters(labels, f"KMeans k={k}")

# ---------------- DBSCAN: tune eps from k-distance elbow ----------------
print("################ DBSCAN ################\n")
min_samples = 5
nbrs = NearestNeighbors(n_neighbors=min_samples).fit(X)
kdist = np.sort(nbrs.kneighbors(X)[0][:, -1])
pcts = [50, 75, 90, 95]
print(f"{min_samples}-NN distance percentiles: "
      + ", ".join(f"p{p}={np.percentile(kdist, p):.2f}" for p in pcts))
print("(sweeping eps around these values)\n")

for eps in [np.percentile(kdist, p) for p in (75, 90, 95)]:
    db = DBSCAN(eps=float(eps), min_samples=min_samples)
    labels = db.fit_predict(X)
    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    n_noise = int((labels == -1).sum())
    print(f"[eps={eps:.2f}, min_samples={min_samples}] "
          f"clusters={n_clusters}  noise={n_noise}")
    if n_clusters == 0:
        print("  -> all points labeled noise; no structure at this eps\n")
        continue
    profile_clusters(labels, f"DBSCAN eps={eps:.2f}")
