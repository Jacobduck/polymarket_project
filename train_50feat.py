"""Train XGBoost on the top 50 features by gain from the 319-feature model.

We pull feature importances from cache/models/xgb_insider_latest.json (the
existing 319-feature XGBoost), pick the top 50 by gain, then refit a fresh
model on just those 50 columns from cache/training_data.parquet.

Outputs:
  cache/models/xgb_insider_50feat_<timestamp>.json
  cache/models/xgb_insider_50feat_<timestamp>.meta.json
  cache/models/xgb_insider_50feat_latest.json        (copy)
  cache/models/xgb_insider_50feat_latest.meta.json   (copy)
"""

from __future__ import annotations

import json
import shutil
import time
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import (
    average_precision_score,
    classification_report,
    confusion_matrix,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split

CACHE = Path("cache")
TRAIN_PARQUET = CACHE / "training_data.parquet"
MODEL_DIR = CACHE / "models"
MODEL_DIR.mkdir(parents=True, exist_ok=True)

EXISTING_MODEL = MODEL_DIR / "xgb_insider_latest.json"
EXISTING_META = MODEL_DIR / "xgb_insider_latest.meta.json"

TOP_K = 50
LABEL_COL = "is_insider"
THRESHOLD = 0.5
RANDOM_STATE = 42


def get_top_k_features(k: int) -> list[str]:
    """Load the 319-feature model and return its top-k features by gain."""
    with open(EXISTING_META) as f:
        meta = json.load(f)
    feature_names = meta["features"]

    booster = xgb.XGBClassifier()
    booster.load_model(str(EXISTING_MODEL))
    importances = booster.feature_importances_

    if len(importances) != len(feature_names):
        raise SystemExit(
            f"importance length {len(importances)} != feature names "
            f"length {len(feature_names)}"
        )

    ranked = sorted(
        zip(feature_names, importances),
        key=lambda kv: kv[1],
        reverse=True,
    )
    top_k = [name for name, _ in ranked[:k]]

    print(f"[train-50] top {k} features by gain from 319-feat model:")
    for i, (name, imp) in enumerate(ranked[:k], 1):
        print(f"  {i:2d}. {imp:7.4f}  {name}")
    print()

    return top_k


def main() -> None:
    print(f"[train-50] loading {TRAIN_PARQUET}")
    df = pd.read_parquet(TRAIN_PARQUET)
    print(f"[train-50] shape={df.shape}, positives={int((df[LABEL_COL] == 1).sum())}, "
          f"negatives={int((df[LABEL_COL] == 0).sum())}")
    print()

    feature_cols = get_top_k_features(TOP_K)

    missing = [c for c in feature_cols if c not in df.columns]
    if missing:
        raise SystemExit(f"missing feature columns in training data: {missing}")

    X = df[feature_cols].astype(float)
    y = df[LABEL_COL].astype(int)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=RANDOM_STATE,
    )

    n_pos = int((y_train == 1).sum())
    n_neg = int((y_train == 0).sum())
    scale_pos_weight = n_neg / max(n_pos, 1)
    print(f"[train-50] train n={len(X_train)} (pos={n_pos}, neg={n_neg})  "
          f"scale_pos_weight={scale_pos_weight:.3f}")
    print(f"[train-50] test  n={len(X_test)} (pos={int((y_test == 1).sum())}, "
          f"neg={int((y_test == 0).sum())})")

    model = xgb.XGBClassifier(
        n_estimators=300,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.9,
        colsample_bytree=0.9,
        reg_lambda=1.0,
        scale_pos_weight=scale_pos_weight,
        eval_metric="aucpr",
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )

    t0 = time.time()
    model.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=False)
    print(f"[train-50] fit in {time.time() - t0:.2f}s")

    proba = model.predict_proba(X_test)[:, 1]
    pred = (proba >= THRESHOLD).astype(int)

    roc_auc = roc_auc_score(y_test, proba)
    pr_auc = average_precision_score(y_test, proba)
    print()
    print(f"[train-50] TEST ROC-AUC = {roc_auc:.4f}")
    print(f"[train-50] TEST PR-AUC  = {pr_auc:.4f}")
    print()
    print(f"[train-50] confusion matrix @ thr={THRESHOLD}")
    print(confusion_matrix(y_test, pred))
    print()
    print(classification_report(y_test, pred, digits=4))

    importances = sorted(
        zip(feature_cols, model.feature_importances_),
        key=lambda kv: kv[1],
        reverse=True,
    )
    print("[train-50] new-model feature importances (gain):")
    for name, imp in importances:
        print(f"  {imp:7.4f}  {name}")

    ts = time.strftime("%Y%m%d_%H%M%S")
    model_path = MODEL_DIR / f"xgb_insider_50feat_{ts}.json"
    meta_path = MODEL_DIR / f"xgb_insider_50feat_{ts}.meta.json"

    model.save_model(str(model_path))

    meta = {
        "trained_at": ts,
        "n_train": len(X_train),
        "n_test": len(X_test),
        "n_features": len(feature_cols),
        "features": feature_cols,
        "feature_selection": (
            f"top {TOP_K} by feature_importances_ (gain) from "
            f"{EXISTING_MODEL.name}"
        ),
        "scale_pos_weight": scale_pos_weight,
        "threshold": THRESHOLD,
        "xgb_params": {
            "n_estimators": 300,
            "max_depth": 4,
            "learning_rate": 0.05,
            "subsample": 0.9,
            "colsample_bytree": 0.9,
            "reg_lambda": 1.0,
            "eval_metric": "aucpr",
            "random_state": RANDOM_STATE,
        },
        "metrics": {"roc_auc": float(roc_auc), "pr_auc": float(pr_auc)},
        "source_training_data": str(TRAIN_PARQUET),
    }
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)

    latest_model = MODEL_DIR / "xgb_insider_50feat_latest.json"
    latest_meta = MODEL_DIR / "xgb_insider_50feat_latest.meta.json"
    shutil.copy(model_path, latest_model)
    shutil.copy(meta_path, latest_meta)

    print()
    print(f"[train-50] saved {model_path}")
    print(f"[train-50] saved {meta_path}")
    print(f"[train-50] copied -> {latest_model}")
    print(f"[train-50] copied -> {latest_meta}")
    print()
    print("[train-50] comparison:")

    def _load_metrics(meta_file: Path) -> dict | None:
        try:
            with open(meta_file) as f:
                return json.load(f).get("metrics", {})
        except FileNotFoundError:
            return None

    m319 = _load_metrics(MODEL_DIR / "xgb_insider_latest.meta.json")
    m14 = _load_metrics(MODEL_DIR / "xgb_insider_14feat_latest.meta.json")

    if m319:
        print(f"  319-feat: ROC-AUC={m319.get('roc_auc', 0):.4f}  "
              f"PR-AUC={m319.get('pr_auc', 0):.4f}")
    if m14:
        print(f"   14-feat: ROC-AUC={m14.get('roc_auc', 0):.4f}  "
              f"PR-AUC={m14.get('pr_auc', 0):.4f}")
    print(f"   50-feat: ROC-AUC={roc_auc:.4f}  PR-AUC={pr_auc:.4f}")


if __name__ == "__main__":
    main()
