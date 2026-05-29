"""Train a small XGBoost insider model on the 14 DEFAULT_BEHAVIORAL_FEATURES.

Reuses cache/training_data.parquet (793 labeled (wallet, market) rows, 319
columns) — we just project down to the 14 columns of interest.

Outputs:
  cache/models/xgb_insider_14feat_<timestamp>.json
  cache/models/xgb_insider_14feat_<timestamp>.meta.json
  cache/models/xgb_insider_14feat_latest.json        (copy)
  cache/models/xgb_insider_14feat_latest.meta.json   (copy)
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

from polycluster.modeling import DEFAULT_BEHAVIORAL_FEATURES

CACHE = Path("cache")
TRAIN_PARQUET = CACHE / "training_data.parquet"
MODEL_DIR = CACHE / "models"
MODEL_DIR.mkdir(parents=True, exist_ok=True)

LABEL_COL = "is_insider"
THRESHOLD = 0.5
RANDOM_STATE = 42


def main() -> None:
    print(f"[train-14] loading {TRAIN_PARQUET}")
    df = pd.read_parquet(TRAIN_PARQUET)
    print(f"[train-14] shape={df.shape}, positives={int((df[LABEL_COL] == 1).sum())}, "
          f"negatives={int((df[LABEL_COL] == 0).sum())}")

    feature_cols = list(DEFAULT_BEHAVIORAL_FEATURES)
    missing = [c for c in feature_cols if c not in df.columns]
    if missing:
        raise SystemExit(f"missing feature columns in training data: {missing}")
    print(f"[train-14] using {len(feature_cols)} features")

    X = df[feature_cols].astype(float)
    y = df[LABEL_COL].astype(int)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=RANDOM_STATE,
    )

    n_pos = int((y_train == 1).sum())
    n_neg = int((y_train == 0).sum())
    scale_pos_weight = n_neg / max(n_pos, 1)
    print(f"[train-14] train n={len(X_train)} (pos={n_pos}, neg={n_neg})  "
          f"scale_pos_weight={scale_pos_weight:.3f}")
    print(f"[train-14] test  n={len(X_test)} (pos={int((y_test == 1).sum())}, "
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
    print(f"[train-14] fit in {time.time() - t0:.2f}s")

    proba = model.predict_proba(X_test)[:, 1]
    pred = (proba >= THRESHOLD).astype(int)

    roc_auc = roc_auc_score(y_test, proba)
    pr_auc = average_precision_score(y_test, proba)
    print()
    print(f"[train-14] TEST ROC-AUC = {roc_auc:.4f}")
    print(f"[train-14] TEST PR-AUC  = {pr_auc:.4f}")
    print()
    print(f"[train-14] confusion matrix @ thr={THRESHOLD}")
    print(confusion_matrix(y_test, pred))
    print()
    print(classification_report(y_test, pred, digits=4))

    importances = sorted(
        zip(feature_cols, model.feature_importances_),
        key=lambda kv: kv[1],
        reverse=True,
    )
    print("[train-14] feature importances (gain):")
    for name, imp in importances:
        print(f"  {imp:7.4f}  {name}")

    ts = time.strftime("%Y%m%d_%H%M%S")
    model_path = MODEL_DIR / f"xgb_insider_14feat_{ts}.json"
    meta_path = MODEL_DIR / f"xgb_insider_14feat_{ts}.meta.json"

    model.save_model(str(model_path))

    meta = {
        "trained_at": ts,
        "n_train": len(X_train),
        "n_test": len(X_test),
        "n_features": len(feature_cols),
        "features": feature_cols,
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

    latest_model = MODEL_DIR / "xgb_insider_14feat_latest.json"
    latest_meta = MODEL_DIR / "xgb_insider_14feat_latest.meta.json"
    shutil.copy(model_path, latest_model)
    shutil.copy(meta_path, latest_meta)

    print()
    print(f"[train-14] saved {model_path}")
    print(f"[train-14] saved {meta_path}")
    print(f"[train-14] copied -> {latest_model}")
    print(f"[train-14] copied -> {latest_meta}")
    print()
    print("[train-14] comparison vs 319-feat model:")
    try:
        with open(MODEL_DIR / "xgb_insider_latest.meta.json") as f:
            old = json.load(f)
        old_metrics = old.get("metrics", {})
        print(f"  319-feat: ROC-AUC={old_metrics.get('roc_auc', 'n/a'):.4f}  "
              f"PR-AUC={old_metrics.get('pr_auc', 'n/a'):.4f}")
        print(f"   14-feat: ROC-AUC={roc_auc:.4f}  PR-AUC={pr_auc:.4f}")
    except FileNotFoundError:
        print("  (no 319-feat meta found to compare)")


if __name__ == "__main__":
    main()
