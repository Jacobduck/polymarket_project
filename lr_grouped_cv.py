"""Recompute the logistic-regression CV PR-AUC under the *tree* CV scheme.

The stored ``lr_insider_30feat`` model tuned C with plain 5-fold
``LogisticRegressionCV`` (StratifiedKFold, no grouping). That lets wallets from
the same market land in both train and validation folds -- look-ahead leakage
that inflates and over-stabilises the CV number. The tree models instead use
``StratifiedGroupKFold(5)`` grouped on ``market_slug`` with the iran market
pinned as the held-out test set (see xgboost3.py).

This script holds the LR pipeline config fixed (median impute -> standardize ->
LogisticRegression at the model's selected C, same class_weight) and swaps ONLY
the CV splitter, so the delta isolates the leakage effect. It prints the plain
vs grouped CV side by side, evaluates on the pinned test market for a
tree-comparable test-PR, and writes a proper ``cv`` block into the LR meta so
compare_models.py / compare_params.py plot the honest, grouped number.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

CACHE = Path("cache")
MODEL_DIR = CACHE / "models"
TRAIN_PARQUET = CACHE / "training_data.parquet"

LABEL_COL = "is_insider"
GROUP_COL = "market_slug"
RANDOM_STATE = 42
N_CV_FOLDS = 5
PIN_TEST_MARKETS = ("us-strikes-iran-by-march-1-2026-492",)

LR_META = MODEL_DIR / "lr_insider_30feat_latest.meta.json"


def build_pipeline(C: float, class_weight: dict) -> Pipeline:
    return Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(
            C=C, penalty="l2", solver="lbfgs", max_iter=2000,
            class_weight=class_weight, random_state=RANDOM_STATE,
        )),
    ])


def cv_pr_auc(splitter, X, y, groups, C, class_weight) -> list[float]:
    scores: list[float] = []
    split_args = (X, y, groups) if groups is not None else (X, y)
    for tr, va in splitter.split(*split_args):
        model = build_pipeline(C, class_weight)
        model.fit(X.iloc[tr], y.iloc[tr])
        proba = model.predict_proba(X.iloc[va])[:, 1]
        scores.append(average_precision_score(y.iloc[va], proba))
    return scores


def main() -> None:
    meta = json.load(open(LR_META))
    feature_cols = meta["features"]
    C = float(meta["selected_C"])
    class_weight = {0: 1.0, 1: float(meta["scale_pos_weight"])}

    df = pd.read_parquet(TRAIN_PARQUET)
    test_mask = df[GROUP_COL].isin(PIN_TEST_MARKETS)
    df_train = df[~test_mask].reset_index(drop=True)
    df_test = df[test_mask].reset_index(drop=True)

    X_tr = df_train[feature_cols].astype(float)
    y_tr = df_train[LABEL_COL].astype(int)
    g_tr = df_train[GROUP_COL]
    X_te = df_test[feature_cols].astype(float)
    y_te = df_test[LABEL_COL].astype(int)

    print(f"[lr-cv] train markets={g_tr.nunique()} rows={len(X_tr)} "
          f"pos={int(y_tr.sum())} | pinned test market(s)={list(PIN_TEST_MARKETS)} "
          f"rows={len(X_te)} pos={int(y_te.sum())}")
    print(f"[lr-cv] fixed config: C={C}, class_weight={class_weight}\n")

    plain = cv_pr_auc(
        StratifiedKFold(n_splits=N_CV_FOLDS, shuffle=True, random_state=RANDOM_STATE),
        X_tr, y_tr, None, C, class_weight)
    grouped = cv_pr_auc(
        StratifiedGroupKFold(n_splits=N_CV_FOLDS, shuffle=True, random_state=RANDOM_STATE),
        X_tr, y_tr, g_tr, C, class_weight)

    print(f"{'CV scheme':<34} {'per-fold PR-AUC':<40} {'mean':>7} {'std':>7}")
    print("-" * 92)
    for name, sc in [("StratifiedKFold (row, leaky)", plain),
                     ("StratifiedGroupKFold (market)", grouped)]:
        folds = "  ".join(f"{s:.3f}" for s in sc)
        print(f"{name:<34} {folds:<40} {np.mean(sc):>7.4f} {np.std(sc):>7.4f}")

    # tree-comparable held-out test on the pinned iran market
    final = build_pipeline(C, class_weight)
    final.fit(X_tr, y_tr)
    proba_te = final.predict_proba(X_te)[:, 1]
    test_pr = average_precision_score(y_te, proba_te)
    test_roc = roc_auc_score(y_te, proba_te)

    print()
    print(f"[lr-cv] OLD stored CV (plain, from LogisticRegressionCV.scores_): "
          f"{meta.get('cv', {}).get('pr_auc_mean', 'n/a')}")
    print(f"[lr-cv] grouped CV PR-AUC : {np.mean(grouped):.4f} ± {np.std(grouped):.4f}  "
          f"(drop of {np.mean(plain) - np.mean(grouped):+.4f} vs plain)")
    print(f"[lr-cv] pinned-test PR-AUC: {test_pr:.4f}   ROC-AUC: {test_roc:.4f}   "
          f"(random-split test in meta was PR={meta['metrics']['pr_auc']:.4f})")

    meta["cv"] = {
        "scheme": (
            f"StratifiedGroupKFold(n_splits={N_CV_FOLDS}, shuffle=True, "
            f"random_state={RANDOM_STATE}) on {GROUP_COL}; folds keep markets "
            "intact and stratify by is_insider (matches xgboost3.py)"
        ),
        "pr_auc_mean": float(np.mean(grouped)),
        "pr_auc_std": float(np.std(grouped)),
        "pr_auc_folds": [float(s) for s in grouped],
        "prev_plain_kfold_mean": float(np.mean(plain)),
        "prev_plain_kfold_std": float(np.std(plain)),
        "note": (
            "Recomputed by lr_grouped_cv.py under the tree CV scheme. The "
            "original LogisticRegressionCV used plain StratifiedKFold (rows), "
            "which leaks same-market wallets across folds."
        ),
    }
    json.dump(meta, open(LR_META, "w"), indent=2)
    print(f"\n[lr-cv] wrote grouped cv block -> {LR_META}")


if __name__ == "__main__":
    main()
