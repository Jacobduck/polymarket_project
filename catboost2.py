"""Train CatBoost with train/val/test split + time-ordered grouped CV.

The dataset is dominated by two very large iran-themed markets at the
tail (iran-feb-28: 189 rows, iran-mar-1: 245 rows -- 55% of all rows
between them). A clean 60/20/20 by either market count or row count
puts test either too small or too big. Compromise:

  - Test  = single newest large market (``us-strikes-iran-by-march-1-2026-492``)
  - Train = every market that resolves BEFORE the test market
  - Val   = the rest (small markets newer than train but not in test,
            plus NaT-timestamp markets)

This preserves no-look-ahead (train all resolves before test) and gives
a test set big enough to evaluate (245 rows, 35 positives) without
swallowing the dataset.

Within train, runs 5-fold expanding-window ``TimeSeriesSplit`` over the
(few) time-ordered train markets. No row from a single market is split
across train/val within a fold.

Features = top-30 by gain from ``xgb_insider_latest`` (same list already used
by ``cb_insider_30feat_latest``).

Predicted probabilities are post-hoc calibrated via Platt scaling
(``LogisticRegression`` fit on the val set's raw probs vs. labels).
Classification threshold is fixed at ``THRESHOLD`` (default 0.80) applied to
the *calibrated* probability: a row is flagged as an insider iff
``calibrator.predict_proba(model.predict_proba(X)[:, 1]) >= THRESHOLD``.

Outputs:
  cache/models/cb_insider_30feat_v2_<ts>.joblib              (CatBoost model)
  cache/models/cb_insider_30feat_v2_<ts>.calibrator.joblib   (sklearn LR)
  cache/models/cb_insider_30feat_v2_<ts>.meta.json
  cache/models/cb_insider_30feat_v2_latest.joblib            (copies)
  cache/models/cb_insider_30feat_v2_latest.calibrator.joblib
  cache/models/cb_insider_30feat_v2_latest.meta.json
  catboost2_<ts>.log                                         (full stdout+stderr)
"""

from __future__ import annotations

import json
import shutil
import sys
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    classification_report,
    confusion_matrix,
    roc_auc_score,
)
from sklearn.model_selection import TimeSeriesSplit

CACHE = Path("cache")
TRAIN_PARQUET = CACHE / "training_data.parquet"
MODEL_DIR = CACHE / "models"
MODEL_DIR.mkdir(parents=True, exist_ok=True)

FEATURE_SOURCE_META = MODEL_DIR / "cb_insider_30feat_latest.meta.json"

LABEL_COL = "is_insider"
GROUP_COL = "market_slug"
TIME_COL = "terminal_start_ts"
THRESHOLD = 0.30  # classify row as insider iff predicted prob >= THRESHOLD
RANDOM_STATE = 42

LOG_DIR = Path(".")
LOG_PREFIX = "catboost2"

TEST_MARKETS: tuple[str, ...] = (
    "us-strikes-iran-by-march-1-2026-492",
)

N_CV_FOLDS = 5

MAX_ITERATIONS = 2000
EARLY_STOPPING_ROUNDS = 75

# scale_pos_weight: None = auto-balanced (n_neg / n_pos), float = fixed.
# We use 2.0 (better PR-AUC than auto-balanced 5.2); calibration handles
# the score-scale problem so the absolute raw-score range no longer matters.
SCALE_POS_WEIGHT_OVERRIDE: float | None = 2.0


def get_top_30_features() -> list[str]:
    """Pull the same top-30 feature list used by cb_insider_30feat_latest."""
    with open(FEATURE_SOURCE_META) as f:
        meta = json.load(f)
    feats = meta["features"]
    if len(feats) != 30:
        raise SystemExit(
            f"expected 30 features in {FEATURE_SOURCE_META}, got {len(feats)}"
        )
    print(f"[cb2] using top-30 features from {FEATURE_SOURCE_META.name}")
    return feats


def split_markets_by_time(
    df: pd.DataFrame,
) -> tuple[list[str], list[str], list[str], pd.DataFrame]:
    """Return (train_markets, val_markets, test_markets, market_order_df).

    Assigns markets to splits:
      - test  = TEST_MARKETS (pinned, see module docstring)
      - train = every market resolving before the OLDEST test market
                (by median ``terminal_start_ts``)
      - val   = every other market (small newer markets not in test,
                plus NaT-timestamp markets)
    """
    market_stats = (
        df.groupby(GROUP_COL)
        .agg(median_ts=(TIME_COL, "median"), n_rows=(LABEL_COL, "size"))
        .reset_index()
    )

    test_set = set(TEST_MARKETS)
    missing_test = test_set - set(market_stats[GROUP_COL])
    if missing_test:
        raise SystemExit(
            f"TEST_MARKETS not in training data: {sorted(missing_test)}"
        )

    test_ts = market_stats.loc[
        market_stats[GROUP_COL].isin(test_set), "median_ts"
    ]
    if test_ts.isna().any():
        raise SystemExit(
            "TEST_MARKETS must all have a non-NaT median timestamp."
        )
    test_cutoff_ts = test_ts.min()

    def _assign(row) -> str:
        if row[GROUP_COL] in test_set:
            return "test"
        if pd.isna(row["median_ts"]):
            return "val"
        if row["median_ts"] < test_cutoff_ts:
            return "train"
        return "val"

    market_stats["split"] = market_stats.apply(_assign, axis=1)
    market_stats["date"] = pd.to_datetime(market_stats["median_ts"], unit="s")
    market_order = market_stats.sort_values(
        "median_ts", na_position="last"
    ).reset_index(drop=True)

    train_markets = market_order.loc[
        market_order["split"] == "train", GROUP_COL
    ].tolist()
    val_markets = market_order.loc[
        market_order["split"] == "val", GROUP_COL
    ].tolist()
    test_markets = market_order.loc[
        market_order["split"] == "test", GROUP_COL
    ].tolist()
    return train_markets, val_markets, test_markets, market_order


def cv_on_train(
    df_train: pd.DataFrame,
    feature_cols: list[str],
    train_market_order: list[str],
) -> dict:
    """Expanding-window CV over time-ordered training markets.

    Each fold:
      - train rows = rows whose market is in the older block
      - val rows   = rows whose market is in the next block

    No market is split across the two sides -> no within-market leakage.
    """
    n_markets = len(train_market_order)
    splitter = TimeSeriesSplit(n_splits=N_CV_FOLDS)

    roc_aucs: list[float] = []
    pr_aucs: list[float] = []
    fold_summaries: list[dict] = []

    print(f"[cb2] running {N_CV_FOLDS}-fold expanding-window CV on "
          f"{n_markets} train markets")
    for fold_idx, (train_mk_idx, val_mk_idx) in enumerate(
        splitter.split(np.arange(n_markets)), start=1
    ):
        train_mks = [train_market_order[i] for i in train_mk_idx]
        val_mks = [train_market_order[i] for i in val_mk_idx]

        tr_mask = df_train[GROUP_COL].isin(train_mks)
        va_mask = df_train[GROUP_COL].isin(val_mks)
        X_tr = df_train.loc[tr_mask, feature_cols].astype(float)
        y_tr = df_train.loc[tr_mask, LABEL_COL].astype(int)
        X_va = df_train.loc[va_mask, feature_cols].astype(float)
        y_va = df_train.loc[va_mask, LABEL_COL].astype(int)

        n_pos = int((y_tr == 1).sum())
        n_neg = int((y_tr == 0).sum())
        spw = (
            SCALE_POS_WEIGHT_OVERRIDE
            if SCALE_POS_WEIGHT_OVERRIDE is not None
            else n_neg / max(n_pos, 1)
        )

        model = CatBoostClassifier(
            iterations=MAX_ITERATIONS,
            depth=4,
            learning_rate=0.05,
            l2_leaf_reg=1.0,
            subsample=0.9,
            bootstrap_type="Bernoulli",
            rsm=0.9,
            scale_pos_weight=spw,
            eval_metric="PRAUC",
            random_seed=RANDOM_STATE,
            thread_count=-1,
            allow_writing_files=False,
            verbose=0,
        )
        # Use a tiny tail of train as the early-stopping signal (last 15% by
        # market order). We don't peek at val markets here -- that would leak
        # val info into the fold's training. Skip early stopping if too few
        # positives to make the signal meaningful.
        n_tail = max(1, int(round(len(train_mks) * 0.20)))
        es_mks = train_mks[-n_tail:]
        inner_train_mks = train_mks[:-n_tail]
        if (
            inner_train_mks
            and (df_train.loc[df_train[GROUP_COL].isin(es_mks), LABEL_COL] == 1).sum() > 0
        ):
            X_in = df_train.loc[df_train[GROUP_COL].isin(inner_train_mks), feature_cols].astype(float)
            y_in = df_train.loc[df_train[GROUP_COL].isin(inner_train_mks), LABEL_COL].astype(int)
            X_es = df_train.loc[df_train[GROUP_COL].isin(es_mks), feature_cols].astype(float)
            y_es = df_train.loc[df_train[GROUP_COL].isin(es_mks), LABEL_COL].astype(int)
            model.fit(
                X_in, y_in,
                eval_set=(X_es, y_es),
                early_stopping_rounds=EARLY_STOPPING_ROUNDS,
                verbose=0,
            )
        else:
            model.fit(X_tr, y_tr, verbose=0)

        proba_va = model.predict_proba(X_va)[:, 1]
        fold_roc = float("nan")
        fold_pr = float("nan")
        if len(np.unique(y_va)) >= 2:
            fold_roc = float(roc_auc_score(y_va, proba_va))
            fold_pr = float(average_precision_score(y_va, proba_va))
            roc_aucs.append(fold_roc)
            pr_aucs.append(fold_pr)

        print(f"  fold {fold_idx}: "
              f"train_mks={len(train_mks)} (rows={len(X_tr)}, pos={n_pos}) "
              f"val_mks={len(val_mks)} (rows={len(X_va)}, "
              f"pos={int((y_va == 1).sum())})  "
              f"ROC-AUC={fold_roc:.4f}  PR-AUC={fold_pr:.4f}  "
              f"best_iter={model.get_best_iteration()}")

        fold_summaries.append({
            "fold": fold_idx,
            "n_train_markets": len(train_mks),
            "n_val_markets": len(val_mks),
            "n_train_rows": int(len(X_tr)),
            "n_val_rows": int(len(X_va)),
            "n_val_positives": int((y_va == 1).sum()),
            "roc_auc": fold_roc,
            "pr_auc": fold_pr,
            "best_iter": int(model.get_best_iteration() or 0),
        })

    summary = {
        "folds": fold_summaries,
        "roc_auc_mean": float(np.mean(roc_aucs)) if roc_aucs else float("nan"),
        "roc_auc_std": float(np.std(roc_aucs)) if roc_aucs else float("nan"),
        "pr_auc_mean": float(np.mean(pr_aucs)) if pr_aucs else float("nan"),
        "pr_auc_std": float(np.std(pr_aucs)) if pr_aucs else float("nan"),
        "n_folds_with_metric": len(roc_aucs),
    }
    print(f"[cb2] CV ROC-AUC = {summary['roc_auc_mean']:.4f} "
          f"± {summary['roc_auc_std']:.4f}  "
          f"(over {summary['n_folds_with_metric']} usable folds)")
    print(f"[cb2] CV PR-AUC  = {summary['pr_auc_mean']:.4f} "
          f"± {summary['pr_auc_std']:.4f}")
    return summary


def eval_split(
    proba: np.ndarray, y: np.ndarray, label: str, thr: float,
) -> dict:
    out = {"n": int(len(proba)), "n_positives": int((y == 1).sum())}
    if len(np.unique(y)) >= 2:
        out["roc_auc"] = float(roc_auc_score(y, proba))
        out["pr_auc"] = float(average_precision_score(y, proba))
    else:
        out["roc_auc"] = float("nan")
        out["pr_auc"] = float("nan")
    cls = classification_metrics(y, proba, thr)
    out["precision"] = cls["precision"]
    out["recall"] = cls["recall"]
    out["f1"] = cls["f1"]
    print(f"[cb2] {label:5s}  n={out['n']:4d} pos={out['n_positives']:3d}  "
          f"ROC-AUC={out['roc_auc']:.4f}  PR-AUC={out['pr_auc']:.4f}  "
          f"P={cls['precision']:.4f}  R={cls['recall']:.4f}  "
          f"F1={cls['f1']:.4f}  (thr={thr})")
    return out


class Tee:
    """Mirror writes to multiple streams (stdout + log file)."""

    def __init__(self, *streams):
        self.streams = streams

    def write(self, data):
        for s in self.streams:
            s.write(data)
            s.flush()
        return len(data)

    def flush(self):
        for s in self.streams:
            s.flush()


def classification_metrics(
    y: np.ndarray, proba: np.ndarray, thr: float,
) -> dict:
    pred = (proba >= thr).astype(int)
    tp = int(((pred == 1) & (y == 1)).sum())
    fp = int(((pred == 1) & (y == 0)).sum())
    fn = int(((pred == 0) & (y == 1)).sum())
    tn = int(((pred == 0) & (y == 0)).sum())
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    f1 = (2 * precision * recall) / max(precision + recall, 1e-12)
    return {
        "threshold": float(thr),
        "tp": tp, "fp": fp, "fn": fn, "tn": tn,
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
    }


def main() -> None:
    ts = time.strftime("%Y%m%d_%H%M%S")
    log_path = LOG_DIR / f"{LOG_PREFIX}_{ts}.log"
    log_file = open(log_path, "w")
    saved_stdout, saved_stderr = sys.stdout, sys.stderr
    sys.stdout = Tee(saved_stdout, log_file)
    sys.stderr = Tee(saved_stderr, log_file)
    try:
        _main(ts, log_path)
    finally:
        sys.stdout, sys.stderr = saved_stdout, saved_stderr
        log_file.close()
        print(f"[cb2] log written to {log_path}")


def _main(ts: str, log_path: Path) -> None:
    print(f"[cb2] run timestamp = {ts}")
    print(f"[cb2] logging to {log_path}")
    print(f"[cb2] loading {TRAIN_PARQUET}")
    df = pd.read_parquet(TRAIN_PARQUET)
    print(f"[cb2] full shape={df.shape}  "
          f"positives={int((df[LABEL_COL] == 1).sum())}  "
          f"negatives={int((df[LABEL_COL] == 0).sum())}")

    feature_cols = get_top_30_features()
    missing = [c for c in feature_cols if c not in df.columns]
    if missing:
        raise SystemExit(f"missing feature columns: {missing}")

    train_mks, val_mks, test_mks, market_order = split_markets_by_time(df)
    print(f"[cb2] markets total: {len(market_order)}  "
          f"(train={len(train_mks)}, val={len(val_mks)}, test={len(test_mks)})")
    print("[cb2] market order (oldest -> newest), split assignment:")
    for _, r in market_order.iterrows():
        date_str = r["date"].strftime("%Y-%m-%d") if pd.notna(r["date"]) else "NaT       "
        print(f"  {r['split']:5s}  {date_str}  {r[GROUP_COL]}")

    df_train = df[df[GROUP_COL].isin(train_mks)].copy()
    df_val = df[df[GROUP_COL].isin(val_mks)].copy()
    df_test = df[df[GROUP_COL].isin(test_mks)].copy()

    n_total = len(df_train) + len(df_val) + len(df_test)
    pos_total = (
        int((df_train[LABEL_COL] == 1).sum())
        + int((df_val[LABEL_COL] == 1).sum())
        + int((df_test[LABEL_COL] == 1).sum())
    )
    pos_rate_total = 100 * pos_total / max(n_total, 1)
    print(f"[cb2] row distribution (total n={n_total}, pos={pos_total}, "
          f"overall pos rate={pos_rate_total:.2f}%):")
    for label, sub in [("train", df_train), ("val", df_val), ("test", df_test)]:
        n = len(sub)
        pos = int((sub[LABEL_COL] == 1).sum())
        pct_total = 100 * n / max(n_total, 1)
        pct_pos = 100 * pos / max(n, 1)
        print(f"  {label:5s}  n={n:4d}  ({pct_total:5.2f}% of total)  "
              f"pos={pos:3d}  ({pct_pos:5.2f}% pos rate within split)")

    cv_summary = cv_on_train(df_train, feature_cols, train_mks)

    X_train = df_train[feature_cols].astype(float)
    y_train = df_train[LABEL_COL].astype(int)
    X_val = df_val[feature_cols].astype(float)
    y_val = df_val[LABEL_COL].astype(int)
    X_test = df_test[feature_cols].astype(float)
    y_test = df_test[LABEL_COL].astype(int)

    n_pos = int((y_train == 1).sum())
    n_neg = int((y_train == 0).sum())
    auto_spw = n_neg / max(n_pos, 1)
    scale_pos_weight = (
        SCALE_POS_WEIGHT_OVERRIDE
        if SCALE_POS_WEIGHT_OVERRIDE is not None
        else auto_spw
    )
    print(f"[cb2] final train scale_pos_weight = {scale_pos_weight:.3f}  "
          f"(auto-balanced would be {auto_spw:.3f})")

    model = CatBoostClassifier(
        iterations=MAX_ITERATIONS,
        depth=4,
        learning_rate=0.05,
        l2_leaf_reg=1.0,
        subsample=0.9,
        bootstrap_type="Bernoulli",
        rsm=0.9,
        scale_pos_weight=scale_pos_weight,
        eval_metric="PRAUC",
        random_seed=RANDOM_STATE,
        thread_count=-1,
        allow_writing_files=False,
        verbose=0,
    )

    t0 = time.time()
    model.fit(
        X_train, y_train,
        eval_set=(X_val, y_val),
        early_stopping_rounds=EARLY_STOPPING_ROUNDS,
        use_best_model=True,
        verbose=100,
    )
    best_iter = int(model.get_best_iteration() or 0)
    print(f"[cb2] fit in {time.time() - t0:.2f}s  best_iter={best_iter} / "
          f"{MAX_ITERATIONS}")
    print()

    proba_train_raw = model.predict_proba(X_train)[:, 1]
    proba_val_raw = model.predict_proba(X_val)[:, 1]
    proba_test_raw = model.predict_proba(X_test)[:, 1]

    # Platt-scaling calibrator: fit on val so test stays unseen.
    calibrator = LogisticRegression(C=1.0, solver="lbfgs")
    calibrator.fit(proba_val_raw.reshape(-1, 1), y_val.values)
    cal_coef = float(calibrator.coef_[0][0])
    cal_intercept = float(calibrator.intercept_[0])
    print(f"[cb2] calibration: sigmoid({cal_coef:.3f} * raw_prob "
          f"{cal_intercept:+.3f})  (fit on val)")

    proba_train = calibrator.predict_proba(proba_train_raw.reshape(-1, 1))[:, 1]
    proba_val = calibrator.predict_proba(proba_val_raw.reshape(-1, 1))[:, 1]
    proba_test = calibrator.predict_proba(proba_test_raw.reshape(-1, 1))[:, 1]

    def _rng(a: np.ndarray) -> str:
        return f"[{a.min():.3f}, {a.max():.3f}]"
    print(f"[cb2] raw  prob range  train={_rng(proba_train_raw)}  "
          f"val={_rng(proba_val_raw)}  test={_rng(proba_test_raw)}")
    print(f"[cb2] cal. prob range  train={_rng(proba_train)}  "
          f"val={_rng(proba_val)}  test={_rng(proba_test)}")
    print()

    print("[cb2] metrics per split (overfit check: train >> val => overfit):")
    train_metrics = eval_split(proba_train, y_train.values, "train", THRESHOLD)
    val_metrics = eval_split(proba_val, y_val.values, "val", THRESHOLD)
    test_metrics = eval_split(proba_test, y_test.values, "test", THRESHOLD)
    print()

    train_val_gap = train_metrics["pr_auc"] - val_metrics["pr_auc"]
    val_test_gap = val_metrics["pr_auc"] - test_metrics["pr_auc"]
    print(f"[cb2] PR-AUC gap train-val = {train_val_gap:+.4f}  "
          f"(big positive = overfit train)")
    print(f"[cb2] PR-AUC gap val-test  = {val_test_gap:+.4f}  "
          f"(big positive = overfit val via early stopping)")

    train_cls = classification_metrics(y_train.values, proba_train, THRESHOLD)
    val_cls = classification_metrics(y_val.values, proba_val, THRESHOLD)
    test_cls = classification_metrics(y_test.values, proba_test, THRESHOLD)

    print()
    print(f"[cb2] classification metrics @ fixed thr={THRESHOLD}:")
    for label, m in [("train", train_cls), ("val", val_cls), ("test", test_cls)]:
        print(f"  {label:5s}  P={m['precision']:.4f}  R={m['recall']:.4f}  "
              f"F1={m['f1']:.4f}  "
              f"(TP={m['tp']}, FP={m['fp']}, FN={m['fn']}, TN={m['tn']})")

    print()
    print(f"[cb2] TEST confusion matrix @ thr={THRESHOLD}")
    pred_test = (proba_test >= THRESHOLD).astype(int)
    print(confusion_matrix(y_test, pred_test))
    print()
    print(classification_report(y_test, pred_test, digits=4, zero_division=0))

    importances = sorted(
        zip(feature_cols, model.get_feature_importance()),
        key=lambda kv: kv[1],
        reverse=True,
    )
    print("[cb2] feature importances:")
    for name, imp in importances:
        print(f"  {imp:7.4f}  {name}")

    model_path = MODEL_DIR / f"cb_insider_30feat_v2_{ts}.joblib"
    meta_path = MODEL_DIR / f"cb_insider_30feat_v2_{ts}.meta.json"
    calibrator_path = MODEL_DIR / f"cb_insider_30feat_v2_{ts}.calibrator.joblib"
    joblib.dump(model, model_path)
    joblib.dump(calibrator, calibrator_path)

    meta = {
        "trained_at": ts,
        "model_type": "catboost.CatBoostClassifier",
        "training_design": "time-ordered grouped train/val/test + CV",
        "group_col": GROUP_COL,
        "time_col": TIME_COL,
        "split_strategy": (
            "pinned test markets; train = older-than-test by median ts; "
            "val = everything else"
        ),
        "test_markets_pinned": list(TEST_MARKETS),
        "n_train_markets": len(train_mks),
        "n_val_markets": len(val_mks),
        "n_test_markets": len(test_mks),
        "n_train_rows": int(len(df_train)),
        "n_val_rows": int(len(df_val)),
        "n_test_rows": int(len(df_test)),
        "train_markets": train_mks,
        "val_markets": val_mks,
        "test_markets": test_mks,
        "n_features": len(feature_cols),
        "features": feature_cols,
        "feature_selection": f"top 30 from {FEATURE_SOURCE_META.name}",
        "scale_pos_weight": scale_pos_weight,
        "threshold": THRESHOLD,
        "calibrator_file": calibrator_path.name,
        "calibration": {
            "method": "Platt scaling (sklearn LogisticRegression on raw prob)",
            "fit_on": "validation set",
            "coef": cal_coef,
            "intercept": cal_intercept,
        },
        "model_params": {
            "iterations": MAX_ITERATIONS,
            "depth": 4,
            "learning_rate": 0.05,
            "l2_leaf_reg": 1.0,
            "subsample": 0.9,
            "bootstrap_type": "Bernoulli",
            "rsm": 0.9,
            "scale_pos_weight": scale_pos_weight,
            "eval_metric": "PRAUC",
            "random_seed": RANDOM_STATE,
            "early_stopping_rounds": EARLY_STOPPING_ROUNDS,
            "best_iteration": best_iter,
        },
        "cv": {
            "scheme": "TimeSeriesSplit on time-ordered training markets",
            "n_folds": N_CV_FOLDS,
            **cv_summary,
        },
        "metrics": {
            "train": {**train_metrics, "classification_at_threshold": train_cls},
            "val": {**val_metrics, "classification_at_threshold": val_cls},
            "test": {**test_metrics, "classification_at_threshold": test_cls},
            "train_val_pr_auc_gap": float(train_val_gap),
            "val_test_pr_auc_gap": float(val_test_gap),
            # legacy keys for parity with prior meta schemas
            "roc_auc": test_metrics["roc_auc"],
            "pr_auc": test_metrics["pr_auc"],
        },
        "log_file": str(log_path),
        "source_training_data": str(TRAIN_PARQUET),
    }
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)

    latest_model = MODEL_DIR / "cb_insider_30feat_v2_latest.joblib"
    latest_meta = MODEL_DIR / "cb_insider_30feat_v2_latest.meta.json"
    latest_calibrator = MODEL_DIR / "cb_insider_30feat_v2_latest.calibrator.joblib"
    shutil.copy(model_path, latest_model)
    shutil.copy(meta_path, latest_meta)
    shutil.copy(calibrator_path, latest_calibrator)

    print()
    print(f"[cb2] saved {model_path}")
    print(f"[cb2] saved {calibrator_path}")
    print(f"[cb2] saved {meta_path}")
    print(f"[cb2] copied -> {latest_model}")
    print(f"[cb2] copied -> {latest_calibrator}")
    print(f"[cb2] copied -> {latest_meta}")
    print()
    print("[cb2] comparison vs prior CatBoost (random 80/20 split):")
    try:
        with open(MODEL_DIR / "cb_insider_30feat_latest.meta.json") as f:
            prior = json.load(f).get("metrics", {})
        print(f"  prior (rnd 80/20): ROC-AUC={prior.get('roc_auc', 0):.4f}  "
              f"PR-AUC={prior.get('pr_auc', 0):.4f}")
    except FileNotFoundError:
        pass
    print(f"  v2 (time-ordered test): "
          f"ROC-AUC={test_metrics['roc_auc']:.4f}  "
          f"PR-AUC={test_metrics['pr_auc']:.4f}")


if __name__ == "__main__":
    main()
