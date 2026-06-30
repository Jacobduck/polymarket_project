"""RandomForest on top-10 behavioral + 17 metadata features, non-Iran test.

Identical to ``randomforest3.py`` (same dataset, same 27-feature set,
same pipeline / CV / threshold) EXCEPT the pinned test market is
``will-axiom-be-accused-of-insider-trading`` instead of an Iran market.
This checks whether the metadata model generalizes to a market from a
different domain than the Iran cluster the project has leaned on.

Outputs:
  cache/models/rf_insider_meta_v4_<ts>.joblib              (Pipeline)
  cache/models/rf_insider_meta_v4_<ts>.meta.json
  cache/models/rf_insider_meta_v4_latest.joblib            (copies)
  cache/models/rf_insider_meta_v4_latest.meta.json
  randomforest4_<ts>.log                                   (full stdout+stderr)
  randomforest4_<ts>_curves.png
"""

from __future__ import annotations

import json
import shutil
import sys
import time
from pathlib import Path

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    average_precision_score,
    classification_report,
    confusion_matrix,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import TimeSeriesSplit
from sklearn.pipeline import Pipeline

from polycluster.metadata import METADATA_FEATURE_COLUMNS

CACHE = Path("cache")
TRAIN_PARQUET = CACHE / "training_data_with_metadata.parquet"
MODEL_DIR = CACHE / "models"
MODEL_DIR.mkdir(parents=True, exist_ok=True)

FEATURE_SOURCE_META = MODEL_DIR / "cb_insider_30feat_latest.meta.json"
TOP_K = 10
MODEL_TAG = "rf_insider_meta_v4"

LABEL_COL = "is_insider"
GROUP_COL = "market_slug"
TIME_COL = "terminal_start_ts"
THRESHOLD = 0.5
RANDOM_STATE = 42

LOG_DIR = Path(".")
LOG_PREFIX = "randomforest4"

TEST_MARKETS: tuple[str, ...] = (
    "will-axiom-be-accused-of-insider-trading",
)

N_CV_FOLDS = 5

# Matches randomforest2.py / catboost3.py so the metadata lift is measured
# against the same class-weighting baseline (auto-balanced would be ~5.7x).
POSITIVE_CLASS_WEIGHT: float = 2.0

N_ESTIMATORS = 300
MAX_DEPTH: int | None = None
MIN_SAMPLES_SPLIT = 2
MIN_SAMPLES_LEAF = 1
MAX_FEATURES = "sqrt"


def build_pipeline() -> Pipeline:
    """Fresh Pipeline so CV folds train independent imputers (no leakage)."""
    return Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("clf", RandomForestClassifier(
            n_estimators=N_ESTIMATORS,
            max_depth=MAX_DEPTH,
            min_samples_split=MIN_SAMPLES_SPLIT,
            min_samples_leaf=MIN_SAMPLES_LEAF,
            max_features=MAX_FEATURES,
            class_weight={0: 1.0, 1: POSITIVE_CLASS_WEIGHT},
            random_state=RANDOM_STATE,
            n_jobs=-1,
        )),
    ])


def get_feature_columns() -> list[str]:
    """Top-K behavioral features + all 17 metadata features.

    The behavioral block reuses the same top-K (=10) ranking rf2 used
    (first K entries of cb_insider_30feat_latest.meta.json, sorted by xgb
    gain); the metadata block is appended in METADATA_FEATURE_COLUMNS order.
    """
    with open(FEATURE_SOURCE_META) as f:
        meta = json.load(f)
    feats = meta["features"]
    if len(feats) < TOP_K:
        raise SystemExit(
            f"need {TOP_K} features in {FEATURE_SOURCE_META}, got {len(feats)}"
        )
    behavioral = feats[:TOP_K]
    metadata = list(METADATA_FEATURE_COLUMNS)
    combined = behavioral + metadata
    print(f"[rf4] using {len(behavioral)} behavioral + {len(metadata)} metadata "
          f"= {len(combined)} features")
    print(f"[rf4] behavioral (top-{TOP_K} from {FEATURE_SOURCE_META.name}):")
    for i, name in enumerate(behavioral, 1):
        print(f"  {i:2d}. {name}")
    print(f"[rf4] metadata:")
    for i, name in enumerate(metadata, 1):
        print(f"  {i:2d}. {name}")
    return combined


def split_markets_by_time(
    df: pd.DataFrame,
) -> tuple[list[str], list[str], list[str], pd.DataFrame]:
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
    n_markets = len(train_market_order)
    splitter = TimeSeriesSplit(n_splits=N_CV_FOLDS)

    roc_aucs: list[float] = []
    pr_aucs: list[float] = []
    fold_summaries: list[dict] = []

    print(f"[rf4] running {N_CV_FOLDS}-fold expanding-window CV on "
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

        model = build_pipeline()
        model.fit(X_tr, y_tr)

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
              f"ROC-AUC={fold_roc:.4f}  PR-AUC={fold_pr:.4f}")

        fold_summaries.append({
            "fold": fold_idx,
            "n_train_markets": len(train_mks),
            "n_val_markets": len(val_mks),
            "n_train_rows": int(len(X_tr)),
            "n_val_rows": int(len(X_va)),
            "n_val_positives": int((y_va == 1).sum()),
            "roc_auc": fold_roc,
            "pr_auc": fold_pr,
        })

    summary = {
        "folds": fold_summaries,
        "roc_auc_mean": float(np.mean(roc_aucs)) if roc_aucs else float("nan"),
        "roc_auc_std": float(np.std(roc_aucs)) if roc_aucs else float("nan"),
        "pr_auc_mean": float(np.mean(pr_aucs)) if pr_aucs else float("nan"),
        "pr_auc_std": float(np.std(pr_aucs)) if pr_aucs else float("nan"),
        "n_folds_with_metric": len(roc_aucs),
    }
    print(f"[rf4] CV ROC-AUC = {summary['roc_auc_mean']:.4f} "
          f"± {summary['roc_auc_std']:.4f}  "
          f"(over {summary['n_folds_with_metric']} usable folds)")
    print(f"[rf4] CV PR-AUC  = {summary['pr_auc_mean']:.4f} "
          f"± {summary['pr_auc_std']:.4f}")
    return summary


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
    print(f"[rf4] {label:5s}  n={out['n']:4d} pos={out['n_positives']:3d}  "
          f"ROC-AUC={out['roc_auc']:.4f}  PR-AUC={out['pr_auc']:.4f}  "
          f"P={cls['precision']:.4f}  R={cls['recall']:.4f}  "
          f"F1={cls['f1']:.4f}  (thr={thr})")
    return out


def plot_curves(
    splits: list[tuple[str, np.ndarray, np.ndarray]],
    out_path: Path,
    threshold: float,
) -> None:
    fig, (ax_roc, ax_pr) = plt.subplots(1, 2, figsize=(12, 5))
    colors = {"train": "#1f77b4", "val": "#ff7f0e", "test": "#2ca02c"}

    for label, y, proba in splits:
        if len(np.unique(y)) < 2:
            continue
        color = colors.get(label, None)

        fpr, tpr, _ = roc_curve(y, proba)
        auc = roc_auc_score(y, proba)
        ax_roc.scatter(
            fpr, tpr, color=color, s=25, alpha=0.85, zorder=3,
            edgecolor="black", linewidth=0.3,
            label=f"{label}  AUC={auc:.4f}",
        )

        prec, rec, pr_thr = precision_recall_curve(y, proba)
        pr_auc = average_precision_score(y, proba)
        ax_pr.scatter(
            rec, prec, color=color, s=25, alpha=0.85, zorder=3,
            edgecolor="black", linewidth=0.3,
            label=f"{label}  AUC={pr_auc:.4f}",
        )

        idx = int(np.searchsorted(pr_thr, threshold))
        if 0 <= idx < len(rec):
            ax_pr.scatter(
                [rec[idx]], [prec[idx]], color=color, marker="o",
                s=50, zorder=5, edgecolor="black", linewidth=0.8,
            )

        base_rate = float((y == 1).mean())
        ax_pr.axhline(
            base_rate, ls=":", color=color, alpha=0.4, linewidth=0.8,
        )

    ax_roc.plot([0, 1], [0, 1], "k--", alpha=0.3, label="random")
    ax_roc.set_xlabel("False Positive Rate")
    ax_roc.set_ylabel("True Positive Rate")
    ax_roc.set_title("ROC curves")
    ax_roc.legend(loc="lower right")
    ax_roc.grid(alpha=0.3)
    ax_roc.set_xlim(0, 1)
    ax_roc.set_ylim(0, 1.01)

    ax_pr.set_xlabel("Recall")
    ax_pr.set_ylabel("Precision")
    ax_pr.set_title(f"Precision-Recall curves (markers = thr={threshold})")
    ax_pr.legend(loc="lower left")
    ax_pr.grid(alpha=0.3)
    ax_pr.set_xlim(0, 1)
    ax_pr.set_ylim(0, 1.01)

    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


class Tee:
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
        print(f"[rf4] log written to {log_path}")


def _main(ts: str, log_path: Path) -> None:
    print(f"[rf4] run timestamp = {ts}")
    print(f"[rf4] logging to {log_path}")
    print(f"[rf4] loading {TRAIN_PARQUET}")
    if not TRAIN_PARQUET.exists():
        raise SystemExit(
            f"missing {TRAIN_PARQUET}; run build_dataset_with_metadata.py first"
        )
    df = pd.read_parquet(TRAIN_PARQUET)
    print(f"[rf4] full shape={df.shape}  "
          f"positives={int((df[LABEL_COL] == 1).sum())}  "
          f"negatives={int((df[LABEL_COL] == 0).sum())}")

    feature_cols = get_feature_columns()
    missing = [c for c in feature_cols if c not in df.columns]
    if missing:
        raise SystemExit(f"missing feature columns: {missing}")

    train_mks, val_mks, test_mks, market_order = split_markets_by_time(df)
    print(f"[rf4] markets total: {len(market_order)}  "
          f"(train={len(train_mks)}, val={len(val_mks)}, test={len(test_mks)})")
    print("[rf4] market order (oldest -> newest), split assignment:")
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
    print(f"[rf4] row distribution (total n={n_total}, pos={pos_total}, "
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
    print(f"[rf4] final positive class weight = {POSITIVE_CLASS_WEIGHT:.3f}  "
          f"(auto-balanced would be {auto_spw:.3f})")

    model = build_pipeline()

    t0 = time.time()
    model.fit(X_train, y_train)
    print(f"[rf4] fit in {time.time() - t0:.2f}s  "
          f"n_estimators={N_ESTIMATORS}")
    print()

    proba_train = model.predict_proba(X_train)[:, 1]
    proba_val = model.predict_proba(X_val)[:, 1]
    proba_test = model.predict_proba(X_test)[:, 1]

    def _rng(a: np.ndarray) -> str:
        return f"[{a.min():.3f}, {a.max():.3f}]"
    print(f"[rf4] prob range  train={_rng(proba_train)}  "
          f"val={_rng(proba_val)}  test={_rng(proba_test)}")
    print()

    print("[rf4] metrics per split (overfit check: train >> val => overfit):")
    train_metrics = eval_split(proba_train, y_train.values, "train", THRESHOLD)
    val_metrics = eval_split(proba_val, y_val.values, "val", THRESHOLD)
    test_metrics = eval_split(proba_test, y_test.values, "test", THRESHOLD)
    print()

    train_val_gap = train_metrics["pr_auc"] - val_metrics["pr_auc"]
    val_test_gap = val_metrics["pr_auc"] - test_metrics["pr_auc"]
    print(f"[rf4] PR-AUC gap train-val = {train_val_gap:+.4f}  "
          f"(big positive = overfit train)")
    print(f"[rf4] PR-AUC gap val-test  = {val_test_gap:+.4f}  "
          f"(big positive = overfit val)")

    train_cls = classification_metrics(y_train.values, proba_train, THRESHOLD)
    val_cls = classification_metrics(y_val.values, proba_val, THRESHOLD)
    test_cls = classification_metrics(y_test.values, proba_test, THRESHOLD)

    print()
    print(f"[rf4] classification metrics @ fixed thr={THRESHOLD}:")
    for label, m in [("train", train_cls), ("val", val_cls), ("test", test_cls)]:
        print(f"  {label:5s}  P={m['precision']:.4f}  R={m['recall']:.4f}  "
              f"F1={m['f1']:.4f}  "
              f"(TP={m['tp']}, FP={m['fp']}, FN={m['fn']}, TN={m['tn']})")

    curves_path = LOG_DIR / f"{LOG_PREFIX}_{ts}_curves.png"
    plot_curves(
        [
            ("train", y_train.values, proba_train),
            ("val", y_val.values, proba_val),
            ("test", y_test.values, proba_test),
        ],
        curves_path,
        THRESHOLD,
    )
    print(f"[rf4] curves saved to {curves_path}")
    print()

    print(f"[rf4] TEST predictions, one row per wallet, sorted by prob desc.")
    print(f"[rf4]  rank  prob    label  cum_tp  cum_fp  prec   recall  wallet")
    order = np.argsort(-proba_test)
    wallets_test = df_test["wallet"].values if "wallet" in df_test.columns else None
    y_arr = y_test.values
    n_pos_test = int((y_arr == 1).sum())
    cum_tp = 0
    cum_fp = 0
    for rank, k in enumerate(order, start=1):
        lab = int(y_arr[k])
        if lab == 1:
            cum_tp += 1
        else:
            cum_fp += 1
        prec = cum_tp / max(cum_tp + cum_fp, 1)
        rec = cum_tp / max(n_pos_test, 1)
        wallet_str = wallets_test[k] if wallets_test is not None else ""
        print(f"[rf4]  {rank:>4}    {proba_test[k]:.4f}    {lab}     "
              f"{cum_tp:>4}    {cum_fp:>4}  {prec:.3f}   {rec:.3f}  "
              f"{wallet_str}")
    print()

    print(f"[rf4] TEST confusion matrix @ thr={THRESHOLD}")
    pred_test = (proba_test >= THRESHOLD).astype(int)
    print(confusion_matrix(y_test, pred_test))
    print()
    print(classification_report(y_test, pred_test, digits=4, zero_division=0))

    rf_clf = model.named_steps["clf"]
    importances = sorted(
        zip(feature_cols, rf_clf.feature_importances_),
        key=lambda kv: kv[1],
        reverse=True,
    )
    print("[rf4] feature importances (mean decrease in impurity):")
    for name, imp in importances:
        print(f"  {imp:7.4f}  {name}")

    model_path = MODEL_DIR / f"{MODEL_TAG}_{ts}.joblib"
    meta_path = MODEL_DIR / f"{MODEL_TAG}_{ts}.meta.json"
    joblib.dump(model, model_path)

    meta = {
        "trained_at": ts,
        "model_type": "sklearn.Pipeline[SimpleImputer, RandomForestClassifier]",
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
        "feature_selection": (
            f"top {TOP_K} behavioral from {FEATURE_SOURCE_META.name} "
            f"+ {len(METADATA_FEATURE_COLUMNS)} metadata features"
        ),
        "n_behavioral_features": TOP_K,
        "n_metadata_features": len(METADATA_FEATURE_COLUMNS),
        "metadata_features": list(METADATA_FEATURE_COLUMNS),
        "positive_class_weight": POSITIVE_CLASS_WEIGHT,
        "auto_balanced_class_weight": auto_spw,
        "threshold": THRESHOLD,
        "calibration": "none (raw RF proba used)",
        "model_params": {
            "n_estimators": N_ESTIMATORS,
            "max_depth": MAX_DEPTH,
            "min_samples_split": MIN_SAMPLES_SPLIT,
            "min_samples_leaf": MIN_SAMPLES_LEAF,
            "max_features": MAX_FEATURES,
            "class_weight": {0: 1.0, 1: POSITIVE_CLASS_WEIGHT},
            "random_state": RANDOM_STATE,
            "imputer_strategy": "median",
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
            "roc_auc": test_metrics["roc_auc"],
            "pr_auc": test_metrics["pr_auc"],
        },
        "log_file": str(log_path),
        "curves_plot": str(curves_path),
        "source_training_data": str(TRAIN_PARQUET),
    }
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)

    latest_model = MODEL_DIR / f"{MODEL_TAG}_latest.joblib"
    latest_meta = MODEL_DIR / f"{MODEL_TAG}_latest.meta.json"
    shutil.copy(model_path, latest_model)
    shutil.copy(meta_path, latest_meta)

    print()
    print(f"[rf4] saved {model_path}")
    print(f"[rf4] saved {meta_path}")
    print(f"[rf4] copied -> {latest_model}")
    print(f"[rf4] copied -> {latest_meta}")
    print()
    print("[rf4] comparison vs other models (NOT same test market -- "
          "rf3 tests Iran, this tests axiom):")
    comparisons = [
        ("rf  10-feat + meta v3 (Iran test)", MODEL_DIR / "rf_insider_meta_v3_latest.meta.json"),
        ("rf  10-feat v2 (behavioral only)",  MODEL_DIR / "rf_insider_10feat_v2_latest.meta.json"),
        ("cb  10-feat v2 (same behavioral)",  MODEL_DIR / "cb_insider_10feat_v2_latest.meta.json"),
    ]
    for tag, path in comparisons:
        try:
            with open(path) as f:
                m = json.load(f).get("metrics", {})
            print(f"  {tag:<35}  ROC-AUC={m.get('roc_auc', 0):.4f}  "
                  f"PR-AUC={m.get('pr_auc', 0):.4f}")
        except FileNotFoundError:
            pass
    print(f"  {'rf  10-feat + meta v4 (this run)':<35}  "
          f"ROC-AUC={test_metrics['roc_auc']:.4f}  "
          f"PR-AUC={test_metrics['pr_auc']:.4f}")


if __name__ == "__main__":
    main()
