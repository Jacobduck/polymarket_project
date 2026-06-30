"""RandomForest on top-10 behavioral + top-10 metadata features, tuned hard.

Unlike randomforest3/4 (fixed hyperparameters, one pinned test market), this
script:

  * uses 20 features: top-10 behavioral (first 10 of
    cb_insider_30feat_latest.meta.json, xgb-gain order) + the top-10 metadata
    features (ranked by xgb5 gain -- the same set xgboost6/7 use);
  * runs over a DIVERSE set of pinned test markets (geopolitics, insider-meta,
    BTC treasury, prediction-market launch, crypto token launch);
  * for EACH test market, trains on all OTHER markets (same pin design as
    xgboost5/7, so rf-vs-xgb is apples-to-apples) and RE-TUNES from scratch;
  * tries TWO tuners per test market -- Optuna (TPE) and sklearn
    RandomizedSearchCV -- both over the same RF hyperparameter space, scored by
    GroupKFold PR-AUC, and keeps whichever wins on CV;
  * logs everything (per-trial-ish, per-fold, per-split, feature importances,
    sorted test predictions) to one log file and prints a final summary table
    across all test markets.

The NaN-heavy metadata block is median-imputed inside the pipeline (RF can't
ingest NaN natively), with a fresh imputer per CV fold to avoid leakage.

Outputs (per test market <slug8>):
  cache/models/rf_insider_meta_v5_<slug8>_<ts>.joblib       (Pipeline)
  cache/models/rf_insider_meta_v5_<slug8>_<ts>.meta.json
  cache/models/rf_insider_meta_v5_<slug8>_<ts>_curves.png
Plus, for the canonical Iran-mar-1 pin, copies to:
  cache/models/rf_insider_meta_v5_latest.joblib
  cache/models/rf_insider_meta_v5_latest.meta.json
And a run-wide summary:
  cache/models/rf_insider_meta_v5_summary_<ts>.json
  randomforest5_<ts>.log
"""

from __future__ import annotations

import json
import shutil
import sys
import time
import warnings
from pathlib import Path

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import optuna
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
from sklearn.model_selection import GroupKFold, RandomizedSearchCV
from sklearn.pipeline import Pipeline

from polycluster.metadata import METADATA_FEATURE_COLUMNS

CACHE = Path("cache")
TRAIN_PARQUET = CACHE / "training_data_with_metadata.parquet"
MODEL_DIR = CACHE / "models"
MODEL_DIR.mkdir(parents=True, exist_ok=True)

FEATURE_SOURCE_META = MODEL_DIR / "cb_insider_30feat_latest.meta.json"
TOP_K = 10
MODEL_TAG = "rf_insider_meta_v5"

# Top 10 of the 17 metadata features, ranked by xgb5's gain importance
# (same set used by xgboost6.py / xgboost7.py). Drops the 7 weakest, which
# are dominated by the NaN-heavy funder-graph block.
TOP_METADATA_FEATURES: tuple[str, ...] = (
    "herfindahl_index_markets",
    "median_bet_size_usdc",
    "time_since_first_deposit_to_first_trade_seconds",
    "wallet_age_seconds",
    "wallet_to_market_age_ratio",
    "avg_bet_size_usdc",
    "wallet_minus_market_age_seconds",
    "funder_flagged_wallet_count",
    "winrate",
    "n_markets_traded",
)

LABEL_COL = "is_insider"
GROUP_COL = "market_slug"
THRESHOLD = 0.5
RANDOM_STATE = 42

LOG_DIR = Path(".")
LOG_PREFIX = "randomforest5"

# Canonical pin used by rf3/xgb5/xgb7 -- its model is copied to *_latest so the
# comparison sections in the other scripts keep resolving.
CANONICAL_TEST_MARKET = "us-strikes-iran-by-march-1-2026-492"

# Diverse pinned test markets, each with >= 3 positives so PR-AUC is meaningful.
TEST_MARKETS: tuple[str, ...] = (
    "us-strikes-iran-by-march-1-2026-492",            # geopolitics   245 / 35
    "will-axiom-be-accused-of-insider-trading",       # insider-meta   49 /  7
    "microstrategy-announces-1000-btc-purchase-december-2-8",  # BTC   42 /  6
    "will-draftkings-launch-a-prediction-market-in-2025",  # pred mkt  21 /  3
    "will-stable-launch-a-token-in-2025",             # token launch   20 /  4
)

N_CV_FOLDS = 4
N_OPTUNA_TRIALS = 50
N_OPTUNA_STARTUP_TRIALS = 12
N_RANDOM_SEARCH_ITERS = 50


# --------------------------------------------------------------------------- #
# Features
# --------------------------------------------------------------------------- #
def get_feature_columns() -> list[str]:
    """Top-K (=10) behavioral features + top-10 metadata features."""
    with open(FEATURE_SOURCE_META) as f:
        meta = json.load(f)
    feats = meta["features"]
    if len(feats) < TOP_K:
        raise SystemExit(
            f"need {TOP_K} features in {FEATURE_SOURCE_META}, got {len(feats)}"
        )
    unknown = [c for c in TOP_METADATA_FEATURES if c not in METADATA_FEATURE_COLUMNS]
    if unknown:
        raise SystemExit(f"unknown metadata features: {unknown}")
    behavioral = feats[:TOP_K]
    metadata = list(TOP_METADATA_FEATURES)
    combined = behavioral + metadata
    print(f"[rf5] using {len(behavioral)} behavioral + {len(metadata)} metadata "
          f"= {len(combined)} features")
    print(f"[rf5] behavioral (top-{TOP_K} from {FEATURE_SOURCE_META.name}):")
    for i, name in enumerate(behavioral, 1):
        print(f"  {i:2d}. {name}")
    print("[rf5] metadata (top 10 of 17, ranked by xgb5 gain):")
    for i, name in enumerate(metadata, 1):
        print(f"  {i:2d}. {name}")
    return combined


# --------------------------------------------------------------------------- #
# Pipeline / tuning
# --------------------------------------------------------------------------- #
def build_pipeline(params: dict) -> Pipeline:
    """Fresh Pipeline so CV folds train independent imputers (no leakage)."""
    return Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("clf", RandomForestClassifier(
            random_state=RANDOM_STATE,
            n_jobs=-1,
            **params,
        )),
    ])


def _class_weight_from_choice(choice) -> dict | str:
    if choice == "balanced":
        return "balanced"
    return {0: 1.0, 1: float(choice)}


def grouped_cv_pr_auc(
    X: pd.DataFrame, y: pd.Series, groups: pd.Series, params: dict,
) -> tuple[float, list[float]]:
    """Mean PR-AUC over GroupKFold folds (folds with one class skipped)."""
    n_groups = groups.nunique()
    n_splits = min(N_CV_FOLDS, n_groups)
    splitter = GroupKFold(n_splits=n_splits)
    fold_pr: list[float] = []
    for tr_idx, va_idx in splitter.split(X, y, groups):
        y_va = y.iloc[va_idx]
        if y_va.nunique() < 2:
            continue
        model = build_pipeline(params)
        model.fit(X.iloc[tr_idx], y.iloc[tr_idx])
        proba = model.predict_proba(X.iloc[va_idx])[:, 1]
        fold_pr.append(float(average_precision_score(y_va, proba)))
    mean_pr = float(np.mean(fold_pr)) if fold_pr else float("nan")
    return mean_pr, fold_pr


def tune_optuna(
    X: pd.DataFrame, y: pd.Series, groups: pd.Series,
) -> tuple[dict, float]:
    def objective(trial: optuna.Trial) -> float:
        cw = trial.suggest_categorical(
            "class_weight_choice", ["balanced", 1.0, 2.0, 3.0, 4.0, 6.0, 8.0]
        )
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 200, 800, step=100),
            "max_depth": trial.suggest_categorical(
                "max_depth", [None, 3, 5, 8, 12, 20]
            ),
            "min_samples_split": trial.suggest_int("min_samples_split", 2, 20),
            "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 10),
            "max_features": trial.suggest_categorical(
                "max_features", ["sqrt", "log2", 0.3, 0.5, 0.7]
            ),
            "class_weight": _class_weight_from_choice(cw),
        }
        mean_pr, _ = grouped_cv_pr_auc(X, y, groups, params)
        return mean_pr if np.isfinite(mean_pr) else 0.0

    sampler = optuna.samplers.TPESampler(
        seed=RANDOM_STATE, n_startup_trials=N_OPTUNA_STARTUP_TRIALS
    )
    study = optuna.create_study(direction="maximize", sampler=sampler)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        study.optimize(objective, n_trials=N_OPTUNA_TRIALS, show_progress_bar=False)

    best = study.best_trial
    params = {
        "n_estimators": best.params["n_estimators"],
        "max_depth": best.params["max_depth"],
        "min_samples_split": best.params["min_samples_split"],
        "min_samples_leaf": best.params["min_samples_leaf"],
        "max_features": best.params["max_features"],
        "class_weight": _class_weight_from_choice(best.params["class_weight_choice"]),
    }
    print(f"[rf5]   optuna best CV PR-AUC = {best.value:.4f}  (trial {best.number})")
    return params, float(best.value)


def tune_random_search(
    X: pd.DataFrame, y: pd.Series, groups: pd.Series,
) -> tuple[dict, float]:
    n_splits = min(N_CV_FOLDS, groups.nunique())
    cv = list(GroupKFold(n_splits=n_splits).split(X, y, groups))
    param_dist = {
        "clf__n_estimators": [200, 300, 400, 500, 600, 700, 800],
        "clf__max_depth": [None, 3, 5, 8, 12, 20],
        "clf__min_samples_split": list(range(2, 21)),
        "clf__min_samples_leaf": list(range(1, 11)),
        "clf__max_features": ["sqrt", "log2", 0.3, 0.5, 0.7],
        "clf__class_weight": [
            "balanced", {0: 1.0, 1: 1.0}, {0: 1.0, 1: 2.0}, {0: 1.0, 1: 3.0},
            {0: 1.0, 1: 4.0}, {0: 1.0, 1: 6.0}, {0: 1.0, 1: 8.0},
        ],
    }
    base = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("clf", RandomForestClassifier(random_state=RANDOM_STATE, n_jobs=-1)),
    ])
    search = RandomizedSearchCV(
        base,
        param_distributions=param_dist,
        n_iter=N_RANDOM_SEARCH_ITERS,
        scoring="average_precision",
        cv=cv,
        random_state=RANDOM_STATE,
        n_jobs=-1,
        refit=False,
        error_score=0.0,
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        search.fit(X, y, groups=groups)

    bp = search.best_params_
    params = {
        "n_estimators": bp["clf__n_estimators"],
        "max_depth": bp["clf__max_depth"],
        "min_samples_split": bp["clf__min_samples_split"],
        "min_samples_leaf": bp["clf__min_samples_leaf"],
        "max_features": bp["clf__max_features"],
        "class_weight": bp["clf__class_weight"],
    }
    print(f"[rf5]   random-search best CV PR-AUC = {search.best_score_:.4f}")
    return params, float(search.best_score_)


# --------------------------------------------------------------------------- #
# Metrics / plotting
# --------------------------------------------------------------------------- #
def classification_metrics(y: np.ndarray, proba: np.ndarray, thr: float) -> dict:
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
        "precision": float(precision), "recall": float(recall), "f1": float(f1),
    }


def eval_split(proba: np.ndarray, y: np.ndarray, label: str, thr: float) -> dict:
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
    print(f"[rf5] {label:6s}  n={out['n']:4d} pos={out['n_positives']:3d}  "
          f"ROC-AUC={out['roc_auc']:.4f}  PR-AUC={out['pr_auc']:.4f}  "
          f"P={cls['precision']:.4f}  R={cls['recall']:.4f}  "
          f"F1={cls['f1']:.4f}  (thr={thr})")
    return out


def plot_curves(
    splits: list[tuple[str, np.ndarray, np.ndarray]], out_path: Path, threshold: float,
) -> None:
    fig, (ax_roc, ax_pr) = plt.subplots(1, 2, figsize=(12, 5))
    colors = {"train": "#1f77b4", "test": "#2ca02c"}
    for label, y, proba in splits:
        if len(np.unique(y)) < 2:
            continue
        color = colors.get(label, None)
        fpr, tpr, _ = roc_curve(y, proba)
        auc = roc_auc_score(y, proba)
        ax_roc.scatter(fpr, tpr, color=color, s=25, alpha=0.85, zorder=3,
                       edgecolor="black", linewidth=0.3,
                       label=f"{label}  AUC={auc:.4f}")
        prec, rec, pr_thr = precision_recall_curve(y, proba)
        pr_auc = average_precision_score(y, proba)
        ax_pr.scatter(rec, prec, color=color, s=25, alpha=0.85, zorder=3,
                      edgecolor="black", linewidth=0.3,
                      label=f"{label}  AUC={pr_auc:.4f}")
        idx = int(np.searchsorted(pr_thr, threshold))
        if 0 <= idx < len(rec):
            ax_pr.scatter([rec[idx]], [prec[idx]], color=color, marker="o",
                          s=50, zorder=5, edgecolor="black", linewidth=0.8)
        base_rate = float((y == 1).mean())
        ax_pr.axhline(base_rate, ls=":", color=color, alpha=0.4, linewidth=0.8)
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


# --------------------------------------------------------------------------- #
# Per-test-market run
# --------------------------------------------------------------------------- #
def _slug8(slug: str) -> str:
    """Short, filesystem-safe stub of a market slug for artifact names."""
    return slug.replace("-", "")[:24] or "market"


def run_one_test_market(
    df: pd.DataFrame, feature_cols: list[str], test_market: str, ts: str,
) -> dict:
    print()
    print("=" * 78)
    print(f"[rf5] TEST MARKET: {test_market}")
    print("=" * 78)

    df_test = df[df[GROUP_COL] == test_market].copy()
    df_train = df[df[GROUP_COL] != test_market].copy()
    if len(df_test) == 0:
        raise SystemExit(f"test market not in data: {test_market}")

    X_train = df_train[feature_cols].astype(float)
    y_train = df_train[LABEL_COL].astype(int)
    g_train = df_train[GROUP_COL]
    X_test = df_test[feature_cols].astype(float)
    y_test = df_test[LABEL_COL].astype(int)

    n_pos_tr = int((y_train == 1).sum())
    n_neg_tr = int((y_train == 0).sum())
    print(f"[rf5] train: {len(df_train)} rows ({g_train.nunique()} markets), "
          f"pos={n_pos_tr}, neg={n_neg_tr}, auto-balanced w={n_neg_tr/max(n_pos_tr,1):.2f}")
    print(f"[rf5] test : {len(df_test)} rows, pos={int((y_test==1).sum())}")

    # --- tune with both methods, keep the better CV PR-AUC --------------- #
    print(f"[rf5] tuning Optuna ({N_OPTUNA_TRIALS} trials, GroupKFold={N_CV_FOLDS})...")
    t0 = time.time()
    opt_params, opt_cv = tune_optuna(X_train, y_train, g_train)
    print(f"[rf5]   optuna done in {time.time()-t0:.1f}s")

    print(f"[rf5] tuning RandomizedSearchCV ({N_RANDOM_SEARCH_ITERS} iters)...")
    t0 = time.time()
    rs_params, rs_cv = tune_random_search(X_train, y_train, g_train)
    print(f"[rf5]   random-search done in {time.time()-t0:.1f}s")

    if opt_cv >= rs_cv:
        winner, best_params, best_cv = "optuna", opt_params, opt_cv
    else:
        winner, best_params, best_cv = "random_search", rs_params, rs_cv
    print(f"[rf5] tuner winner = {winner}  (optuna={opt_cv:.4f}, "
          f"random_search={rs_cv:.4f})  -> best CV PR-AUC={best_cv:.4f}")
    print(f"[rf5] best params: {best_params}")

    # --- refit on full train, evaluate on held-out test ------------------ #
    model = build_pipeline(best_params)
    t0 = time.time()
    model.fit(X_train, y_train)
    print(f"[rf5] final fit in {time.time()-t0:.2f}s")

    proba_train = model.predict_proba(X_train)[:, 1]
    proba_test = model.predict_proba(X_test)[:, 1]

    print("[rf5] metrics per split:")
    train_metrics = eval_split(proba_train, y_train.values, "train", THRESHOLD)
    test_metrics = eval_split(proba_test, y_test.values, "test", THRESHOLD)
    gap = train_metrics["pr_auc"] - test_metrics["pr_auc"]
    print(f"[rf5] PR-AUC gap train-test = {gap:+.4f}  (big positive = overfit)")

    test_cls = classification_metrics(y_test.values, proba_test, THRESHOLD)
    print(f"[rf5] TEST classification @ thr={THRESHOLD}: "
          f"P={test_cls['precision']:.4f} R={test_cls['recall']:.4f} "
          f"F1={test_cls['f1']:.4f} "
          f"(TP={test_cls['tp']}, FP={test_cls['fp']}, "
          f"FN={test_cls['fn']}, TN={test_cls['tn']})")

    # --- sorted test predictions ---------------------------------------- #
    print("[rf5] TEST predictions, one row per wallet, sorted by prob desc:")
    print("[rf5]  rank  prob    label  cum_tp  cum_fp  prec   recall  wallet")
    order = np.argsort(-proba_test)
    wallets_test = df_test["wallet"].values if "wallet" in df_test.columns else None
    y_arr = y_test.values
    n_pos_test = int((y_arr == 1).sum())
    cum_tp = cum_fp = 0
    for rank, k in enumerate(order, start=1):
        lab = int(y_arr[k])
        if lab == 1:
            cum_tp += 1
        else:
            cum_fp += 1
        prec = cum_tp / max(cum_tp + cum_fp, 1)
        rec = cum_tp / max(n_pos_test, 1)
        wallet_str = wallets_test[k] if wallets_test is not None else ""
        print(f"[rf5]  {rank:>4}    {proba_test[k]:.4f}    {lab}     "
              f"{cum_tp:>4}    {cum_fp:>4}  {prec:.3f}   {rec:.3f}  {wallet_str}")

    print(f"[rf5] TEST confusion matrix @ thr={THRESHOLD}")
    pred_test = (proba_test >= THRESHOLD).astype(int)
    print(confusion_matrix(y_test, pred_test))
    print(classification_report(y_test, pred_test, digits=4, zero_division=0))

    rf_clf = model.named_steps["clf"]
    importances = sorted(zip(feature_cols, rf_clf.feature_importances_),
                         key=lambda kv: kv[1], reverse=True)
    print("[rf5] feature importances (mean decrease in impurity):")
    for name, imp in importances:
        print(f"  {imp:7.4f}  {name}")

    # --- artifacts ------------------------------------------------------- #
    slug8 = _slug8(test_market)
    curves_path = LOG_DIR / f"{LOG_PREFIX}_{slug8}_{ts}_curves.png"
    plot_curves([("train", y_train.values, proba_train),
                 ("test", y_test.values, proba_test)], curves_path, THRESHOLD)
    print(f"[rf5] curves saved to {curves_path}")

    model_path = MODEL_DIR / f"{MODEL_TAG}_{slug8}_{ts}.joblib"
    meta_path = MODEL_DIR / f"{MODEL_TAG}_{slug8}_{ts}.meta.json"
    joblib.dump(model, model_path)

    # JSON-safe params (max_features may be float/str; class_weight dict/str)
    params_json = dict(best_params)
    if isinstance(params_json.get("class_weight"), dict):
        params_json["class_weight"] = {
            str(k): v for k, v in params_json["class_weight"].items()
        }

    meta = {
        "trained_at": ts,
        "model_type": "sklearn.Pipeline[SimpleImputer, RandomForestClassifier]",
        "test_market": test_market,
        "split_strategy": "pin one test market; train = all other markets",
        "n_train_markets": int(g_train.nunique()),
        "n_train_rows": int(len(df_train)),
        "n_test_rows": int(len(df_test)),
        "n_features": len(feature_cols),
        "features": feature_cols,
        "feature_selection": (
            f"top {TOP_K} behavioral from {FEATURE_SOURCE_META.name} "
            f"+ top {len(TOP_METADATA_FEATURES)} of {len(METADATA_FEATURE_COLUMNS)} "
            f"metadata features (ranked by xgb5 gain)"
        ),
        "n_behavioral_features": TOP_K,
        "n_metadata_features": len(TOP_METADATA_FEATURES),
        "metadata_features": list(TOP_METADATA_FEATURES),
        "threshold": THRESHOLD,
        "calibration": "none (raw RF proba used)",
        "tuning": {
            "cv_scheme": f"GroupKFold(n_splits={N_CV_FOLDS}) by market, PR-AUC",
            "optuna_trials": N_OPTUNA_TRIALS,
            "optuna_cv_pr_auc": opt_cv,
            "random_search_iters": N_RANDOM_SEARCH_ITERS,
            "random_search_cv_pr_auc": rs_cv,
            "winner": winner,
            "best_cv_pr_auc": best_cv,
        },
        "model_params": params_json,
        "metrics": {
            "train": {**train_metrics, "classification_at_threshold":
                      classification_metrics(y_train.values, proba_train, THRESHOLD)},
            "test": {**test_metrics, "classification_at_threshold": test_cls},
            "train_test_pr_auc_gap": float(gap),
            "roc_auc": test_metrics["roc_auc"],
            "pr_auc": test_metrics["pr_auc"],
        },
        "feature_importances": {n: float(i) for n, i in importances},
        "curves_plot": str(curves_path),
        "source_training_data": str(TRAIN_PARQUET),
    }
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)
    print(f"[rf5] saved {model_path}")
    print(f"[rf5] saved {meta_path}")

    if test_market == CANONICAL_TEST_MARKET:
        shutil.copy(model_path, MODEL_DIR / f"{MODEL_TAG}_latest.joblib")
        shutil.copy(meta_path, MODEL_DIR / f"{MODEL_TAG}_latest.meta.json")
        print(f"[rf5] canonical pin -> copied to {MODEL_TAG}_latest.*")

    return {
        "test_market": test_market,
        "n_test_rows": int(len(df_test)),
        "n_test_pos": int((y_test == 1).sum()),
        "winner": winner,
        "optuna_cv_pr_auc": opt_cv,
        "random_search_cv_pr_auc": rs_cv,
        "best_cv_pr_auc": best_cv,
        "test_roc_auc": test_metrics["roc_auc"],
        "test_pr_auc": test_metrics["pr_auc"],
        "test_precision": test_cls["precision"],
        "test_recall": test_cls["recall"],
        "test_f1": test_cls["f1"],
        "train_test_pr_auc_gap": float(gap),
        "best_params": params_json,
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
        print(f"[rf5] log written to {log_path}")


def _main(ts: str, log_path: Path) -> None:
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    print(f"[rf5] run timestamp = {ts}")
    print(f"[rf5] logging to {log_path}")
    print(f"[rf5] loading {TRAIN_PARQUET}")
    if not TRAIN_PARQUET.exists():
        raise SystemExit(
            f"missing {TRAIN_PARQUET}; run build_dataset_with_metadata.py first"
        )
    df = pd.read_parquet(TRAIN_PARQUET)
    print(f"[rf5] full shape={df.shape}  "
          f"positives={int((df[LABEL_COL] == 1).sum())}  "
          f"negatives={int((df[LABEL_COL] == 0).sum())}")

    feature_cols = get_feature_columns()
    missing = [c for c in feature_cols if c not in df.columns]
    if missing:
        raise SystemExit(f"missing feature columns: {missing}")

    missing_mkts = [m for m in TEST_MARKETS if m not in set(df[GROUP_COL])]
    if missing_mkts:
        raise SystemExit(f"TEST_MARKETS not in data: {missing_mkts}")

    results: list[dict] = []
    for market in TEST_MARKETS:
        results.append(run_one_test_market(df, feature_cols, market, ts))

    # --- run-wide summary ------------------------------------------------ #
    print()
    print("=" * 78)
    print("[rf5] SUMMARY across diverse test markets (20 feat: top-10 behav + "
          "top-10 meta, tuned per market)")
    print("=" * 78)
    hdr = (f"  {'test market':<46} {'n/pos':>8}  {'winner':<13} "
           f"{'CV-PR':>6} {'ROC':>6} {'PR':>6} {'P':>5} {'R':>5} {'F1':>5}")
    print(hdr)
    for r in results:
        npos = f"{r['n_test_rows']}/{r['n_test_pos']}"
        print(f"  {r['test_market'][:46]:<46} {npos:>8}  {r['winner']:<13} "
              f"{r['best_cv_pr_auc']:.3f} {r['test_roc_auc']:.3f} "
              f"{r['test_pr_auc']:.3f} {r['test_precision']:.2f} "
              f"{r['test_recall']:.2f} {r['test_f1']:.2f}")

    finite_pr = [r["test_pr_auc"] for r in results if np.isfinite(r["test_pr_auc"])]
    finite_roc = [r["test_roc_auc"] for r in results if np.isfinite(r["test_roc_auc"])]
    print(f"\n[rf5] mean test PR-AUC  = {np.mean(finite_pr):.4f} "
          f"± {np.std(finite_pr):.4f}  (over {len(finite_pr)} markets)")
    print(f"[rf5] mean test ROC-AUC = {np.mean(finite_roc):.4f} "
          f"± {np.std(finite_roc):.4f}")

    # --- comparison vs other models on the canonical Iran-mar-1 pin ------ #
    canon = next((r for r in results if r["test_market"] == CANONICAL_TEST_MARKET), None)
    if canon is not None:
        print()
        print(f"[rf5] comparison on canonical pin ({CANONICAL_TEST_MARKET}):")
        comparisons = [
            ("xgb 20-feat v2 (behavioral)",        MODEL_DIR / "xgb_insider_20feat_v2_latest.meta.json"),
            ("cb  10-feat v2 (behavioral)",        MODEL_DIR / "cb_insider_10feat_v2_latest.meta.json"),
            ("rf  10-feat + 17meta v3",            MODEL_DIR / "rf_insider_meta_v3_latest.meta.json"),
            ("xgb 10-feat + 17meta v5",            MODEL_DIR / "xgb_insider_meta_v5_latest.meta.json"),
            ("xgb 20-feat + 10meta v7",            MODEL_DIR / "xgb_insider_meta_v7_latest.meta.json"),
            ("xgb metadata-only v6",               MODEL_DIR / "xgb_insider_meta10_v6_latest.meta.json"),
        ]
        for tag, path in comparisons:
            try:
                with open(path) as f:
                    m = json.load(f).get("metrics", {})
                print(f"  {tag:<34}  ROC-AUC={m.get('roc_auc', 0):.4f}  "
                      f"PR-AUC={m.get('pr_auc', 0):.4f}")
            except FileNotFoundError:
                pass
        print(f"  {'rf  10-feat + 10meta v5 (this run)':<34}  "
              f"ROC-AUC={canon['test_roc_auc']:.4f}  "
              f"PR-AUC={canon['test_pr_auc']:.4f}")

    summary_path = MODEL_DIR / f"{MODEL_TAG}_summary_{ts}.json"
    with open(summary_path, "w") as f:
        json.dump({
            "trained_at": ts,
            "feature_cols": feature_cols,
            "n_features": len(feature_cols),
            "test_markets": list(TEST_MARKETS),
            "mean_test_pr_auc": float(np.mean(finite_pr)) if finite_pr else None,
            "mean_test_roc_auc": float(np.mean(finite_roc)) if finite_roc else None,
            "results": results,
            "log_file": str(log_path),
        }, f, indent=2)
    print(f"\n[rf5] summary saved to {summary_path}")


if __name__ == "__main__":
    main()
