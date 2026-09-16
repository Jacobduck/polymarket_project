"""RandomForest (top-10 behavioral + top-10 metadata), all-big-Iran removed.

Experiment variant of ``randomforest5.py`` that tests the hypothesis that Iran
markets are corrupting the model, by removing them entirely. Both large Iran
markets are dropped:
    us-strikes-iran-by-march-1-2026-492                    (245 rows / 35 ins)
    us-strikes-iran-by-february-28-2026-227-...            (189 rows / 27 ins)
The 3 tiny Iran markets (ceasefire / israel-strike / khamenei, 63 rows total)
are LEFT IN, per the agreed scope ("only the two big Iran markets"). See
``25ir_randomforest.py`` for the down-sample-only variant.

A single seeded stratified-by-group train/test split (GroupShuffleSplit ~15%)
is used since Iran can no longer be the pinned test market. CROSS-VALIDATION
PR-AUC (GroupKFold over the training set) is the primary metric; held-out test
PR-AUC is a secondary, higher-variance number.

Tuning: both Optuna (TPE) and sklearn RandomizedSearchCV over the same RF
hyperparameter space, scored by GroupKFold PR-AUC; the CV winner is kept.

Outputs:
  cache/models/rf_insider_0ir_<ts>.joblib         (Pipeline)
  cache/models/rf_insider_0ir_<ts>.meta.json
  0ir_randomforest_<ts>_curves.png
  0ir_randomforest_<ts>.log
"""

from __future__ import annotations

import json
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
from sklearn.model_selection import GroupKFold, GroupShuffleSplit, RandomizedSearchCV
from sklearn.pipeline import Pipeline

from polycluster.metadata import METADATA_FEATURE_COLUMNS

CACHE = Path("cache")
TRAIN_PARQUET = CACHE / "training_data_with_metadata.parquet"
MODEL_DIR = CACHE / "models"
MODEL_DIR.mkdir(parents=True, exist_ok=True)

FEATURE_SOURCE_META = MODEL_DIR / "cb_insider_30feat_latest.meta.json"
TOP_K = 10
MODEL_TAG = __import__("os").environ.get("MODEL_TAG_PREFIX", "") + "rf_insider_0ir"
LOG_PREFIX = "0ir_randomforest"
TAG = "rf0ir"

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
TEST_SIZE = 0.15
LOG_DIR = Path(".")

# --- Iran filtering config ------------------------------------------------- #
MAR1_MARKET = "us-strikes-iran-by-march-1-2026-492"
FEB28_PREFIX = "us-strikes-iran-by-february-28-2026-227"

N_CV_FOLDS = 4
N_OPTUNA_TRIALS = 50
N_OPTUNA_STARTUP_TRIALS = 12
N_RANDOM_SEARCH_ITERS = 50


# --------------------------------------------------------------------------- #
# Iran filtering
# --------------------------------------------------------------------------- #
def filter_iran(df: pd.DataFrame) -> pd.DataFrame:
    """Drop both large Iran markets entirely (mar-1 + feb-28)."""
    feb28 = [m for m in df[GROUP_COL].unique() if m.startswith(FEB28_PREFIX)]
    drop_markets = [MAR1_MARKET] + feb28
    present = [m for m in drop_markets if m in set(df[GROUP_COL])]
    if MAR1_MARKET not in present:
        raise SystemExit(f"canonical Iran market not in data: {MAR1_MARKET}")
    dropped_rows = int(df[GROUP_COL].isin(present).sum())
    out = df[~df[GROUP_COL].isin(present)].copy()
    print(f"[{TAG}] Iran filter = 0ir (drop both big Iran markets)")
    for m in present:
        sub = df[df[GROUP_COL] == m]
        print(f"[{TAG}]   dropped {m}: {len(sub)} rows "
              f"({int((sub[LABEL_COL]==1).sum())} ins)")
    print(f"[{TAG}]   dataset: {len(df)} -> {len(out)} rows "
          f"(removed {dropped_rows}); tiny Iran markets kept")
    return out


# --------------------------------------------------------------------------- #
# Features
# --------------------------------------------------------------------------- #
def get_feature_columns() -> list[str]:
    with open(FEATURE_SOURCE_META) as f:
        meta = json.load(f)
    feats = meta["features"]
    if len(feats) < TOP_K:
        raise SystemExit(f"need {TOP_K} features, got {len(feats)}")
    unknown = [c for c in TOP_METADATA_FEATURES if c not in METADATA_FEATURE_COLUMNS]
    if unknown:
        raise SystemExit(f"unknown metadata features: {unknown}")
    behavioral = feats[:TOP_K]
    metadata = list(TOP_METADATA_FEATURES)
    combined = behavioral + metadata
    print(f"[{TAG}] using {len(behavioral)} behavioral + {len(metadata)} metadata "
          f"= {len(combined)} features")
    for i, name in enumerate(behavioral, 1):
        print(f"  beh {i:2d}. {name}")
    for i, name in enumerate(metadata, 1):
        print(f"  met {i:2d}. {name}")
    return combined


# --------------------------------------------------------------------------- #
# Pipeline / tuning
# --------------------------------------------------------------------------- #
def build_pipeline(params: dict) -> Pipeline:
    return Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("clf", RandomForestClassifier(
            random_state=RANDOM_STATE, n_jobs=-1, **params,
        )),
    ])


def _class_weight_from_choice(choice) -> dict | str:
    if choice == "balanced":
        return "balanced"
    return {0: 1.0, 1: float(choice)}


def grouped_cv_pr_auc(
    X: pd.DataFrame, y: pd.Series, groups: pd.Series, params: dict,
) -> tuple[float, list[float]]:
    n_splits = min(N_CV_FOLDS, groups.nunique())
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
            "max_depth": trial.suggest_categorical("max_depth", [None, 3, 5, 8, 12, 20]),
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
    print(f"[{TAG}]   optuna best CV PR-AUC = {best.value:.4f}  (trial {best.number})")
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
        base, param_distributions=param_dist, n_iter=N_RANDOM_SEARCH_ITERS,
        scoring="average_precision", cv=cv, random_state=RANDOM_STATE,
        n_jobs=-1, refit=False, error_score=0.0,
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
    print(f"[{TAG}]   random-search best CV PR-AUC = {search.best_score_:.4f}")
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
        "threshold": float(thr), "tp": tp, "fp": fp, "fn": fn, "tn": tn,
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
    print(f"[{TAG}] {label:6s}  n={out['n']:4d} pos={out['n_positives']:3d}  "
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
    ax_roc.set_title("ROC curves (0ir: big Iran markets removed)")
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
# Main
# --------------------------------------------------------------------------- #
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
        print(f"[{TAG}] log written to {log_path}")


def _main(ts: str, log_path: Path) -> None:
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    print(f"[{TAG}] run timestamp = {ts}")
    print(f"[{TAG}] loading {TRAIN_PARQUET}")
    if not TRAIN_PARQUET.exists():
        raise SystemExit(f"missing {TRAIN_PARQUET}")
    df = pd.read_parquet(TRAIN_PARQUET)
    print(f"[{TAG}] full shape={df.shape}  "
          f"pos={int((df[LABEL_COL] == 1).sum())} neg={int((df[LABEL_COL] == 0).sum())}")

    df = filter_iran(df)
    print(f"[{TAG}] post-filter shape={df.shape}  "
          f"pos={int((df[LABEL_COL] == 1).sum())} neg={int((df[LABEL_COL] == 0).sum())}")

    feature_cols = get_feature_columns()
    missing = [c for c in feature_cols if c not in df.columns]
    if missing:
        raise SystemExit(f"missing feature columns: {missing}")

    # --- single seeded stratified-by-group split ------------------------- #
    gss = GroupShuffleSplit(n_splits=1, test_size=TEST_SIZE, random_state=RANDOM_STATE)
    tr_idx, te_idx = next(gss.split(df, df[LABEL_COL], df[GROUP_COL]))
    df_train = df.iloc[tr_idx].copy()
    df_test = df.iloc[te_idx].copy()

    X_train = df_train[feature_cols].astype(float)
    y_train = df_train[LABEL_COL].astype(int)
    g_train = df_train[GROUP_COL]
    X_test = df_test[feature_cols].astype(float)
    y_test = df_test[LABEL_COL].astype(int)

    print(f"[{TAG}] split (GroupShuffleSplit test_size={TEST_SIZE}, seed={RANDOM_STATE}):")
    print(f"[{TAG}]   train: {len(df_train)} rows, {g_train.nunique()} markets, "
          f"pos={int((y_train==1).sum())} neg={int((y_train==0).sum())}")
    print(f"[{TAG}]   test : {len(df_test)} rows, {df_test[GROUP_COL].nunique()} markets, "
          f"pos={int((y_test==1).sum())}")
    print(f"[{TAG}]   test markets: {sorted(df_test[GROUP_COL].unique())}")

    # --- tune with both methods, keep the better CV PR-AUC --------------- #
    print(f"[{TAG}] tuning Optuna ({N_OPTUNA_TRIALS} trials, GroupKFold={N_CV_FOLDS})...")
    t0 = time.time()
    opt_params, opt_cv = tune_optuna(X_train, y_train, g_train)
    print(f"[{TAG}]   optuna done in {time.time()-t0:.1f}s")

    print(f"[{TAG}] tuning RandomizedSearchCV ({N_RANDOM_SEARCH_ITERS} iters)...")
    t0 = time.time()
    rs_params, rs_cv = tune_random_search(X_train, y_train, g_train)
    print(f"[{TAG}]   random-search done in {time.time()-t0:.1f}s")

    if opt_cv >= rs_cv:
        winner, best_params, best_cv = "optuna", opt_params, opt_cv
    else:
        winner, best_params, best_cv = "random_search", rs_params, rs_cv
    print(f"[{TAG}] tuner winner = {winner}  (optuna={opt_cv:.4f}, "
          f"random_search={rs_cv:.4f})  -> best CV PR-AUC={best_cv:.4f}")
    print(f"[{TAG}] best params: {best_params}")

    # --- per-fold CV with best params (primary metric) ------------------- #
    cv_mean, cv_folds = grouped_cv_pr_auc(X_train, y_train, g_train, best_params)
    print(f"[{TAG}] CV PR-AUC (primary) = {cv_mean:.4f} ± {np.std(cv_folds):.4f}  "
          f"over {len(cv_folds)} folds: {[round(f,4) for f in cv_folds]}")

    # --- refit on full train, evaluate on held-out test ------------------ #
    model = build_pipeline(best_params)
    t0 = time.time()
    model.fit(X_train, y_train)
    print(f"[{TAG}] final fit in {time.time()-t0:.2f}s")

    proba_train = model.predict_proba(X_train)[:, 1]
    proba_test = model.predict_proba(X_test)[:, 1]

    print(f"[{TAG}] metrics per split:")
    train_metrics = eval_split(proba_train, y_train.values, "train", THRESHOLD)
    test_metrics = eval_split(proba_test, y_test.values, "test", THRESHOLD)
    gap = train_metrics["pr_auc"] - test_metrics["pr_auc"]
    cv_test_gap = cv_mean - test_metrics["pr_auc"]
    print(f"[{TAG}] PR-AUC gap train-test = {gap:+.4f}  (big positive = overfit)")
    print(f"[{TAG}] PR-AUC gap CV-test    = {cv_test_gap:+.4f}")

    test_cls = classification_metrics(y_test.values, proba_test, THRESHOLD)
    print(f"[{TAG}] TEST classification @ thr={THRESHOLD}: "
          f"P={test_cls['precision']:.4f} R={test_cls['recall']:.4f} "
          f"F1={test_cls['f1']:.4f} "
          f"(TP={test_cls['tp']}, FP={test_cls['fp']}, "
          f"FN={test_cls['fn']}, TN={test_cls['tn']})")

    print(f"[{TAG}] TEST predictions, sorted by prob desc:")
    print(f"[{TAG}]  rank  prob    label  cum_tp  cum_fp  prec   recall  market  wallet")
    order = np.argsort(-proba_test)
    wallets_test = df_test["wallet"].values if "wallet" in df_test.columns else None
    markets_test = df_test[GROUP_COL].values
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
        print(f"[{TAG}]  {rank:>4}    {proba_test[k]:.4f}    {lab}     "
              f"{cum_tp:>4}    {cum_fp:>4}  {prec:.3f}   {rec:.3f}  "
              f"{markets_test[k][:28]:<28}  {wallet_str}")

    print(f"[{TAG}] TEST confusion matrix @ thr={THRESHOLD}")
    pred_test = (proba_test >= THRESHOLD).astype(int)
    print(confusion_matrix(y_test, pred_test))
    print(classification_report(y_test, pred_test, digits=4, zero_division=0))

    rf_clf = model.named_steps["clf"]
    importances = sorted(zip(feature_cols, rf_clf.feature_importances_),
                         key=lambda kv: kv[1], reverse=True)
    print(f"[{TAG}] feature importances (mean decrease in impurity):")
    for name, imp in importances:
        print(f"  {imp:7.4f}  {name}")

    # --- artifacts ------------------------------------------------------- #
    curves_path = LOG_DIR / f"{LOG_PREFIX}_{ts}_curves.png"
    plot_curves([("train", y_train.values, proba_train),
                 ("test", y_test.values, proba_test)], curves_path, THRESHOLD)
    print(f"[{TAG}] curves saved to {curves_path}")

    model_path = MODEL_DIR / f"{MODEL_TAG}_{ts}.joblib"
    meta_path = MODEL_DIR / f"{MODEL_TAG}_{ts}.meta.json"
    joblib.dump(model, model_path)

    params_json = dict(best_params)
    if isinstance(params_json.get("class_weight"), dict):
        params_json["class_weight"] = {
            str(k): v for k, v in params_json["class_weight"].items()
        }

    meta = {
        "trained_at": ts,
        "model_type": "sklearn.Pipeline[SimpleImputer, RandomForestClassifier]",
        "experiment": "0ir -- both big Iran markets removed (mar-1 + feb-28)",
        "iran_filter": {
            "mode": "0ir",
            "dropped_markets": [MAR1_MARKET, f"{FEB28_PREFIX}..."],
            "note": "tiny Iran markets (ceasefire/israel-strike/khamenei) kept",
        },
        "split_strategy": (
            f"GroupShuffleSplit(test_size={TEST_SIZE}, random_state={RANDOM_STATE}) "
            "by market_slug"
        ),
        "n_train_markets": int(g_train.nunique()),
        "n_train_rows": int(len(df_train)),
        "n_test_rows": int(len(df_test)),
        "test_markets": sorted(df_test[GROUP_COL].unique().tolist()),
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
        "cv": {
            "scheme": f"GroupKFold(n_splits={N_CV_FOLDS}) by market_slug, PR-AUC",
            "pr_auc_mean": cv_mean,
            "pr_auc_std": float(np.std(cv_folds)),
            "pr_auc_folds": cv_folds,
        },
        "model_params": params_json,
        "metrics": {
            "cv_pr_auc": cv_mean,
            "train": {**train_metrics, "classification_at_threshold":
                      classification_metrics(y_train.values, proba_train, THRESHOLD)},
            "test": {**test_metrics, "classification_at_threshold": test_cls},
            "train_test_pr_auc_gap": float(gap),
            "cv_test_pr_auc_gap": float(cv_test_gap),
            "roc_auc": test_metrics["roc_auc"],
            "pr_auc": test_metrics["pr_auc"],
        },
        "feature_importances": {n: float(i) for n, i in importances},
        "curves_plot": str(curves_path),
        "log_file": str(log_path),
        "source_training_data": str(TRAIN_PARQUET),
    }
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)
    print(f"[{TAG}] saved {model_path}")
    print(f"[{TAG}] saved {meta_path}")
    print()
    print(f"[{TAG}] === RESULT ===  CV PR-AUC={cv_mean:.4f}  "
          f"TEST PR-AUC={test_metrics['pr_auc']:.4f}  "
          f"TEST ROC-AUC={test_metrics['roc_auc']:.4f}")


if __name__ == "__main__":
    main()
