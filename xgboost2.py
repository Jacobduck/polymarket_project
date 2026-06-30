"""Train XGBoost on the top-10 features with Optuna-tuned hyperparams.

Design (per user spec):
  - Features: top 10 from cb_insider_30feat_latest.meta.json (xgb gain order).
  - Split: stratified random by market with seed RANDOM_STATE.
    * Test  = 1 randomly-picked LARGE market (>=100 rows) + 5 randomly-picked
              small markets. >1 market in test, includes a big one.
    * Train = the remaining markets (no separate val set).
  - CV: 5-fold ``GroupKFold`` over training markets (random fold assignment,
    NO time ordering), so each fold's holdout is a disjoint set of markets.
  - Tuning: Optuna TPE sampler over 9 XGB hyperparams + threshold-relevant
    ``scale_pos_weight``. Objective = mean PR-AUC across the 5 CV folds.
    ``MedianPruner`` cuts trials whose intermediate fold scores are below
    the running median.
  - Refit best params on the full train set; evaluate on test once.
  - No calibration (raw XGB proba used).

Outputs:
  cache/models/xgb_insider_10feat_v2_<ts>.json              (XGBoost native fmt)
  cache/models/xgb_insider_10feat_v2_<ts>.meta.json
  cache/models/xgb_insider_10feat_v2_latest.json            (copies)
  cache/models/xgb_insider_10feat_v2_latest.meta.json
  xgboost2_<ts>.log                                         (full stdout+stderr)
  xgboost2_<ts>_curves.png
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
import optuna
import pandas as pd
from optuna.pruners import MedianPruner
from optuna.samplers import TPESampler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    classification_report,
    confusion_matrix,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import StratifiedGroupKFold
from xgboost import XGBClassifier

CACHE = Path("cache")
TRAIN_PARQUET = CACHE / "training_data.parquet"
MODEL_DIR = CACHE / "models"
MODEL_DIR.mkdir(parents=True, exist_ok=True)

FEATURE_SOURCE_META = MODEL_DIR / "cb_insider_30feat_latest.meta.json"
TOP_K = 10
MODEL_TAG = f"xgb_insider_{TOP_K}feat_v2"

LABEL_COL = "is_insider"
GROUP_COL = "market_slug"
THRESHOLD = 0.5
RANDOM_STATE = 42

LOG_DIR = Path(".")
LOG_PREFIX = "xgboost2"

LARGE_MARKET_ROW_THRESHOLD = 100
N_TEST_LARGE_MARKETS = 1
N_TEST_SMALL_MARKETS = 5

# If set to a non-empty tuple, override the stratified random split and pin
# these markets as the test set. Used for apples-to-apples comparison vs
# cb3 / cb4 (which both pin ``us-strikes-iran-by-march-1-2026-492``).
PIN_TEST_MARKETS: tuple[str, ...] = (
    "us-strikes-iran-by-march-1-2026-492",
)

N_CV_FOLDS = 5
N_OPTUNA_TRIALS = 100
N_OPTUNA_STARTUP_TRIALS = 20


def get_top_features() -> list[str]:
    """Top-K features from cb_insider_30feat_latest.meta.json."""
    with open(FEATURE_SOURCE_META) as f:
        meta = json.load(f)
    feats = meta["features"]
    if len(feats) < TOP_K:
        raise SystemExit(
            f"need {TOP_K} features in {FEATURE_SOURCE_META}, got {len(feats)}"
        )
    top = feats[:TOP_K]
    print(f"[xgb2] using top-{TOP_K} features from {FEATURE_SOURCE_META.name}:")
    for i, name in enumerate(top, 1):
        print(f"  {i:2d}. {name}")
    return top


def split_markets_stratified_random(
    df: pd.DataFrame, seed: int,
) -> tuple[list[str], list[str], pd.DataFrame]:
    """Train/test split by market.

    If ``PIN_TEST_MARKETS`` is non-empty, those markets are pinned as the
    test set (used for apples-to-apples comparisons vs cb3/cb4).
    Otherwise: markets are partitioned by row count into "large"
    (>= LARGE_MARKET_ROW_THRESHOLD) and "small" (< threshold);
    ``N_TEST_LARGE_MARKETS`` large + ``N_TEST_SMALL_MARKETS`` small markets
    are picked uniformly at random (seeded) for test; the rest become train.
    """
    stats = (
        df.groupby(GROUP_COL)
        .agg(n_rows=(LABEL_COL, "size"), n_pos=(LABEL_COL, "sum"))
        .reset_index()
    )
    large = stats[stats["n_rows"] >= LARGE_MARKET_ROW_THRESHOLD][GROUP_COL].tolist()
    small = stats[stats["n_rows"] < LARGE_MARKET_ROW_THRESHOLD][GROUP_COL].tolist()

    if PIN_TEST_MARKETS:
        test_markets = list(PIN_TEST_MARKETS)
        missing = set(test_markets) - set(stats[GROUP_COL])
        if missing:
            raise SystemExit(
                f"PIN_TEST_MARKETS not in training data: {sorted(missing)}"
            )
        print(f"[xgb2] PIN_TEST_MARKETS active -> "
              f"test pinned to {len(test_markets)} market(s); "
              f"random stratified split disabled")
    else:
        if len(large) < N_TEST_LARGE_MARKETS:
            raise SystemExit(
                f"need {N_TEST_LARGE_MARKETS} large market(s) "
                f"(>={LARGE_MARKET_ROW_THRESHOLD} rows), only have {len(large)}"
            )
        if len(small) < N_TEST_SMALL_MARKETS:
            raise SystemExit(
                f"need {N_TEST_SMALL_MARKETS} small market(s), only have {len(small)}"
            )
        rng = np.random.default_rng(seed)
        large_test = list(rng.choice(large, size=N_TEST_LARGE_MARKETS, replace=False))
        small_test = list(rng.permutation(small)[:N_TEST_SMALL_MARKETS])
        test_markets = large_test + small_test

    train_markets = [
        m for m in stats[GROUP_COL].tolist() if m not in set(test_markets)
    ]

    stats["split"] = stats[GROUP_COL].apply(
        lambda m: "test" if m in set(test_markets) else "train"
    )
    stats["size_band"] = stats[GROUP_COL].apply(
        lambda m: "large" if m in set(large) else "small"
    )
    return train_markets, test_markets, stats


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
    print(f"[xgb2] {label:5s}  n={out['n']:4d} pos={out['n_positives']:3d}  "
          f"ROC-AUC={out['roc_auc']:.4f}  PR-AUC={out['pr_auc']:.4f}  "
          f"P={cls['precision']:.4f}  R={cls['recall']:.4f}  "
          f"F1={cls['f1']:.4f}  (thr={thr})")
    return out


def build_xgb(params: dict) -> XGBClassifier:
    """Construct an XGBClassifier from a tuned param dict."""
    return XGBClassifier(
        n_estimators=int(params["n_estimators"]),
        max_depth=int(params["max_depth"]),
        learning_rate=float(params["learning_rate"]),
        min_child_weight=float(params["min_child_weight"]),
        subsample=float(params["subsample"]),
        colsample_bytree=float(params["colsample_bytree"]),
        reg_lambda=float(params["reg_lambda"]),
        reg_alpha=float(params["reg_alpha"]),
        gamma=float(params["gamma"]),
        scale_pos_weight=float(params["scale_pos_weight"]),
        objective="binary:logistic",
        eval_metric="aucpr",
        tree_method="hist",
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )


def cv_score(
    params: dict,
    X: pd.DataFrame,
    y: pd.Series,
    groups: pd.Series,
    trial: optuna.trial.Trial | None = None,
) -> tuple[float, list[float]]:
    """5-fold GroupKFold PR-AUC for a given param dict.

    If ``trial`` is given, fold-by-fold PR-AUC is reported to Optuna so the
    MedianPruner can drop trials early.
    """
    splitter = StratifiedGroupKFold(
        n_splits=N_CV_FOLDS, shuffle=True, random_state=RANDOM_STATE,
    )
    fold_pr_aucs: list[float] = []
    for fold_idx, (tr_idx, va_idx) in enumerate(splitter.split(X, y, groups)):
        X_tr, X_va = X.iloc[tr_idx], X.iloc[va_idx]
        y_tr, y_va = y.iloc[tr_idx], y.iloc[va_idx]
        model = build_xgb(params)
        model.fit(X_tr, y_tr, verbose=False)
        proba_va = model.predict_proba(X_va)[:, 1]
        if len(np.unique(y_va)) >= 2:
            pr = float(average_precision_score(y_va, proba_va))
        else:
            pr = float("nan")
        fold_pr_aucs.append(pr)

        if trial is not None:
            running_mean = float(np.nanmean(fold_pr_aucs))
            trial.report(running_mean, fold_idx)
            if trial.should_prune():
                raise optuna.TrialPruned()

    mean_pr = float(np.nanmean(fold_pr_aucs))
    return mean_pr, fold_pr_aucs


def make_objective(
    X: pd.DataFrame, y: pd.Series, groups: pd.Series,
):
    """Optuna objective closure: returns mean CV PR-AUC for the trial's params."""

    def objective(trial: optuna.trial.Trial) -> float:
        # Tightened ranges (step-a):
        #   - max_depth capped at 5 to match cb3/cb4 (which use 3-4) and
        #     prevent the depth-7 overfit seen on the random-test run.
        #   - learning_rate ceiling lowered to 0.1 (was 0.3); floor lowered
        #     to 0.005 so Optuna can pair slower lr with more trees.
        #   - n_estimators floor raised to 200 to actually exercise the
        #     low-lr regime (cb v2 uses 2000 iterations).
        #   - reg_lambda / reg_alpha floors raised so trials can't disable
        #     regularization entirely.
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 200, 2000, step=50),
            "max_depth": trial.suggest_int("max_depth", 2, 5),
            "learning_rate": trial.suggest_float("learning_rate", 0.005, 0.1, log=True),
            "min_child_weight": trial.suggest_float("min_child_weight", 1.0, 15.0),
            "subsample": trial.suggest_float("subsample", 0.5, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
            "reg_lambda": trial.suggest_float("reg_lambda", 0.5, 20.0, log=True),
            "reg_alpha": trial.suggest_float("reg_alpha", 1e-3, 5.0, log=True),
            "gamma": trial.suggest_float("gamma", 0.0, 5.0),
            "scale_pos_weight": trial.suggest_float("scale_pos_weight", 1.0, 7.0),
        }
        mean_pr, _ = cv_score(params, X, y, groups, trial=trial)
        return mean_pr

    return objective


def plot_curves(
    splits: list[tuple[str, np.ndarray, np.ndarray]],
    out_path: Path,
    threshold: float,
) -> None:
    fig, (ax_roc, ax_pr) = plt.subplots(1, 2, figsize=(12, 5))
    colors = {"train": "#1f77b4", "test": "#2ca02c"}

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
        print(f"[xgb2] log written to {log_path}")


def _main(ts: str, log_path: Path) -> None:
    print(f"[xgb2] run timestamp = {ts}")
    print(f"[xgb2] logging to {log_path}")
    print(f"[xgb2] loading {TRAIN_PARQUET}")
    df = pd.read_parquet(TRAIN_PARQUET)
    print(f"[xgb2] full shape={df.shape}  "
          f"positives={int((df[LABEL_COL] == 1).sum())}  "
          f"negatives={int((df[LABEL_COL] == 0).sum())}")

    feature_cols = get_top_features()
    missing = [c for c in feature_cols if c not in df.columns]
    if missing:
        raise SystemExit(f"missing feature columns: {missing}")

    train_mks, test_mks, stats = split_markets_stratified_random(df, RANDOM_STATE)
    print(f"[xgb2] markets total: {len(stats)}  "
          f"(train={len(train_mks)}, test={len(test_mks)}; large in test="
          f"{N_TEST_LARGE_MARKETS}, small in test={N_TEST_SMALL_MARKETS})")
    print("[xgb2] market assignment (sorted by row count desc):")
    for _, r in stats.sort_values("n_rows", ascending=False).iterrows():
        print(f"  {r['split']:5s}  {r['size_band']:5s}  "
              f"n={int(r['n_rows']):4d}  pos={int(r['n_pos']):3d}  {r[GROUP_COL]}")

    df_train = df[df[GROUP_COL].isin(train_mks)].copy()
    df_test = df[df[GROUP_COL].isin(test_mks)].copy()

    n_total = len(df_train) + len(df_test)
    pos_total = (
        int((df_train[LABEL_COL] == 1).sum())
        + int((df_test[LABEL_COL] == 1).sum())
    )
    pos_rate_total = 100 * pos_total / max(n_total, 1)
    print(f"[xgb2] row distribution (total n={n_total}, pos={pos_total}, "
          f"overall pos rate={pos_rate_total:.2f}%):")
    for label, sub in [("train", df_train), ("test", df_test)]:
        n = len(sub)
        pos = int((sub[LABEL_COL] == 1).sum())
        pct_total = 100 * n / max(n_total, 1)
        pct_pos = 100 * pos / max(n, 1)
        print(f"  {label:5s}  n={n:4d}  ({pct_total:5.2f}% of total)  "
              f"pos={pos:3d}  ({pct_pos:5.2f}% pos rate within split)")

    X_train = df_train[feature_cols].astype(float).reset_index(drop=True)
    y_train = df_train[LABEL_COL].astype(int).reset_index(drop=True)
    groups_train = df_train[GROUP_COL].reset_index(drop=True)
    X_test = df_test[feature_cols].astype(float).reset_index(drop=True)
    y_test = df_test[LABEL_COL].astype(int).reset_index(drop=True)

    print(f"[xgb2] starting Optuna tuning: {N_OPTUNA_TRIALS} trials, "
          f"TPE sampler ({N_OPTUNA_STARTUP_TRIALS} random startup), "
          f"MedianPruner, {N_CV_FOLDS}-fold GroupKFold CV")
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    sampler = TPESampler(seed=RANDOM_STATE, n_startup_trials=N_OPTUNA_STARTUP_TRIALS)
    pruner = MedianPruner(n_startup_trials=N_OPTUNA_STARTUP_TRIALS, n_warmup_steps=2)
    study = optuna.create_study(
        direction="maximize", sampler=sampler, pruner=pruner,
        study_name=f"{MODEL_TAG}_{ts}",
    )

    t0 = time.time()

    def _log_cb(study_: optuna.Study, trial: optuna.trial.FrozenTrial) -> None:
        best_so_far = study_.best_value if study_.best_trial is not None else float("nan")
        state = trial.state.name
        score_str = f"{trial.value:.4f}" if trial.value is not None else "  pruned"
        print(f"[xgb2]   trial {trial.number:>3}  state={state:<8}  "
              f"score={score_str}  best={best_so_far:.4f}")

    study.optimize(
        make_objective(X_train, y_train, groups_train),
        n_trials=N_OPTUNA_TRIALS,
        callbacks=[_log_cb],
        show_progress_bar=False,
    )
    tuning_secs = time.time() - t0

    best_params = study.best_params
    best_cv_pr_auc = float(study.best_value)
    n_completed = sum(
        1 for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE
    )
    n_pruned = sum(
        1 for t in study.trials if t.state == optuna.trial.TrialState.PRUNED
    )
    print(f"[xgb2] tuning done in {tuning_secs:.1f}s  "
          f"({n_completed} completed, {n_pruned} pruned)")
    print(f"[xgb2] best CV PR-AUC = {best_cv_pr_auc:.4f}")
    print("[xgb2] best params:")
    for k, v in best_params.items():
        if isinstance(v, float):
            print(f"  {k:20s} = {v:.5f}")
        else:
            print(f"  {k:20s} = {v}")

    print()
    print(f"[xgb2] re-running CV with best params to collect OOF preds + per-fold scores")
    splitter = StratifiedGroupKFold(
        n_splits=N_CV_FOLDS, shuffle=True, random_state=RANDOM_STATE,
    )
    oof_preds = np.full(len(X_train), np.nan, dtype=float)
    fold_summaries: list[dict] = []
    best_fold_pr_aucs: list[float] = []
    for fold_idx, (tr_idx, va_idx) in enumerate(
        splitter.split(X_train, y_train, groups_train), start=1
    ):
        tr_mks = sorted(set(groups_train.iloc[tr_idx]))
        va_mks = sorted(set(groups_train.iloc[va_idx]))
        fold_model = build_xgb(best_params)
        fold_model.fit(X_train.iloc[tr_idx], y_train.iloc[tr_idx], verbose=False)
        pr_fold = fold_model.predict_proba(X_train.iloc[va_idx])[:, 1]
        oof_preds[va_idx] = pr_fold
        if len(np.unique(y_train.iloc[va_idx])) >= 2:
            fold_pr = float(average_precision_score(y_train.iloc[va_idx], pr_fold))
        else:
            fold_pr = float("nan")
        best_fold_pr_aucs.append(fold_pr)
        n_va_pos = int((y_train.iloc[va_idx] == 1).sum())
        print(f"  fold {fold_idx}: train_mks={len(tr_mks)} val_mks={len(va_mks)}  "
              f"val_rows={len(va_idx)} val_pos={n_va_pos}  PR-AUC={fold_pr:.4f}")
        fold_summaries.append({
            "fold": fold_idx,
            "n_train_markets": len(tr_mks),
            "n_val_markets": len(va_mks),
            "val_markets": va_mks,
            "n_val_rows": int(len(va_idx)),
            "n_val_positives": n_va_pos,
            "pr_auc": fold_pr,
        })

    if np.isnan(oof_preds).any():
        raise SystemExit("OOF preds incomplete -- some train rows not covered by CV")

    cv_pr_auc_mean = float(np.nanmean(best_fold_pr_aucs))
    cv_pr_auc_std = float(np.nanstd(best_fold_pr_aucs))
    print(f"[xgb2] CV PR-AUC = {cv_pr_auc_mean:.4f} ± {cv_pr_auc_std:.4f}")

    # Fit OOF Platt calibrator: sigmoid(a*raw + b) on uncalibrated CV preds.
    # PR-AUC / ROC-AUC are invariant under this monotonic transform, so they
    # will not move; only thresholded metrics (P / R / F1 at thr=0.5) shift.
    calibrator = LogisticRegression(C=1.0, solver="lbfgs")
    calibrator.fit(oof_preds.reshape(-1, 1), y_train.values)
    cal_coef = float(calibrator.coef_[0][0])
    cal_intercept = float(calibrator.intercept_[0])
    print(f"[xgb2] OOF Platt calibrator: "
          f"sigmoid({cal_coef:.3f} * raw_prob {cal_intercept:+.3f})")
    print()

    print(f"[xgb2] fitting final model on full train ({len(X_train)} rows)")
    final_model = build_xgb(best_params)
    t1 = time.time()
    final_model.fit(X_train, y_train, verbose=False)
    print(f"[xgb2] final fit in {time.time() - t1:.2f}s")
    print()

    proba_train_raw = final_model.predict_proba(X_train)[:, 1]
    proba_test_raw = final_model.predict_proba(X_test)[:, 1]
    proba_train = calibrator.predict_proba(proba_train_raw.reshape(-1, 1))[:, 1]
    proba_test = calibrator.predict_proba(proba_test_raw.reshape(-1, 1))[:, 1]

    def _rng(a: np.ndarray) -> str:
        return f"[{a.min():.3f}, {a.max():.3f}]"
    print(f"[xgb2] raw  prob range  train={_rng(proba_train_raw)}  "
          f"test={_rng(proba_test_raw)}")
    print(f"[xgb2] cal. prob range  train={_rng(proba_train)}  "
          f"test={_rng(proba_test)}")

    raw_test_pr = float(average_precision_score(y_test.values, proba_test_raw)) \
        if len(np.unique(y_test.values)) >= 2 else float("nan")
    raw_test_roc = float(roc_auc_score(y_test.values, proba_test_raw)) \
        if len(np.unique(y_test.values)) >= 2 else float("nan")
    print(f"[xgb2] sanity: TEST PR-AUC raw={raw_test_pr:.4f}  "
          f"(should equal calibrated PR-AUC; AUC is invariant to Platt)")
    print()

    print("[xgb2] metrics per split (overfit check: train >> test => overfit):")
    train_metrics = eval_split(proba_train, y_train.values, "train", THRESHOLD)
    test_metrics = eval_split(proba_test, y_test.values, "test", THRESHOLD)
    print()

    train_test_gap = train_metrics["pr_auc"] - test_metrics["pr_auc"]
    cv_test_gap = cv_pr_auc_mean - test_metrics["pr_auc"]
    print(f"[xgb2] PR-AUC gap train-test = {train_test_gap:+.4f}  "
          f"(big positive = overfit train)")
    print(f"[xgb2] PR-AUC gap CV-test    = {cv_test_gap:+.4f}  "
          f"(big positive = test markets harder than CV folds)")

    train_cls = classification_metrics(y_train.values, proba_train, THRESHOLD)
    test_cls = classification_metrics(y_test.values, proba_test, THRESHOLD)

    print()
    print(f"[xgb2] classification metrics @ fixed thr={THRESHOLD}:")
    for label, m in [("train", train_cls), ("test", test_cls)]:
        print(f"  {label:5s}  P={m['precision']:.4f}  R={m['recall']:.4f}  "
              f"F1={m['f1']:.4f}  "
              f"(TP={m['tp']}, FP={m['fp']}, FN={m['fn']}, TN={m['tn']})")

    curves_path = LOG_DIR / f"{LOG_PREFIX}_{ts}_curves.png"
    plot_curves(
        [
            ("train", y_train.values, proba_train),
            ("test", y_test.values, proba_test),
        ],
        curves_path,
        THRESHOLD,
    )
    print(f"[xgb2] curves saved to {curves_path}")
    print()

    print("[xgb2] TEST predictions, one row per wallet, sorted by prob desc.")
    print("[xgb2]  rank  prob    label  cum_tp  cum_fp  prec   recall  market  wallet")
    order = np.argsort(-proba_test)
    wallets_test = df_test["wallet"].values if "wallet" in df_test.columns else None
    markets_test = df_test[GROUP_COL].values
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
        mkt_str = markets_test[k][:30]
        print(f"[xgb2]  {rank:>4}    {proba_test[k]:.4f}    {lab}     "
              f"{cum_tp:>4}    {cum_fp:>4}  {prec:.3f}   {rec:.3f}  "
              f"{mkt_str:<30}  {wallet_str}")
    print()

    print(f"[xgb2] TEST confusion matrix @ thr={THRESHOLD}")
    pred_test = (proba_test >= THRESHOLD).astype(int)
    print(confusion_matrix(y_test, pred_test))
    print()
    print(classification_report(y_test, pred_test, digits=4, zero_division=0))

    importances = sorted(
        zip(feature_cols, final_model.feature_importances_),
        key=lambda kv: kv[1],
        reverse=True,
    )
    print("[xgb2] feature importances (gain):")
    for name, imp in importances:
        print(f"  {imp:7.4f}  {name}")

    model_path = MODEL_DIR / f"{MODEL_TAG}_{ts}.json"
    meta_path = MODEL_DIR / f"{MODEL_TAG}_{ts}.meta.json"
    calibrator_path = MODEL_DIR / f"{MODEL_TAG}_{ts}.calibrator.joblib"
    final_model.save_model(str(model_path))
    joblib.dump(calibrator, calibrator_path)

    meta = {
        "trained_at": ts,
        "model_type": "xgboost.XGBClassifier",
        "training_design": "stratified random train/test + GroupKFold CV + Optuna",
        "group_col": GROUP_COL,
        "split_strategy": (
            f"pinned test markets = {list(PIN_TEST_MARKETS)}"
            if PIN_TEST_MARKETS
            else (
                f"random by market with seed {RANDOM_STATE}; test = "
                f"{N_TEST_LARGE_MARKETS} large "
                f"(>={LARGE_MARKET_ROW_THRESHOLD} rows) + "
                f"{N_TEST_SMALL_MARKETS} small; train = rest"
            )
        ),
        "pin_test_markets": list(PIN_TEST_MARKETS),
        "n_train_markets": len(train_mks),
        "n_test_markets": len(test_mks),
        "n_train_rows": int(len(df_train)),
        "n_test_rows": int(len(df_test)),
        "train_markets": train_mks,
        "test_markets": test_mks,
        "n_features": len(feature_cols),
        "features": feature_cols,
        "feature_selection": f"top {TOP_K} from {FEATURE_SOURCE_META.name}",
        "threshold": THRESHOLD,
        "calibrator_file": calibrator_path.name,
        "calibration": {
            "method": "Platt scaling (sklearn LogisticRegression on raw prob)",
            "fit_on": "OOF predictions from CV (no leakage; train rows held out)",
            "coef": cal_coef,
            "intercept": cal_intercept,
        },
        "model_params": {
            "objective": "binary:logistic",
            "eval_metric": "aucpr",
            "tree_method": "hist",
            "random_state": RANDOM_STATE,
            **best_params,
        },
        "tuning": {
            "framework": "optuna",
            "sampler": "TPESampler",
            "pruner": "MedianPruner",
            "n_trials_requested": N_OPTUNA_TRIALS,
            "n_trials_completed": n_completed,
            "n_trials_pruned": n_pruned,
            "n_startup_trials": N_OPTUNA_STARTUP_TRIALS,
            "objective_metric": "mean PR-AUC across CV folds",
            "tuning_seconds": tuning_secs,
            "best_cv_pr_auc": best_cv_pr_auc,
        },
        "cv": {
            "scheme": (
                f"StratifiedGroupKFold(n_splits={N_CV_FOLDS}, shuffle=True, "
                f"random_state={RANDOM_STATE}) on market_slug; folds keep "
                "markets intact and stratify by is_insider"
            ),
            "n_folds": N_CV_FOLDS,
            "folds": fold_summaries,
            "pr_auc_mean": cv_pr_auc_mean,
            "pr_auc_std": cv_pr_auc_std,
        },
        "metrics": {
            "train": {**train_metrics, "classification_at_threshold": train_cls},
            "test": {**test_metrics, "classification_at_threshold": test_cls},
            "train_test_pr_auc_gap": float(train_test_gap),
            "cv_test_pr_auc_gap": float(cv_test_gap),
            "roc_auc": test_metrics["roc_auc"],
            "pr_auc": test_metrics["pr_auc"],
        },
        "log_file": str(log_path),
        "curves_plot": str(curves_path),
        "source_training_data": str(TRAIN_PARQUET),
    }
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)

    latest_model = MODEL_DIR / f"{MODEL_TAG}_latest.json"
    latest_meta = MODEL_DIR / f"{MODEL_TAG}_latest.meta.json"
    latest_calibrator = MODEL_DIR / f"{MODEL_TAG}_latest.calibrator.joblib"
    shutil.copy(model_path, latest_model)
    shutil.copy(meta_path, latest_meta)
    shutil.copy(calibrator_path, latest_calibrator)

    print()
    print(f"[xgb2] saved {model_path}")
    print(f"[xgb2] saved {calibrator_path}")
    print(f"[xgb2] saved {meta_path}")
    print(f"[xgb2] copied -> {latest_model}")
    print(f"[xgb2] copied -> {latest_calibrator}")
    print(f"[xgb2] copied -> {latest_meta}")
    print()
    print("[xgb2] comparison vs other models on each model's own test set "
          "(NOT directly comparable -- different splits):")
    comparisons = [
        ("cb  10-feat v2 (catboost3)",         MODEL_DIR / "cb_insider_10feat_v2_latest.meta.json"),
        ("cb  10-feat v2 d2 (catboost4)",      MODEL_DIR / "cb_insider_10feat_v2_d2_latest.meta.json"),
        ("cb  30-feat v2 (catboost2)",         MODEL_DIR / "cb_insider_30feat_v2_latest.meta.json"),
        ("rf  10-feat v2 (randomforest2)",     MODEL_DIR / "rf_insider_10feat_v2_latest.meta.json"),
        ("xgb 30-feat (notebook 80/20)",       MODEL_DIR / "xgb_insider_30feat_latest.meta.json"),
    ]
    for tag, path in comparisons:
        try:
            with open(path) as f:
                m = json.load(f).get("metrics", {})
            print(f"  {tag:<37}  ROC-AUC={m.get('roc_auc', 0):.4f}  "
                  f"PR-AUC={m.get('pr_auc', 0):.4f}")
        except FileNotFoundError:
            pass
    print(f"  {f'xgb {TOP_K}-feat v2 (this run)':<37}  "
          f"ROC-AUC={test_metrics['roc_auc']:.4f}  "
          f"PR-AUC={test_metrics['pr_auc']:.4f}")


if __name__ == "__main__":
    main()
