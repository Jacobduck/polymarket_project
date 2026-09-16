"""XGBoost (top-20 behavioral + top-10 metadata), all-big-Iran removed.

Experiment variant of ``xgboost7.py``. Tests whether Iran markets corrupt the
model by removing them entirely. Both large Iran markets are dropped:
    us-strikes-iran-by-march-1-2026-492                    (245 rows / 35 ins)
    us-strikes-iran-by-february-28-2026-227-...            (189 rows / 27 ins)
The 3 tiny Iran markets (63 rows total) are LEFT IN, per the agreed scope
("only the two big Iran markets"). See ``25ir_xgboost.py`` for the
down-sample-only variant.

A single seeded stratified-by-group split (GroupShuffleSplit ~15%) is used
because Iran can no longer be the pinned test market. CROSS-VALIDATION PR-AUC
(StratifiedGroupKFold over the training set) is the primary metric; held-out
test PR-AUC is the secondary, higher-variance number. Same Optuna tuning + OOF
Platt calibration as xgboost7.

Outputs:
  cache/models/xgb_insider_0ir_<ts>.json            (XGBoost native fmt)
  cache/models/xgb_insider_0ir_<ts>.calibrator.joblib
  cache/models/xgb_insider_0ir_<ts>.meta.json
  0ir_xgboost_<ts>_curves.png
  0ir_xgboost_<ts>.log
"""

from __future__ import annotations

import json
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
from sklearn.model_selection import GroupShuffleSplit, StratifiedGroupKFold
from xgboost import XGBClassifier

from polycluster.metadata import METADATA_FEATURE_COLUMNS

CACHE = Path("cache")
TRAIN_PARQUET = CACHE / "training_data_with_metadata.parquet"
MODEL_DIR = CACHE / "models"
MODEL_DIR.mkdir(parents=True, exist_ok=True)

FEATURE_SOURCE_META = MODEL_DIR / "cb_insider_30feat_latest.meta.json"
TOP_K = 20
MODEL_TAG = __import__("os").environ.get("MODEL_TAG_PREFIX", "") + "xgb_insider_0ir"
LOG_PREFIX = "0ir_xgboost"
TAG = "xgb0ir"

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
THRESHOLD = 0.15
RANDOM_STATE = 42
TEST_SIZE = 0.15
LOG_DIR = Path(".")

# --- Iran filtering config ------------------------------------------------- #
MAR1_MARKET = "us-strikes-iran-by-march-1-2026-492"
FEB28_PREFIX = "us-strikes-iran-by-february-28-2026-227"

N_CV_FOLDS = 5
N_OPTUNA_TRIALS = 100
N_OPTUNA_STARTUP_TRIALS = 20


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
# Metrics
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
    print(f"[{TAG}] {label:5s}  n={out['n']:4d} pos={out['n_positives']:3d}  "
          f"ROC-AUC={out['roc_auc']:.4f}  PR-AUC={out['pr_auc']:.4f}  "
          f"P={cls['precision']:.4f}  R={cls['recall']:.4f}  "
          f"F1={cls['f1']:.4f}  (thr={thr})")
    return out


def build_xgb(params: dict) -> XGBClassifier:
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
    params: dict, X: pd.DataFrame, y: pd.Series, groups: pd.Series,
    trial: optuna.trial.Trial | None = None,
) -> tuple[float, list[float]]:
    splitter = StratifiedGroupKFold(
        n_splits=N_CV_FOLDS, shuffle=True, random_state=RANDOM_STATE,
    )
    fold_pr_aucs: list[float] = []
    for fold_idx, (tr_idx, va_idx) in enumerate(splitter.split(X, y, groups)):
        model = build_xgb(params)
        model.fit(X.iloc[tr_idx], y.iloc[tr_idx], verbose=False)
        proba_va = model.predict_proba(X.iloc[va_idx])[:, 1]
        if len(np.unique(y.iloc[va_idx])) >= 2:
            pr = float(average_precision_score(y.iloc[va_idx], proba_va))
        else:
            pr = float("nan")
        fold_pr_aucs.append(pr)
        if trial is not None:
            trial.report(float(np.nanmean(fold_pr_aucs)), fold_idx)
            if trial.should_prune():
                raise optuna.TrialPruned()
    return float(np.nanmean(fold_pr_aucs)), fold_pr_aucs


def make_objective(X: pd.DataFrame, y: pd.Series, groups: pd.Series):
    def objective(trial: optuna.trial.Trial) -> float:
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

    X_train = df_train[feature_cols].astype(float).reset_index(drop=True)
    y_train = df_train[LABEL_COL].astype(int).reset_index(drop=True)
    groups_train = df_train[GROUP_COL].reset_index(drop=True)
    X_test = df_test[feature_cols].astype(float).reset_index(drop=True)
    y_test = df_test[LABEL_COL].astype(int).reset_index(drop=True)

    print(f"[{TAG}] split (GroupShuffleSplit test_size={TEST_SIZE}, seed={RANDOM_STATE}):")
    print(f"[{TAG}]   train: {len(df_train)} rows, {groups_train.nunique()} markets, "
          f"pos={int((y_train==1).sum())} neg={int((y_train==0).sum())}")
    print(f"[{TAG}]   test : {len(df_test)} rows, {df_test[GROUP_COL].nunique()} markets, "
          f"pos={int((y_test==1).sum())}")
    print(f"[{TAG}]   test markets: {sorted(df_test[GROUP_COL].unique())}")

    # --- Optuna tuning --------------------------------------------------- #
    print(f"[{TAG}] Optuna tuning: {N_OPTUNA_TRIALS} trials, TPE "
          f"({N_OPTUNA_STARTUP_TRIALS} startup), MedianPruner, "
          f"{N_CV_FOLDS}-fold StratifiedGroupKFold CV")
    sampler = TPESampler(seed=RANDOM_STATE, n_startup_trials=N_OPTUNA_STARTUP_TRIALS)
    pruner = MedianPruner(n_startup_trials=N_OPTUNA_STARTUP_TRIALS, n_warmup_steps=2)
    study = optuna.create_study(
        direction="maximize", sampler=sampler, pruner=pruner,
        study_name=f"{MODEL_TAG}_{ts}",
    )
    t0 = time.time()

    def _log_cb(study_: optuna.Study, trial: optuna.trial.FrozenTrial) -> None:
        best_so_far = study_.best_value if study_.best_trial is not None else float("nan")
        score_str = f"{trial.value:.4f}" if trial.value is not None else "  pruned"
        print(f"[{TAG}]   trial {trial.number:>3}  state={trial.state.name:<8}  "
              f"score={score_str}  best={best_so_far:.4f}")

    study.optimize(
        make_objective(X_train, y_train, groups_train),
        n_trials=N_OPTUNA_TRIALS, callbacks=[_log_cb], show_progress_bar=False,
    )
    tuning_secs = time.time() - t0
    best_params = study.best_params
    best_cv_pr_auc = float(study.best_value)
    n_completed = sum(1 for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE)
    n_pruned = sum(1 for t in study.trials if t.state == optuna.trial.TrialState.PRUNED)
    print(f"[{TAG}] tuning done in {tuning_secs:.1f}s  "
          f"({n_completed} completed, {n_pruned} pruned)")
    print(f"[{TAG}] best CV PR-AUC = {best_cv_pr_auc:.4f}")
    print(f"[{TAG}] best params:")
    for k, v in best_params.items():
        print(f"  {k:20s} = {v:.5f}" if isinstance(v, float) else f"  {k:20s} = {v}")

    # --- refit CV for OOF preds + per-fold scores (primary metric) ------- #
    print()
    print(f"[{TAG}] re-running CV with best params -> OOF preds + per-fold PR-AUC")
    splitter = StratifiedGroupKFold(
        n_splits=N_CV_FOLDS, shuffle=True, random_state=RANDOM_STATE,
    )
    oof_preds = np.full(len(X_train), np.nan, dtype=float)
    fold_summaries: list[dict] = []
    best_fold_pr_aucs: list[float] = []
    for fold_idx, (tr_idx2, va_idx2) in enumerate(
        splitter.split(X_train, y_train, groups_train), start=1
    ):
        fold_model = build_xgb(best_params)
        fold_model.fit(X_train.iloc[tr_idx2], y_train.iloc[tr_idx2], verbose=False)
        pr_fold = fold_model.predict_proba(X_train.iloc[va_idx2])[:, 1]
        oof_preds[va_idx2] = pr_fold
        if len(np.unique(y_train.iloc[va_idx2])) >= 2:
            fold_pr = float(average_precision_score(y_train.iloc[va_idx2], pr_fold))
        else:
            fold_pr = float("nan")
        best_fold_pr_aucs.append(fold_pr)
        n_va_pos = int((y_train.iloc[va_idx2] == 1).sum())
        print(f"  fold {fold_idx}: val_rows={len(va_idx2)} val_pos={n_va_pos}  "
              f"PR-AUC={fold_pr:.4f}")
        fold_summaries.append({
            "fold": fold_idx, "n_val_rows": int(len(va_idx2)),
            "n_val_positives": n_va_pos, "pr_auc": fold_pr,
        })

    if np.isnan(oof_preds).any():
        raise SystemExit("OOF preds incomplete -- some train rows not covered by CV")

    cv_pr_auc_mean = float(np.nanmean(best_fold_pr_aucs))
    cv_pr_auc_std = float(np.nanstd(best_fold_pr_aucs))
    print(f"[{TAG}] CV PR-AUC (primary) = {cv_pr_auc_mean:.4f} ± {cv_pr_auc_std:.4f}")

    # --- OOF Platt calibrator ------------------------------------------- #
    calibrator = LogisticRegression(C=1.0, solver="lbfgs")
    calibrator.fit(oof_preds.reshape(-1, 1), y_train.values)
    cal_coef = float(calibrator.coef_[0][0])
    cal_intercept = float(calibrator.intercept_[0])
    print(f"[{TAG}] OOF Platt calibrator: "
          f"sigmoid({cal_coef:.3f} * raw_prob {cal_intercept:+.3f})")

    # --- final fit + evaluation ----------------------------------------- #
    print(f"[{TAG}] fitting final model on full train ({len(X_train)} rows)")
    final_model = build_xgb(best_params)
    final_model.fit(X_train, y_train, verbose=False)

    proba_train_raw = final_model.predict_proba(X_train)[:, 1]
    proba_test_raw = final_model.predict_proba(X_test)[:, 1]
    proba_train = calibrator.predict_proba(proba_train_raw.reshape(-1, 1))[:, 1]
    proba_test = calibrator.predict_proba(proba_test_raw.reshape(-1, 1))[:, 1]

    print()
    print(f"[{TAG}] metrics per split (overfit check: train >> test => overfit):")
    train_metrics = eval_split(proba_train, y_train.values, "train", THRESHOLD)
    test_metrics = eval_split(proba_test, y_test.values, "test", THRESHOLD)
    train_test_gap = train_metrics["pr_auc"] - test_metrics["pr_auc"]
    cv_test_gap = cv_pr_auc_mean - test_metrics["pr_auc"]
    print(f"[{TAG}] PR-AUC gap train-test = {train_test_gap:+.4f}  (big positive = overfit)")
    print(f"[{TAG}] PR-AUC gap CV-test    = {cv_test_gap:+.4f}")

    train_cls = classification_metrics(y_train.values, proba_train, THRESHOLD)
    test_cls = classification_metrics(y_test.values, proba_test, THRESHOLD)
    print(f"[{TAG}] classification @ thr={THRESHOLD}:")
    for label, m in [("train", train_cls), ("test", test_cls)]:
        print(f"  {label:5s}  P={m['precision']:.4f}  R={m['recall']:.4f}  "
              f"F1={m['f1']:.4f}  (TP={m['tp']}, FP={m['fp']}, FN={m['fn']}, TN={m['tn']})")

    curves_path = LOG_DIR / f"{LOG_PREFIX}_{ts}_curves.png"
    plot_curves([("train", y_train.values, proba_train),
                 ("test", y_test.values, proba_test)], curves_path, THRESHOLD)
    print(f"[{TAG}] curves saved to {curves_path}")

    print()
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

    print()
    print(f"[{TAG}] TEST confusion matrix @ thr={THRESHOLD}")
    pred_test = (proba_test >= THRESHOLD).astype(int)
    print(confusion_matrix(y_test, pred_test))
    print(classification_report(y_test, pred_test, digits=4, zero_division=0))

    importances = sorted(zip(feature_cols, final_model.feature_importances_),
                         key=lambda kv: kv[1], reverse=True)
    print(f"[{TAG}] feature importances (gain):")
    for name, imp in importances:
        print(f"  {imp:7.4f}  {name}")

    # --- artifacts ------------------------------------------------------- #
    model_path = MODEL_DIR / f"{MODEL_TAG}_{ts}.json"
    meta_path = MODEL_DIR / f"{MODEL_TAG}_{ts}.meta.json"
    calibrator_path = MODEL_DIR / f"{MODEL_TAG}_{ts}.calibrator.joblib"
    final_model.save_model(str(model_path))
    joblib.dump(calibrator, calibrator_path)

    meta = {
        "trained_at": ts,
        "model_type": "xgboost.XGBClassifier",
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
        "n_train_markets": int(groups_train.nunique()),
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
        "calibrator_file": calibrator_path.name,
        "calibration": {
            "method": "Platt scaling (sklearn LogisticRegression on raw prob)",
            "fit_on": "OOF predictions from CV (no leakage)",
            "coef": cal_coef, "intercept": cal_intercept,
        },
        "model_params": {
            "objective": "binary:logistic", "eval_metric": "aucpr",
            "tree_method": "hist", "random_state": RANDOM_STATE, **best_params,
        },
        "tuning": {
            "framework": "optuna", "sampler": "TPESampler", "pruner": "MedianPruner",
            "n_trials_requested": N_OPTUNA_TRIALS,
            "n_trials_completed": n_completed, "n_trials_pruned": n_pruned,
            "n_startup_trials": N_OPTUNA_STARTUP_TRIALS,
            "objective_metric": "mean PR-AUC across CV folds",
            "tuning_seconds": tuning_secs, "best_cv_pr_auc": best_cv_pr_auc,
        },
        "cv": {
            "scheme": (
                f"StratifiedGroupKFold(n_splits={N_CV_FOLDS}, shuffle=True, "
                f"random_state={RANDOM_STATE}) on market_slug"
            ),
            "n_folds": N_CV_FOLDS, "folds": fold_summaries,
            "pr_auc_mean": cv_pr_auc_mean, "pr_auc_std": cv_pr_auc_std,
        },
        "metrics": {
            "cv_pr_auc": cv_pr_auc_mean,
            "train": {**train_metrics, "classification_at_threshold": train_cls},
            "test": {**test_metrics, "classification_at_threshold": test_cls},
            "train_test_pr_auc_gap": float(train_test_gap),
            "cv_test_pr_auc_gap": float(cv_test_gap),
            "roc_auc": test_metrics["roc_auc"], "pr_auc": test_metrics["pr_auc"],
        },
        "log_file": str(log_path),
        "curves_plot": str(curves_path),
        "source_training_data": str(TRAIN_PARQUET),
    }
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)
    print(f"[{TAG}] saved {model_path}")
    print(f"[{TAG}] saved {calibrator_path}")
    print(f"[{TAG}] saved {meta_path}")
    print()
    print(f"[{TAG}] === RESULT ===  CV PR-AUC={cv_pr_auc_mean:.4f}  "
          f"TEST PR-AUC={test_metrics['pr_auc']:.4f}  "
          f"TEST ROC-AUC={test_metrics['roc_auc']:.4f}")


if __name__ == "__main__":
    main()
