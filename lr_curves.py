"""ROC + PR curves (train vs test) for the logistic-regression insider model.

Mirrors the curve plot produced by 25ir_xgboost.py / 25ir_randomforest.py, but
for lr_insider_30feat. To keep the evaluation honest and comparable to the tree
models, the split is the same market-pinned design used everywhere else: the
``us-strikes-iran-by-march-1-2026-492`` market is held out as test, every other
market is train (no wallet from a test market leaks into train). The LR pipeline
config (median impute -> standardize -> LogisticRegression at the model's
selected C, class_weight replicating scale_pos_weight) is taken from the stored
meta so this reproduces the deployed model exactly.

Outputs:
  lr_curves_<ts>_curves.png
  lr_curves_<ts>.log
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    classification_report,
    confusion_matrix,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

CACHE = Path("cache")
MODEL_DIR = CACHE / "models"
TRAIN_PARQUET = CACHE / "training_data.parquet"
LR_META = MODEL_DIR / "lr_insider_30feat_latest.meta.json"

LABEL_COL = "is_insider"
GROUP_COL = "market_slug"
RANDOM_STATE = 42
PIN_TEST_MARKETS = ("us-strikes-iran-by-march-1-2026-492",)
LOG_PREFIX = "lr_curves"


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
    ax_roc.set_title("ROC curves (logreg 30feat: iran-pinned test)")
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


def report_split(name: str, y: np.ndarray, proba: np.ndarray, threshold: float) -> None:
    pred = (proba >= threshold).astype(int)
    print(f"\n[{name}] ROC-AUC = {roc_auc_score(y, proba):.4f}   "
          f"PR-AUC = {average_precision_score(y, proba):.4f}")
    print(f"[{name}] confusion matrix @ thr={threshold}")
    print(confusion_matrix(y, pred))
    print(classification_report(y, pred, digits=4))


def main() -> None:
    ts = time.strftime("%Y%m%d_%H%M%S")
    log_path = Path(f"{LOG_PREFIX}_{ts}.log")
    log_file = open(log_path, "w")
    saved = sys.stdout
    sys.stdout = Tee(saved, log_file)
    try:
        meta = json.load(open(LR_META))
        feature_cols = meta["features"]
        C = float(meta["selected_C"])
        threshold = float(meta.get("threshold", 0.5))
        class_weight = {0: 1.0, 1: float(meta["scale_pos_weight"])}

        df = pd.read_parquet(TRAIN_PARQUET)
        test_mask = df[GROUP_COL].isin(PIN_TEST_MARKETS)
        df_train = df[~test_mask].reset_index(drop=True)
        df_test = df[test_mask].reset_index(drop=True)

        X_train = df_train[feature_cols].astype(float)
        y_train = df_train[LABEL_COL].astype(int)
        X_test = df_test[feature_cols].astype(float)
        y_test = df_test[LABEL_COL].astype(int)

        print(f"[lr-curves] model=lr_insider_30feat  C={C}  thr={threshold}")
        print(f"[lr-curves] pinned test market(s) = {list(PIN_TEST_MARKETS)}")
        print(f"[lr-curves] train: {df_train[GROUP_COL].nunique()} markets, "
              f"{len(X_train)} rows, {int(y_train.sum())} insiders")
        print(f"[lr-curves] test : {df_test[GROUP_COL].nunique()} market(s), "
              f"{len(X_test)} rows, {int(y_test.sum())} insiders")

        model = Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(
                C=C, solver="lbfgs", max_iter=2000,
                class_weight=class_weight, random_state=RANDOM_STATE)),
        ])
        model.fit(X_train, y_train)

        proba_train = model.predict_proba(X_train)[:, 1]
        proba_test = model.predict_proba(X_test)[:, 1]

        report_split("train", y_train.values, proba_train, threshold)
        report_split("test", y_test.values, proba_test, threshold)

        curves_path = Path(f"{LOG_PREFIX}_{ts}_curves.png")
        plot_curves([("train", y_train.values, proba_train),
                     ("test", y_test.values, proba_test)], curves_path, threshold)
        print(f"\n[lr-curves] curves saved to {curves_path}")
        print(f"[lr-curves] log saved to {log_path}")
    finally:
        sys.stdout = saved
        log_file.close()


if __name__ == "__main__":
    main()
