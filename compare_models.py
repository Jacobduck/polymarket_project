"""Compare every insider model we've trained, ranked by cross-validated PR-AUC.

Auto-discovers all ``cache/models/*_latest.meta.json`` artifacts, pulls each
model's CV PR-AUC (mean ± std where available), held-out test PR-AUC/ROC-AUC,
and feature count, then:

  * prints a ranked table to stdout + a log file;
  * names the best model by CV, the best by test, and a recommended "simple"
    model (fewest features that still generalizes well) for deployment on new
    markets;
  * writes a 3-panel plot:
      (A) models ranked by CV PR-AUC (error bars = CV std);
      (B) CV PR-AUC vs test PR-AUC  -- exposes CV-vs-test inversion / overfit;
      (C) test PR-AUC vs n_features -- the "simple model" Pareto view.

CAVEAT printed on the figure: CV schemes and test pins differ across model
families (StratifiedGroupKFold for xgb, TimeSeriesSplit for cb/rf, GroupKFold
for rf5; behavioral models train on training_data.parquet, metadata models on
training_data_with_metadata.parquet). The CV numbers are directional, not a
perfectly controlled ranking.

Outputs:
  model_comparison_<ts>.png
  model_comparison_<ts>.log
"""

from __future__ import annotations

import json
import glob
import os
import sys
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

MODEL_DIR = Path("cache/models")
LOG_DIR = Path(".")

# Exact-duplicate / redundant artifacts to skip (identical metrics to a kept one).
SKIP_TAGS = {
    "xgb_insider_20feat_v2_nocal",  # identical numbers to xgb_insider_20feat_v2
    "cb_insider_30feat",            # superseded single-split by cb_insider_30feat_v2
    "rf_insider_30feat",            # superseded single-split by rf_insider_10feat_v2
    "xgb_insider_30feat",           # superseded single-split by xgb_insider_30feat_v2
}

# Pretty labels (fallback = tag). Keep them short for the plot.
LABELS = {
    "xgb_insider_meta_v7": "xgb 20b+10m v7",
    "xgb_insider_meta_v5": "xgb 10b+17m v5",
    "xgb_insider_meta10_v6": "xgb meta-only v6",
    "rf_insider_meta_v5": "rf 10b+10m v5 (tuned)",
    "rf_insider_meta_v4": "rf 10b+17m v4 (axiom)",
    "rf_insider_meta_v3": "rf 10b+17m v3",
    "xgb_insider_20feat_v2": "xgb 20feat v2",
    "xgb_insider_20feat_v2_mstr": "xgb 20feat v2 (mstr)",
    "xgb_insider_20feat_v2_stable": "xgb 20feat v2 (stable)",
    "xgb_insider_20feat_v2_axiom": "xgb 20feat v2 (axiom)",
    "xgb_insider_10feat_v2": "xgb 10feat v2",
    "xgb_insider_30feat_v2": "xgb 30feat v2",
    "xgb_insider_14feat": "xgb 14feat",
    "xgb_insider_50feat": "xgb 50feat",
    "xgb_insider": "xgb all-319feat",
    "cb_insider_30feat_v2": "cb 30feat v2",
    "cb_insider_10feat_v2": "cb 10feat v2",
    "cb_insider_10feat_v2_d2": "cb 10feat v2 (d2)",
    "rf_insider_10feat_v2": "rf 10feat v2",
    "lgbm_insider_30feat": "lgbm 30feat",
    "lr_insider_30feat": "logreg 30feat",
    "iso_insider_30feat": "isoforest 30feat",
    "stack_insider_30feat": "stack 30feat",
}


def is_metadata(tag: str) -> bool:
    return "meta" in tag


def algo(tag: str) -> str:
    for k in ("xgb", "cb", "rf", "lgbm", "lr", "iso", "stack"):
        if tag.startswith(k):
            return k
    return "other"


def load_models() -> list[dict]:
    rows: list[dict] = []
    for p in sorted(glob.glob(str(MODEL_DIR / "*_latest.meta.json"))):
        tag = os.path.basename(p).replace("_latest.meta.json", "")
        if tag in SKIP_TAGS:
            continue
        try:
            m = json.load(open(p))
        except Exception as exc:
            print(f"[cmp] skip {tag}: {exc}")
            continue
        cv = m.get("cv") or {}
        tun = m.get("tuning") or {}
        met = m.get("metrics") or {}
        cv_mean = cv.get("pr_auc_mean")
        if cv_mean is None:
            cv_mean = tun.get("best_cv_pr_auc")
        rows.append({
            "tag": tag,
            "label": LABELS.get(tag, tag),
            "algo": algo(tag),
            "is_meta": is_metadata(tag),
            "n_features": m.get("n_features"),
            "cv_pr": cv_mean,
            "cv_std": cv.get("pr_auc_std"),
            "test_pr": met.get("pr_auc"),
            "test_roc": met.get("roc_auc"),
        })
    return rows


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


def print_table(rows: list[dict]) -> None:
    ranked = sorted(rows, key=lambda r: (r["cv_pr"] is None, -(r["cv_pr"] or -1)))
    print(f"{'model':<24} {'algo':>5} {'meta':>5} {'nf':>4} "
          f"{'CV-PR':>8} {'±std':>7} {'test-PR':>8} {'test-ROC':>8}")
    print("-" * 78)
    for r in ranked:
        cvp = f"{r['cv_pr']:.4f}" if r["cv_pr"] is not None else "   -"
        cvs = f"{r['cv_std']:.4f}" if r["cv_std"] is not None else "   -"
        tp = f"{r['test_pr']:.4f}" if r["test_pr"] is not None else "   -"
        tr = f"{r['test_roc']:.4f}" if r["test_roc"] is not None else "   -"
        print(f"{r['label']:<24} {r['algo']:>5} {str(r['is_meta']):>5} "
              f"{str(r['n_features']):>4} {cvp:>8} {cvs:>7} {tp:>8} {tr:>8}")


def recommend(rows: list[dict]) -> tuple[dict, dict, dict]:
    with_cv = [r for r in rows if r["cv_pr"] is not None]
    with_test = [r for r in rows if r["test_pr"] is not None]
    best_cv = max(with_cv, key=lambda r: r["cv_pr"])
    best_test = max(with_test, key=lambda r: r["test_pr"])
    # Simple = fewest features among models that generalize well (test PR>=0.95),
    # tie-broken by higher test PR-AUC.
    strong = [r for r in with_test if r["test_pr"] is not None and r["test_pr"] >= 0.95
              and r["n_features"] is not None]
    if strong:
        simple = min(strong, key=lambda r: (r["n_features"], -r["test_pr"]))
    else:
        simple = best_test
    return best_cv, best_test, simple


def plot(rows: list[dict], out_path: Path, best_cv, best_test, simple) -> None:
    fig = plt.figure(figsize=(15, 11))
    gs = fig.add_gridspec(2, 2, height_ratios=[1.25, 1.0], hspace=0.32, wspace=0.22)
    ax_bar = fig.add_subplot(gs[0, :])
    ax_cvt = fig.add_subplot(gs[1, 0])
    ax_simp = fig.add_subplot(gs[1, 1])

    BLUE, ORANGE = "#1f77b4", "#ff7f0e"

    def color(r):
        return ORANGE if r["is_meta"] else BLUE

    # ---- Panel A: ranked CV PR-AUC bar chart --------------------------- #
    cv_rows = sorted([r for r in rows if r["cv_pr"] is not None],
                     key=lambda r: r["cv_pr"])
    y = np.arange(len(cv_rows))
    vals = [r["cv_pr"] for r in cv_rows]
    errs = [r["cv_std"] if r["cv_std"] is not None else 0 for r in cv_rows]
    bars = ax_bar.barh(y, vals, xerr=errs, color=[color(r) for r in cv_rows],
                       alpha=0.85, edgecolor="black", linewidth=0.4,
                       error_kw=dict(ecolor="gray", lw=1, capsize=3))
    ax_bar.set_yticks(y)
    ax_bar.set_yticklabels([f"{r['label']}  (nf={r['n_features']})" for r in cv_rows],
                           fontsize=9)
    ax_bar.set_xlabel("cross-validated PR-AUC (mean ± std)")
    ax_bar.set_title("A. Models ranked by CV PR-AUC  "
                     "(blue = behavioral only, orange = + metadata)",
                     fontsize=11, fontweight="bold")
    ax_bar.grid(axis="x", alpha=0.3)
    ax_bar.set_xlim(0, 1.0)
    for yi, r in zip(y, cv_rows):
        tp = f"{r['test_pr']:.3f}" if r["test_pr"] is not None else "-"
        ax_bar.text(r["cv_pr"] + (r["cv_std"] or 0) + 0.012, yi,
                    f"{r['cv_pr']:.3f}  (test {tp})", va="center", fontsize=8)
    # mark best-by-CV
    for yi, r in zip(y, cv_rows):
        if r["tag"] == best_cv["tag"]:
            bars[yi].set_edgecolor("red")
            bars[yi].set_linewidth(2.0)

    # ---- Panel B: CV PR-AUC vs test PR-AUC ----------------------------- #
    pts = [r for r in rows if r["cv_pr"] is not None and r["test_pr"] is not None]
    for r in pts:
        ax_cvt.scatter(r["cv_pr"], r["test_pr"], s=90, color=color(r),
                       edgecolor="black", linewidth=0.5, alpha=0.85, zorder=3)
        ax_cvt.annotate(r["label"], (r["cv_pr"], r["test_pr"]),
                        fontsize=7, xytext=(4, 3), textcoords="offset points")
    lim = [0.3, 1.02]
    ax_cvt.plot(lim, lim, "k--", alpha=0.35, label="CV = test")
    ax_cvt.set_xlim(*lim)
    ax_cvt.set_ylim(*lim)
    ax_cvt.set_xlabel("CV PR-AUC")
    ax_cvt.set_ylabel("test PR-AUC")
    ax_cvt.set_title("B. CV vs test PR-AUC\n(points below dashed line = CV-optimistic)",
                     fontsize=11, fontweight="bold")
    ax_cvt.grid(alpha=0.3)
    ax_cvt.legend(loc="lower right", fontsize=8)

    # ---- Panel C: test PR-AUC vs n_features (simple-model view) -------- #
    fp = [r for r in rows if r["test_pr"] is not None and r["n_features"] is not None]
    for r in fp:
        ax_simp.scatter(r["n_features"], r["test_pr"], s=90, color=color(r),
                        edgecolor="black", linewidth=0.5, alpha=0.85, zorder=3)
        ax_simp.annotate(r["label"], (r["n_features"], r["test_pr"]),
                         fontsize=7, xytext=(4, 3), textcoords="offset points")
    ax_simp.scatter([simple["n_features"]], [simple["test_pr"]], s=260,
                    facecolors="none", edgecolors="red", linewidth=2.2, zorder=4,
                    label=f"recommended simple: {simple['label']}")
    ax_simp.scatter([best_test["n_features"]], [best_test["test_pr"]], s=320,
                    facecolors="none", edgecolors="green", linewidth=2.2, zorder=4,
                    label=f"best test: {best_test['label']}")
    ax_simp.set_xscale("log")
    ax_simp.set_xlabel("n_features (log scale)")
    ax_simp.set_ylabel("test PR-AUC")
    ax_simp.set_title("C. Test PR-AUC vs model complexity\n(top-left = simple & strong)",
                      fontsize=11, fontweight="bold")
    ax_simp.grid(alpha=0.3, which="both")
    ax_simp.legend(loc="lower right", fontsize=8)

    caveat = ("CAVEAT: CV schemes & test pins differ across families "
              "(xgb=StratifiedGroupKFold, cb/rf=TimeSeriesSplit, rf5=GroupKFold; "
              "behavioral vs +metadata datasets). CV is directional, not a "
              "perfectly controlled ranking.")
    fig.text(0.5, 0.005, caveat, ha="center", fontsize=8, style="italic",
             color="#555555", wrap=True)
    fig.suptitle("Insider-detection models: CV vs test comparison",
                 fontsize=14, fontweight="bold")
    fig.savefig(out_path, dpi=130, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    ts = time.strftime("%Y%m%d_%H%M%S")
    log_path = LOG_DIR / f"model_comparison_{ts}.log"
    log_file = open(log_path, "w")
    saved = sys.stdout
    sys.stdout = Tee(saved, log_file)
    try:
        rows = load_models()
        print(f"[cmp] discovered {len(rows)} models in {MODEL_DIR}")
        print()
        print_table(rows)

        best_cv, best_test, simple = recommend(rows)
        print()
        print("=" * 78)
        print("RECOMMENDATIONS")
        print("=" * 78)
        print(f"  BEST by CV PR-AUC : {best_cv['label']:<24} "
              f"CV={best_cv['cv_pr']:.4f}±{(best_cv['cv_std'] or 0):.4f}  "
              f"test-PR={best_cv['test_pr']}  nf={best_cv['n_features']}")
        print(f"  BEST by test PR   : {best_test['label']:<24} "
              f"test-PR={best_test['test_pr']:.4f}  "
              f"test-ROC={best_test['test_roc']:.4f}  "
              f"CV={best_test['cv_pr']}  nf={best_test['n_features']}")
        print(f"  SIMPLE (deploy)   : {simple['label']:<24} "
              f"test-PR={simple['test_pr']:.4f}  "
              f"test-ROC={simple['test_roc']:.4f}  nf={simple['n_features']}")
        print()
        print("  NOTE: the metadata models top the CV ranking but the behavioral")
        print("  models top the TEST ranking -- the CV ordering inverts on the")
        print("  held-out market, a classic CV-optimism / overfit signal. For")
        print("  deploying on NEW markets, trust the test-PR leaders.")

        out_png = LOG_DIR / f"model_comparison_{ts}.png"
        plot(rows, out_png, best_cv, best_test, simple)
        print()
        print(f"[cmp] plot saved to {out_png}")
        print(f"[cmp] log saved to {log_path}")
    finally:
        sys.stdout = saved
        log_file.close()


if __name__ == "__main__":
    main()
