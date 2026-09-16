"""Plot CV PR-AUC vs number of model *parameters* (not features).

Feature count is a poor proxy for capacity: a 10-feature XGBoost with 600
boosted trees carries far more fitted parameters than a 20-feature random
forest, and both dwarf a logistic regression. This script loads every trained
artifact we can open, counts the total number of nodes across all trees in the
ensemble (internal split nodes + leaves -- each stores fitted values), and
plots CV PR-AUC against that count on a log x-axis.

Parameter counting per family:
  * sklearn forests (RandomForest): sum(tree_.node_count) over estimators_
  * XGBoost boosters (.json):        len(booster.trees_to_dataframe())
  * CatBoost (symmetric trees):      tree_count_ * (2**(depth+1) - 1)

Reuses compare_models.load_models() so the model set (including the 4
Iran-hidden experiment models) stays in sync with the sibling comparison plot.

Outputs:
  param_comparison_<ts>.png
  param_comparison_<ts>.log
"""

from __future__ import annotations

import argparse
import glob
import os
import sys
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np

from compare_models import load_models, apply_roc_metric, MODEL_DIR, NEW_TAGS, Tee

BLUE, ORANGE, PURPLE = "#1f77b4", "#ff7f0e", "#9467bd"


def color(r):
    if r.get("is_new"):
        return PURPLE
    return ORANGE if r["is_meta"] else BLUE


def _artifact_path(tag: str) -> str | None:
    """Locate the model artifact matching how load_models() picked its meta."""
    if tag in NEW_TAGS:
        cands = [q for q in glob.glob(str(MODEL_DIR / f"{tag}_*"))
                 if q.endswith((".joblib", ".json"))
                 and not q.endswith((".calibrator.joblib", ".meta.json"))
                 and "_latest." not in q]
        return max(cands, key=os.path.getmtime) if cands else None
    for ext in (".joblib", ".json"):
        p = MODEL_DIR / f"{tag}_latest{ext}"
        if p.exists():
            return str(p)
    return None


def _find_forest(o):
    if hasattr(o, "estimators_"):
        return o
    for attr in ("steps", "named_steps"):
        container = getattr(o, attr, None)
        if container is None:
            continue
        steps = container.values() if hasattr(container, "values") else [s for _, s in container]
        for s in steps:
            if hasattr(s, "estimators_"):
                return s
    return None


def _find_linear(o):
    """Locate a fitted linear model (coef_ + intercept_), unwrapping pipelines."""
    def _is_linear(s):
        return hasattr(s, "coef_") and hasattr(s, "intercept_")
    if _is_linear(o):
        return o
    for attr in ("steps", "named_steps"):
        container = getattr(o, attr, None)
        if container is None:
            continue
        steps = container.values() if hasattr(container, "values") else [s for _, s in container]
        for s in steps:
            if _is_linear(s):
                return s
    return None


def count_params(path: str) -> int | None:
    """Total tree nodes (split nodes + leaves) across the ensemble."""
    try:
        if path.endswith(".json"):
            import xgboost as xgb
            b = xgb.Booster()
            b.load_model(path)
            return int(len(b.trees_to_dataframe()))
        import joblib
        obj = joblib.load(path)
        forest = _find_forest(obj)
        if forest is not None:  # sklearn RandomForest / bagging
            return int(sum(t.tree_.node_count for t in forest.estimators_))
        if type(obj).__name__ == "CatBoostClassifier":  # symmetric trees
            depth = int(obj.get_all_params().get("depth", 6))
            return int(obj.tree_count_ * (2 ** (depth + 1) - 1))
        if hasattr(obj, "booster_"):  # LightGBM
            return int(obj.booster_.trees_to_dataframe().shape[0])
        linear = _find_linear(obj)
        if linear is not None:  # logistic / linear: coefficients + intercept
            return int(np.asarray(linear.coef_).size + np.asarray(linear.intercept_).size)
    except Exception as exc:
        print(f"[par] could not count {os.path.basename(path)}: {exc}")
    return None


def plot(rows: list[dict], out_path: Path, best_cv, simple_par,
         metric: str = "PR-AUC") -> None:
    fig, ax = plt.subplots(figsize=(11, 8))
    for r in rows:
        ax.scatter(r["n_params"], r["cv_pr"], s=110, color=color(r),
                   edgecolor="black", linewidth=0.5, alpha=0.85, zorder=3)
        ax.annotate(f"{r['label']}\n({r['n_params']:,} params)",
                    (r["n_params"], r["cv_pr"]), fontsize=7,
                    xytext=(5, 3), textcoords="offset points")
    ax.scatter([best_cv["n_params"]], [best_cv["cv_pr"]], s=340,
               facecolors="none", edgecolors="green", linewidth=2.2, zorder=4,
               label=f"best CV: {best_cv['label']}")
    ax.scatter([simple_par["n_params"]], [simple_par["cv_pr"]], s=280,
               facecolors="none", edgecolors="red", linewidth=2.2, zorder=4,
               label=f"leanest strong (CV): {simple_par['label']}")
    ax.set_xscale("log")
    ax.set_xlabel("number of model parameters  (total tree nodes, log scale)")
    ax.set_ylabel(f"cross-validated {metric}")
    ax.set_title(f"CV {metric} vs model size (parameters, not features)\n"
                 "top-left = strong & compact",
                 fontsize=12, fontweight="bold")
    ax.grid(alpha=0.3, which="both")

    cat = [
        Line2D([0], [0], marker="o", color="w", markerfacecolor=BLUE,
               markeredgecolor="black", markersize=9, label="behavioral only"),
        Line2D([0], [0], marker="o", color="w", markerfacecolor=ORANGE,
               markeredgecolor="black", markersize=9, label="+ metadata"),
        Line2D([0], [0], marker="o", color="w", markerfacecolor=PURPLE,
               markeredgecolor="black", markersize=9, label="Iran-hidden experiment"),
    ]
    leg1 = ax.legend(handles=cat, loc="lower right", fontsize=8, title="feature set")
    ax.add_artist(leg1)
    ax.legend(loc="upper left", fontsize=8)

    caveat = ("Parameters = total tree nodes across the ensemble (splits + leaves). "
              "XGB/CatBoost = many boosted trees; RF = fewer but deeper trees. "
              "Counts are comparable in spirit, not identical in what each node stores.")
    fig.text(0.5, -0.02, caveat, ha="center", fontsize=8, style="italic",
             color="#555555", wrap=True)
    fig.savefig(out_path, dpi=130, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--metric", choices=["pr", "roc"], default="pr",
                    help="CV metric on the y-axis: PR-AUC (default) or ROC-AUC "
                         "(reads cache/cv_roc_recomputed.json).")
    args = ap.parse_args()
    metric = "ROC-AUC" if args.metric == "roc" else "PR-AUC"
    suffix = "_roc" if args.metric == "roc" else ""

    ts = time.strftime("%Y%m%d_%H%M%S")
    log_path = Path(f"param_comparison{suffix}_{ts}.log")
    log_file = open(log_path, "w")
    saved = sys.stdout
    sys.stdout = Tee(saved, log_file)
    try:
        all_rows = load_models()
        if args.metric == "roc":
            apply_roc_metric(all_rows)
        rows = [r for r in all_rows if r["cv_pr"] is not None]
        kept: list[dict] = []
        for r in rows:
            path = _artifact_path(r["tag"])
            if path is None:
                print(f"[par] no artifact for {r['tag']}, skipping")
                continue
            n = count_params(path)
            if n is None:
                continue
            r["n_params"] = n
            kept.append(r)

        kept.sort(key=lambda r: r["n_params"])
        print(f"[par] counted parameters for {len(kept)} models  (metric = {metric})\n")
        mshort = "ROC" if args.metric == "roc" else "PR"
        print(f"{'model':<24} {'algo':>5} {'nf':>4} {'params':>10} {'CV-' + mshort:>8}")
        print("-" * 56)
        for r in kept:
            print(f"{r['label']:<24} {r['algo']:>5} {str(r['n_features']):>4} "
                  f"{r['n_params']:>10,} {r['cv_pr']:>8.4f}")

        best_cv = max(kept, key=lambda r: r["cv_pr"])
        cut = best_cv["cv_pr"] - 0.05
        strong = [r for r in kept if r["cv_pr"] >= cut]
        simple_par = min(strong, key=lambda r: (r["n_params"], -r["cv_pr"]))
        print()
        print("=" * 56)
        print(f"  BEST by CV {mshort:<6}: {best_cv['label']:<24} "
              f"CV={best_cv['cv_pr']:.4f}  params={best_cv['n_params']:,}")
        print(f"  LEANEST strong    : {simple_par['label']:<24} "
              f"CV={simple_par['cv_pr']:.4f}  params={simple_par['n_params']:,} "
              f"(within 0.05 CV of best)")

        out_png = Path(f"param_comparison{suffix}_{ts}.png")
        plot(kept, out_png, best_cv, simple_par, metric=metric)
        print()
        print(f"[par] plot saved to {out_png}")
        print(f"[par] log saved to {log_path}")
    finally:
        sys.stdout = saved
        log_file.close()


if __name__ == "__main__":
    main()
