r"""Standalone 'CV PR-AUC vs test PR-AUC' scatter for the paper.

Reproduces Panel B of ``compare_models.py`` as a single, self-contained figure
suitable for the Cross-Validation section. Every model that has both a
cross-validated PR-AUC and a held-out test PR-AUC is plotted as one point:

  * x-axis: cross-validated PR-AUC (mean over the grouped CV folds);
  * y-axis: held-out test PR-AUC (on the pinned market);
  * dashed line: CV = test. Points BELOW the line are CV-optimistic, i.e. they
    scored higher in cross-validation than they generalize on the held-out
    market -- the visual signature of overfitting.

Colours match compare_models.py:
  blue   = behavioral-only models,
  orange = + metadata models,
  purple = Iran-hidden experiment.

The point data is loaded through ``compare_models.load_models`` so this script
stays byte-for-byte consistent with the master comparison and never hard-codes
numbers. Run it whenever the underlying ``cache/models/*_latest.meta.json``
artifacts change.

Outputs (next to this file):
  cv_vs_test_pr_auc.pdf   (vector, for LaTeX \includegraphics)
  cv_vs_test_pr_auc.png   (raster preview)

Usage:
  python plot_cv_vs_test.py
"""

from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

from compare_models import load_models

ROOT = Path(__file__).resolve().parent
OUT_STEM = ROOT / "cv_vs_test_pr_auc"

BLUE, ORANGE, PURPLE = "#1f77b4", "#ff7f0e", "#9467bd"


def _color(r: dict) -> str:
    if r.get("is_new"):
        return PURPLE
    return ORANGE if r["is_meta"] else BLUE


def main() -> None:
    rows = load_models()
    pts = [r for r in rows if r["cv_pr"] is not None and r["test_pr"] is not None]
    if not pts:
        raise SystemExit("no models with both CV and test PR-AUC found")

    fig, ax = plt.subplots(figsize=(7.2, 6.0))

    for r in pts:
        ax.scatter(r["cv_pr"], r["test_pr"], s=90, color=_color(r),
                   edgecolor="black", linewidth=0.5, alpha=0.85, zorder=3)
        ax.annotate(r["label"], (r["cv_pr"], r["test_pr"]),
                    fontsize=7, xytext=(4, 3), textcoords="offset points")

    lim = [0.3, 1.02]
    ax.plot(lim, lim, "k--", alpha=0.35, label="CV = test")
    ax.set_xlim(*lim)
    ax.set_ylim(*lim)
    ax.set_xlabel("CV PR-AUC")
    ax.set_ylabel("test PR-AUC")
    ax.set_title("CV vs test PR-AUC\n(points below dashed line = CV-optimistic)",
                 fontsize=12, fontweight="bold")
    ax.grid(alpha=0.3)

    handles = [
        Line2D([0], [0], marker="o", color="w", markerfacecolor=BLUE,
               markeredgecolor="black", markersize=9, label="behavioral only"),
        Line2D([0], [0], marker="o", color="w", markerfacecolor=ORANGE,
               markeredgecolor="black", markersize=9, label="+ metadata"),
        Line2D([0], [0], marker="o", color="w", markerfacecolor=PURPLE,
               markeredgecolor="black", markersize=9, label="Iran-hidden experiment"),
        Line2D([0], [0], linestyle="--", color="k", alpha=0.35, label="CV = test"),
    ]
    ax.legend(handles=handles, loc="lower right", fontsize=8)

    fig.tight_layout()
    fig.savefig(f"{OUT_STEM}.pdf", bbox_inches="tight")
    fig.savefig(f"{OUT_STEM}.png", dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"[plot] {len(pts)} models plotted")
    print(f"[plot] wrote {OUT_STEM}.pdf")
    print(f"[plot] wrote {OUT_STEM}.png")


if __name__ == "__main__":
    main()
