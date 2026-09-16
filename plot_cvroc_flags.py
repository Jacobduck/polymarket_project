#!/usr/bin/env python3
"""Two-panel model overview for the insider write-up.

Panel A: every insider model (originals + the 10 retrained_ models) ranked by
cross-validated ROC-AUC (mean +/- std), reusing compare_models' discovery and
the recomputed grouped-CV ROC numbers. Retrained models are hatched so they can
be told apart from their pre-fix originals at a glance.

Panel B: for each of the 7 found insiders, how many of the 35 scored models
flag that (wallet, market) pair at the per-model threshold tuned for
recall >= 0.95 (from audit_pair_precision.py, aggregated over the recall95
_models.csv files).
"""
from __future__ import annotations

from pathlib import Path

import json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
import numpy as np
import pandas as pd

from compare_models import load_models, LABELS

CV_ROC = Path("cache/cv_roc_with_retrained.json")
OUT = Path("audit_out/model_cvroc_and_insider_flags.png")

BLUE, ORANGE, PURPLE = "#1f77b4", "#ff7f0e", "#9467bd"

# The 7 found insiders -> their recall>=0.95 tuned-flag CSV.
INSIDERS = [
    ("d4vd  (google #1)",      "audit_out/recall95/insider1_d4vd_models.csv"),
    ("axiom #4  (73f09e8d)",   "audit_out/recall95/auditprec_willaxiombeaccusedof_73f09e8d_20260715_110032_models.csv"),
    ("axiom #6  (f8b6702e)",   "audit_out/recall95/auditprec_willaxiombeaccusedof_f8b6702e_20260715_110049_models.csv"),
    ("axiom #7  (8f7fe67a)",   "audit_out/recall95/auditprec_willaxiombeaccusedof_8f7fe67a_20260715_110105_models.csv"),
    ("iran #11  (2370bd31)",   "audit_out/recall95/auditprec_willusorisraelstrike_2370bd31_20260715_110122_models.csv"),
    ("iran #18  (a1308d9a)",   "audit_out/recall95/auditprec_willusorisraelstrike_a1308d9a_20260715_110142_models.csv"),
    ("cardi #6  (87856626)",   "audit_out/recall95/auditprec_willcardibperformdur_87856626_20260715_110200_models.csv"),
]


def is_iran(tag: str) -> bool:
    return tag.endswith("_0ir") or tag.endswith("_25ir")


def is_meta(tag: str) -> bool:
    return "meta" in tag


def color(tag: str) -> str:
    if is_iran(tag):
        return PURPLE
    return ORANGE if is_meta(tag) else BLUE


def pretty(tag: str) -> str:
    if tag.startswith("retrained_"):
        base = tag[len("retrained_"):]
        return "* " + LABELS.get(base, base)
    return LABELS.get(tag, tag)


def load_cvroc_rows() -> list[dict]:
    cvroc = json.load(open(CV_ROC))
    rows = []
    for r in load_models():
        entry = cvroc.get(r["tag"])
        if not entry:
            continue
        rows.append({
            "tag": r["tag"],
            "label": pretty(r["tag"]),
            "nf": r["n_features"],
            "roc": entry["roc_mean"],
            "std": entry.get("roc_std") or 0.0,
            "retrained": r["tag"].startswith("retrained_"),
        })
    return rows


def flag_counts() -> list[tuple[str, int, int]]:
    out = []
    for name, path in INSIDERS:
        d = pd.read_csv(path)
        tf = d["tuned_flag"].astype(str).str.lower().isin(["true", "1"])
        out.append((name, int(tf.sum()), len(d)))
    return out


def main() -> None:
    rows = sorted(load_cvroc_rows(), key=lambda r: r["roc"])
    fig = plt.figure(figsize=(16, 12))
    gs = fig.add_gridspec(2, 1, height_ratios=[2.4, 1.0], hspace=0.22)
    ax = fig.add_subplot(gs[0])
    axb = fig.add_subplot(gs[1])

    # ---- Panel A: ranked CV ROC-AUC ---------------------------------- #
    y = np.arange(len(rows))
    vals = [r["roc"] for r in rows]
    errs = [r["std"] for r in rows]
    bars = ax.barh(y, vals, xerr=errs, color=[color(r["tag"]) for r in rows],
                   alpha=0.85, edgecolor="black", linewidth=0.5,
                   error_kw=dict(ecolor="gray", lw=1, capsize=3))
    for b, r in zip(bars, rows):
        if r["retrained"]:
            b.set_hatch("////")
            b.set_edgecolor("black")
            b.set_linewidth(1.1)
    ax.set_yticks(y)
    ax.set_yticklabels([f"{r['label']}  (nf={r['nf']})" for r in rows], fontsize=9)
    ax.set_xlabel("cross-validated ROC-AUC (mean +/- std, grouped CV under each "
                  "model's own scheme)")
    ax.set_xlim(0, 1.0)
    ax.grid(axis="x", alpha=0.3)
    for yi, r in zip(y, rows):
        ax.text(r["roc"] + r["std"] + 0.008, yi, f"{r['roc']:.3f}",
                va="center", fontsize=8)
    n_retr = sum(r["retrained"] for r in rows)
    ax.set_title(
        f"A. All {len(rows)} insider models ranked by CV ROC-AUC  "
        f"(blue = behavioral only, orange = + metadata, purple = Iran-hidden; "
        f"hatched = {n_retr} retrained on the bet-size fix, prefixed '*')",
        fontsize=11, fontweight="bold")
    legend_handles = [
        Patch(facecolor=BLUE, edgecolor="black", label="behavioral only"),
        Patch(facecolor=ORANGE, edgecolor="black", label="+ metadata"),
        Patch(facecolor=PURPLE, edgecolor="black", label="Iran-hidden experiment"),
        Patch(facecolor="white", edgecolor="black", hatch="////",
              label="retrained (post bet-size fix)"),
    ]
    ax.legend(handles=legend_handles, loc="lower right", fontsize=9)

    # ---- Panel B: per-insider flag counts ---------------------------- #
    fc = flag_counts()
    names = [f[0] for f in fc]
    counts = [f[1] for f in fc]
    totals = [f[2] for f in fc]
    x = np.arange(len(names))
    b2 = axb.bar(x, counts, color="#c0392b", alpha=0.85, edgecolor="black",
                 linewidth=0.6, width=0.62)
    n_total = totals[0]
    axb.axhline(n_total, color="gray", ls="--", lw=1, alpha=0.7)
    axb.text(len(names) - 0.5, n_total + 0.3, f"{n_total} models scored",
             ha="right", va="bottom", fontsize=8, color="gray")
    for xi, (c, t) in zip(x, zip(counts, totals)):
        axb.text(xi, c + 0.3, f"{c}/{t}\n({100*c/t:.0f}%)",
                 ha="center", va="bottom", fontsize=8.5, fontweight="bold")
    axb.set_xticks(x)
    axb.set_xticklabels(names, fontsize=9)
    axb.set_ylabel("# models flagging the insider")
    axb.set_ylim(0, n_total + 4)
    axb.grid(axis="y", alpha=0.3)
    axb.set_title(
        "B. Consensus per found insider: how many of the scored models flag the "
        "(wallet, market) pair at each model's threshold tuned for recall >= 0.95",
        fontsize=11, fontweight="bold")

    fig.suptitle("Insider-detection model panel: CV ranking + per-insider "
                 "flag consensus (recall >= 0.95 tuned thresholds)",
                 fontsize=14, fontweight="bold")
    fig.savefig(OUT, dpi=140, bbox_inches="tight")
    print("wrote", OUT)
    print(f"\nPanel A: {len(rows)} models ({n_retr} retrained)")
    print("Panel B: flag consensus")
    for name, c, t in fc:
        print(f"  {name:<24} {c}/{t}  ({100*c/t:.0f}%)")


if __name__ == "__main__":
    main()
