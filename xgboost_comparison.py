"""Render a visual comparison table of all XGBoost fine-tuning runs.

Reads metrics straight from cache/models/<MODEL>_*.meta.json and produces
xgboost_comparison.png -- one figure with two color-graded tables (xgb runs
on top, non-xgb reference models below).
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

MODEL_DIR = Path("cache/models")
OUT_PATH = Path("xgboost_comparison.png")

# (id, name, feats, best-params string, meta filename)
XGB_RUNS = [
    ("0a", "xgb 14-feat (notebook 80/20)",       14, "-",                       "xgb_insider_14feat_latest.meta.json"),
    ("0b", "xgb 30-feat (notebook 80/20)",       30, "-",                       "xgb_insider_30feat_latest.meta.json"),
    ("0c", "xgb 50-feat (notebook 80/20)",       50, "-",                       "xgb_insider_50feat_latest.meta.json"),
    ("1",  "xgb2 random split",                  10, "d=7, lr=0.069, n=500",    "xgb_insider_10feat_v2_20260615_115126.meta.json"),
    ("2",  "xgb2 pinned iran-mar-1",             10, "d=5, lr=0.019, n=100",    "xgb_insider_10feat_v2_20260615_120424.meta.json"),
    ("3",  "xgb2 step 2a (tighter Optuna)",      10, "d=5, lr=0.045, n=1000",   "xgb_insider_10feat_v2_20260615_121020.meta.json"),
    ("4",  "xgb2 step 2b (Platt cal)",           10, "d=5, lr=0.045, n=1000",   "xgb_insider_10feat_v2_20260615_123115.meta.json"),
    ("5",  "xgb2 step 2c (StratGroupKFold)",     10, "d=3, lr=0.014, n=1050",   "xgb_insider_10feat_v2_latest.meta.json"),
    ("6",  "xgb2 top-30 (step 2d)",              30, "d=4, lr=0.007, n=300",    "xgb_insider_30feat_v2_latest.meta.json"),
    ("7",  "xgb3 top-20 (with cal)",             20, "d=2, lr=0.007, n=300",    "xgb_insider_20feat_v2_latest.meta.json"),
    ("8",  "xgb4 top-20 (no cal)",               20, "d=2, lr=0.007, n=300",    "xgb_insider_20feat_v2_nocal_latest.meta.json"),
]

REF_MODELS = [
    ("cb3", "catboost 10-feat v2",    "cb_insider_10feat_v2_latest.meta.json"),
    ("cb4", "catboost 10-feat v2 d2", "cb_insider_10feat_v2_d2_latest.meta.json"),
    ("cb2", "catboost 30-feat v2",    "cb_insider_30feat_v2_latest.meta.json"),
    ("rf",  "random forest 10-feat",  "rf_insider_10feat_v2_latest.meta.json"),
]


def load_metrics(path: str) -> dict:
    m = json.load(open(MODEL_DIR / path))
    metrics = m.get("metrics", {})
    cv = m.get("cv", {})
    # Old notebook runs only have flat roc/pr at metrics root.
    if "test" not in metrics:
        return {
            "cv_pr_auc": None,
            "test_pr_auc": metrics.get("pr_auc"),
            "test_roc": metrics.get("roc_auc"),
            "p": None, "r": None, "f1": None,
            "thr": None,
        }
    test = metrics["test"]
    return {
        "cv_pr_auc": cv.get("pr_auc_mean"),
        "test_pr_auc": test["pr_auc"],
        "test_roc": test["roc_auc"],
        "p": test["precision"],
        "r": test["recall"],
        "f1": test["f1"],
        "thr": m.get("threshold"),
    }


def fmt(v) -> str:
    if v is None:
        return "-"
    return f"{v:.4f}"


def color_for(val_str: str, vmin: float, vmax: float) -> tuple:
    if val_str == "-":
        return (1, 1, 1, 1)
    try:
        v = float(val_str)
    except ValueError:
        return (1, 1, 1, 1)
    if vmax == vmin:
        return (0.85, 0.95, 0.85, 1)
    norm = max(0.0, min(1.0, (v - vmin) / (vmax - vmin)))
    cmap = plt.get_cmap("RdYlGn")
    return cmap(0.15 + 0.7 * norm)


def style_header(table, headers, header_color="#2c3e50"):
    for j in range(len(headers)):
        cell = table[0, j]
        cell.set_text_props(weight="bold", color="white")
        cell.set_facecolor(header_color)


def color_grade_column(table, rows, col_idx, *, bold_best: bool = True):
    vals = [float(r[col_idx]) for r in rows if r[col_idx] != "-"]
    if not vals:
        return
    vmin, vmax = min(vals), max(vals)
    best_idx = max(
        (i for i, r in enumerate(rows) if r[col_idx] != "-"),
        key=lambda i: float(rows[i][col_idx]),
    )
    for i, row in enumerate(rows, start=1):
        table[i, col_idx].set_facecolor(color_for(row[col_idx], vmin, vmax))
        if bold_best and (i - 1) == best_idx:
            table[i, col_idx].set_text_props(weight="bold", color="black")


def build_xgb_rows() -> list[list[str]]:
    rows = []
    for run_id, name, feats, params, path in XGB_RUNS:
        try:
            mx = load_metrics(path)
        except FileNotFoundError:
            mx = {k: None for k in ("cv_pr_auc", "test_pr_auc", "test_roc", "p", "r", "f1", "thr")}
        thr = mx["thr"]
        thr_str = f"{thr:.2f}" if isinstance(thr, (int, float)) else "-"
        rows.append([
            run_id, name, str(feats), params,
            fmt(mx["cv_pr_auc"]),
            fmt(mx["test_pr_auc"]),
            fmt(mx["test_roc"]),
            thr_str,
            fmt(mx["p"]),
            fmt(mx["r"]),
            fmt(mx["f1"]),
        ])
    return rows


def build_ref_rows() -> list[list[str]]:
    rows = []
    for tag, name, path in REF_MODELS:
        try:
            mx = load_metrics(path)
        except FileNotFoundError:
            continue
        thr = mx["thr"]
        thr_str = f"{thr:.2f}" if isinstance(thr, (int, float)) else "-"
        rows.append([
            tag, name,
            fmt(mx["test_pr_auc"]),
            fmt(mx["test_roc"]),
            thr_str,
            fmt(mx["p"]),
            fmt(mx["r"]),
            fmt(mx["f1"]),
        ])
    return rows


def render(xgb_rows, ref_rows, out_path: Path) -> None:
    xgb_headers = [
        "#", "Run", "Feats", "Best params",
        "CV PR-AUC", "Test PR-AUC", "Test ROC",
        "thr", "P@thr", "R@thr", "F1@thr",
    ]
    ref_headers = [
        "Tag", "Model",
        "Test PR-AUC", "Test ROC",
        "thr", "P@thr", "R@thr", "F1@thr",
    ]

    fig = plt.figure(figsize=(19, 9))
    gs = fig.add_gridspec(
        2, 1,
        height_ratios=[len(xgb_rows) + 1.5, len(ref_rows) + 1.5],
        hspace=0.22,
    )

    # ---- top: XGB history ----
    ax1 = fig.add_subplot(gs[0])
    ax1.axis("off")
    ax1.set_title(
        "XGBoost insider-detection fine-tuning history\n"
        "test market = us-strikes-iran-by-march-1-2026-492  "
        "(245 rows / 35 positives)",
        loc="left", fontsize=13, fontweight="bold", pad=14,
    )

    col_widths = [0.04, 0.20, 0.04, 0.17, 0.08, 0.09, 0.08, 0.05, 0.08, 0.08, 0.08]
    table1 = ax1.table(
        cellText=xgb_rows,
        colLabels=xgb_headers,
        colWidths=col_widths,
        loc="center",
        cellLoc="center",
        colLoc="center",
    )
    table1.auto_set_font_size(False)
    table1.set_fontsize(9)
    table1.scale(1, 1.55)

    style_header(table1, xgb_headers)
    color_grade_column(table1, xgb_rows, xgb_headers.index("Test PR-AUC"))
    color_grade_column(table1, xgb_rows, xgb_headers.index("F1@thr"))

    # Left-align the "Run" and "Best params" columns for readability.
    for i in range(1, len(xgb_rows) + 1):
        for j in (1, 3):
            table1[i, j].get_text().set_ha("left")
            table1[i, j].PAD = 0.02

    # ---- bottom: reference models ----
    ax2 = fig.add_subplot(gs[1])
    ax2.axis("off")
    ax2.set_title(
        "Reference non-XGBoost models (same pinned test market)",
        loc="left", fontsize=12, fontweight="bold", pad=8,
    )

    ref_col_widths = [0.06, 0.28, 0.12, 0.12, 0.06, 0.10, 0.10, 0.10]
    table2 = ax2.table(
        cellText=ref_rows,
        colLabels=ref_headers,
        colWidths=ref_col_widths,
        loc="center",
        cellLoc="center",
        colLoc="center",
    )
    table2.auto_set_font_size(False)
    table2.set_fontsize(9)
    table2.scale(1, 1.55)

    style_header(table2, ref_headers, header_color="#1f3a5f")
    color_grade_column(table2, ref_rows, ref_headers.index("Test PR-AUC"))
    color_grade_column(table2, ref_rows, ref_headers.index("F1@thr"))
    for i in range(1, len(ref_rows) + 1):
        table2[i, 1].get_text().set_ha("left")
        table2[i, 1].PAD = 0.02

    fig.text(
        0.02, 0.02,
        "Color grade: red = worst, green = best (per column). "
        "Bold = best in column. Source: cache/models/<MODEL>.meta.json",
        fontsize=8, color="#666666",
    )

    fig.savefig(out_path, dpi=140, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"saved {out_path}")


def main() -> None:
    xgb_rows = build_xgb_rows()
    ref_rows = build_ref_rows()
    render(xgb_rows, ref_rows, OUT_PATH)


if __name__ == "__main__":
    main()
