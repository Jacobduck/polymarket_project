"""Render a visual breakdown of how train/test split and CV were done per run.

Reads the same cache/models/<MODEL>_*.meta.json files used by
xgboost_comparison.py and produces xgboost_cv_breakdown.png with:
  - top: per-run table of split strategy + CV scheme + train/test sizes
  - bottom: fold-by-fold breakdown for the step-2c run (representative
    StratifiedGroupKFold example).
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

MODEL_DIR = Path("cache/models")
OUT_PATH = Path("xgboost_cv_breakdown.png")

# Same row order as xgboost_comparison.py for easy cross-reference.
XGB_RUNS = [
    ("0a", "xgb 14-feat (notebook)",      "xgb_insider_14feat_latest.meta.json"),
    ("0b", "xgb 30-feat (notebook)",      "xgb_insider_30feat_latest.meta.json"),
    ("0c", "xgb 50-feat (notebook)",      "xgb_insider_50feat_latest.meta.json"),
    ("1",  "xgb2 random split",           "xgb_insider_10feat_v2_20260615_115126.meta.json"),
    ("2",  "xgb2 pinned iran-mar-1",      "xgb_insider_10feat_v2_20260615_120424.meta.json"),
    ("3",  "xgb2 step 2a tighter Optuna", "xgb_insider_10feat_v2_20260615_121020.meta.json"),
    ("4",  "xgb2 step 2b Platt cal",      "xgb_insider_10feat_v2_20260615_123115.meta.json"),
    ("5",  "xgb2 step 2c StratGroupKF",   "xgb_insider_10feat_v2_latest.meta.json"),
    ("6",  "xgb2 top-30 (step 2d)",       "xgb_insider_30feat_v2_latest.meta.json"),
    ("7",  "xgb3 top-20 (with cal)",      "xgb_insider_20feat_v2_latest.meta.json"),
    ("8",  "xgb4 top-20 (no cal)",        "xgb_insider_20feat_v2_nocal_latest.meta.json"),
]

# Worked-example run for the per-fold breakdown table.
EXAMPLE_RUN_META = "xgb_insider_10feat_v2_latest.meta.json"
EXAMPLE_RUN_TAG = "Row 5 — xgb2 step 2c (StratifiedGroupKFold, 5 folds)"


def short_market(slug: str, maxlen: int = 32) -> str:
    """Truncate giant slugs like the iran-feb-28 one to a readable label."""
    if slug.startswith("us-strikes-iran-by-february-28-2026"):
        return "iran-feb-28 (multi-variant)"
    if slug.startswith("will-us-or-israel-strike-iran-by-february-28-2026"):
        return "us-or-isr-iran-feb-28"
    if len(slug) <= maxlen:
        return slug
    return slug[: maxlen - 1] + "..."


def market_list(slugs: list[str], max_show: int = 6) -> str:
    """Join a list of market slugs with truncation."""
    shown = [short_market(s, maxlen=30) for s in slugs[:max_show]]
    extra = len(slugs) - max_show
    if extra > 0:
        shown.append(f"(+{extra} more)")
    return ", ".join(shown)


def cv_scheme_short(scheme: str) -> str:
    """Reduce a verbose CV scheme description to a short label."""
    if not scheme:
        return "(none)"
    if "StratifiedGroupKFold" in scheme:
        return "StratifiedGroupKFold(shuffle=True)"
    if "GroupKFold" in scheme:
        return "GroupKFold"
    if "TimeSeriesSplit" in scheme:
        return "TimeSeriesSplit"
    return scheme[:30]


def split_kind(strategy: str) -> str:
    if "pinned" in strategy.lower():
        return "pinned"
    if "random by market" in strategy.lower():
        return "random by market (seeded)"
    if not strategy:
        return "row-level 80/20"
    return strategy[:30]


def load_run_summary(meta_path: str) -> dict:
    m = json.load(open(MODEL_DIR / meta_path))
    cv = m.get("cv", {})
    return {
        "split_strategy": m.get("split_strategy", ""),
        "test_markets": m.get("test_markets", []),
        "n_train_markets": m.get("n_train_markets"),
        "n_test_markets": m.get("n_test_markets"),
        "n_train_rows": m.get("n_train_rows"),
        "n_test_rows": m.get("n_test_rows"),
        "cv_scheme": cv.get("scheme", ""),
        "n_folds": cv.get("n_folds"),
        "folds": cv.get("folds", []),
    }


def build_summary_rows() -> list[list[str]]:
    rows = []
    for run_id, name, meta_path in XGB_RUNS:
        try:
            s = load_run_summary(meta_path)
        except FileNotFoundError:
            rows.append([run_id, name, "-", "-", "-", "-", "-", "-", "-"])
            continue
        if not s["test_markets"]:
            test_str = "row-level 80/20 split"
        else:
            test_str = market_list(s["test_markets"], max_show=3)
        rows.append([
            run_id,
            name,
            split_kind(s["split_strategy"]),
            test_str,
            str(s["n_train_markets"]) if s["n_train_markets"] is not None else "-",
            str(s["n_train_rows"]) if s["n_train_rows"] is not None else "-",
            str(s["n_test_rows"]) if s["n_test_rows"] is not None else "-",
            cv_scheme_short(s["cv_scheme"]),
            str(s["n_folds"]) if s["n_folds"] is not None else "-",
        ])
    return rows


def build_fold_rows(meta_path: str) -> list[list[str]]:
    s = load_run_summary(meta_path)
    rows = []
    for f in s["folds"]:
        val_mks = f.get("val_markets", [])
        val_mks_str = market_list(val_mks, max_show=3) if val_mks else "-"
        pr = f.get("pr_auc")
        rows.append([
            str(f.get("fold", "-")),
            str(f.get("n_train_markets", "-")),
            str(f.get("n_val_markets", "-")),
            val_mks_str,
            str(f.get("n_val_rows", "-")),
            str(f.get("n_val_positives", "-")),
            f"{pr:.4f}" if isinstance(pr, (int, float)) else "-",
        ])
    return rows


def style_header(table, headers, header_color="#2c3e50"):
    for j in range(len(headers)):
        cell = table[0, j]
        cell.set_text_props(weight="bold", color="white")
        cell.set_facecolor(header_color)


def left_align(table, row_idx, col_idx):
    cell = table[row_idx, col_idx]
    cell.get_text().set_ha("left")
    cell.PAD = 0.02


def render(summary_rows, fold_rows, out_path: Path) -> None:
    summary_headers = [
        "#", "Run", "Split kind", "Test market(s)",
        "Train mkts", "Train rows", "Test rows",
        "CV scheme", "n folds",
    ]
    fold_headers = [
        "Fold", "Train mkts in fold", "Val mkts in fold",
        "Val markets", "Val rows", "Val positives", "Fold PR-AUC",
    ]

    fig = plt.figure(figsize=(20, 9))
    gs = fig.add_gridspec(
        2, 1,
        height_ratios=[len(summary_rows) + 1.5, len(fold_rows) + 1.5],
        hspace=0.25,
    )

    # ---- top: per-run summary ----
    ax1 = fig.add_subplot(gs[0])
    ax1.axis("off")
    ax1.set_title(
        "Train / test split and cross-validation scheme per XGBoost run\n"
        "(\"pinned\" = test markets hardcoded; \"random by market (seeded)\" = "
        "test markets picked by np.random with random_state=42)",
        loc="left", fontsize=12, fontweight="bold", pad=14,
    )

    col_widths = [0.03, 0.16, 0.16, 0.27, 0.07, 0.06, 0.06, 0.16, 0.05]
    table1 = ax1.table(
        cellText=summary_rows,
        colLabels=summary_headers,
        colWidths=col_widths,
        loc="center", cellLoc="center", colLoc="center",
    )
    table1.auto_set_font_size(False)
    table1.set_fontsize(8.5)
    table1.scale(1, 1.6)

    style_header(table1, summary_headers)

    # Color-code split kind so pinned vs random is visually obvious.
    split_col = summary_headers.index("Split kind")
    cv_col = summary_headers.index("CV scheme")
    for i, row in enumerate(summary_rows, start=1):
        kind = row[split_col]
        if "pinned" in kind:
            table1[i, split_col].set_facecolor("#d4efdf")  # light green
        elif "random by market" in kind:
            table1[i, split_col].set_facecolor("#fdebd0")  # light orange
        elif "row-level" in kind:
            table1[i, split_col].set_facecolor("#fadbd8")  # light red
        # CV scheme color
        cv = row[cv_col]
        if "Stratified" in cv:
            table1[i, cv_col].set_facecolor("#d4efdf")
        elif "GroupKFold" == cv:
            table1[i, cv_col].set_facecolor("#fdebd0")
        elif cv == "(none)":
            table1[i, cv_col].set_facecolor("#fadbd8")
        # left-align the "Run", "Split kind", "Test market(s)", "CV scheme"
        for j in (1, 2, 3, 7):
            left_align(table1, i, j)

    # ---- bottom: fold breakdown for example run ----
    ax2 = fig.add_subplot(gs[1])
    ax2.axis("off")
    ax2.set_title(
        f"Fold-by-fold composition for {EXAMPLE_RUN_TAG}\n"
        "Each fold trains on \"Train mkts in fold\" markets and validates "
        "on \"Val mkts in fold\" markets (markets never appear in both).",
        loc="left", fontsize=12, fontweight="bold", pad=14,
    )

    fold_col_widths = [0.05, 0.10, 0.10, 0.45, 0.08, 0.10, 0.10]
    table2 = ax2.table(
        cellText=fold_rows,
        colLabels=fold_headers,
        colWidths=fold_col_widths,
        loc="center", cellLoc="center", colLoc="center",
    )
    table2.auto_set_font_size(False)
    table2.set_fontsize(9)
    table2.scale(1, 1.6)
    style_header(table2, fold_headers, header_color="#1f3a5f")

    # Left-align val_markets column
    for i in range(1, len(fold_rows) + 1):
        left_align(table2, i, 3)

    # Footer legend
    fig.text(
        0.02, 0.02,
        "Green = StratifiedGroupKFold / pinned test  •  Orange = vanilla GroupKFold / "
        "random-by-market split  •  Red = no CV / row-level split",
        fontsize=8, color="#666666",
    )

    fig.savefig(out_path, dpi=140, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"saved {out_path}")


def main() -> None:
    summary_rows = build_summary_rows()
    fold_rows = build_fold_rows(EXAMPLE_RUN_META)
    render(summary_rows, fold_rows, OUT_PATH)


if __name__ == "__main__":
    main()
