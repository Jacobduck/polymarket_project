"""Render an insider vs non-insider comparison table for metadata features.

Reads ``cache/metadata_features.parquet`` (produced by
``metadata_layer.py``) and emits:

1. A console summary printed to stdout.
2. A PNG figure with a header block explaining the data source and a
   side-by-side stats table (insider mean / non-insider mean / ratio /
   coverage), saved as
   ``metadata_separability_<YYYYMMDD_HHMMSS>.png``.

Per-feature display units are human-readable: durations are converted
to days, USDC sizes are printed with a ``$`` prefix.

Usage:
    .venv/bin/python metadata_separability.py
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
PARQUET = ROOT / "cache" / "metadata_features.parquet"

# (column, display_label, unit) — unit ∈ {"days", "usdc", "count", "ratio", "share"}
FEATURE_DISPLAY: list[tuple[str, str, str]] = [
    ("wallet_age_seconds", "Wallet age", "days"),
    ("market_age_seconds", "Market age at first trade", "days"),
    ("wallet_minus_market_age_seconds", "Wallet age − market age", "days"),
    ("wallet_to_market_age_ratio", "Wallet/market age ratio", "ratio"),
    ("time_since_first_deposit_to_first_trade_seconds",
     "Time from deposit to first trade", "days"),
    ("n_markets_traded", "Markets traded (lifetime)", "count"),
    ("is_first_market", "Is first market (rate)", "share"),
    ("herfindahl_index_markets", "Herfindahl across markets", "ratio"),
    ("lifetime_volume_share_on_market", "Volume share on this market", "share"),
    ("winrate", "Winrate across markets", "share"),
    ("pnl_sharpe", "PnL Sharpe", "ratio"),
    ("avg_bet_size_usdc", "Avg bet size", "usdc"),
    ("median_bet_size_usdc", "Median bet size", "usdc"),
    ("total_bet_size_usdc", "Lifetime volume", "usdc"),
    ("n_wallets_same_funder", "Wallets sharing funder", "count"),
    ("n_other_wallets_same_funder", "Other wallets sharing funder", "count"),
    ("funder_flagged_wallet_count", "Flagged wallets w/ same funder", "count"),
]


def _fmt(val: float, unit: str) -> str:
    if not np.isfinite(val):
        return "—"
    if unit == "days":
        d = val / 86_400
        if abs(d) >= 1000:
            return f"{d:,.0f}d"
        if abs(d) >= 10:
            return f"{d:,.1f}d"
        return f"{d:,.2f}d"
    if unit == "usdc":
        return f"${val:,.0f}"
    if unit == "count":
        return f"{val:,.1f}"
    if unit == "ratio":
        if abs(val) >= 100:
            return f"{val:,.0f}×"
        return f"{val:,.2f}×"
    if unit == "share":
        return f"{val * 100:,.1f}%"
    return f"{val:,.3f}"


def _ratio_label(insider: float, non_insider: float) -> str:
    """Insider-vs-non-insider multiplier, sign-aware."""
    if not (np.isfinite(insider) and np.isfinite(non_insider)):
        return "—"
    if non_insider == 0:
        return "∞" if insider != 0 else "1.0×"
    r = insider / non_insider
    if r < 0:
        return f"{r:,.2f}×"
    if abs(r) >= 100:
        return f"{r:,.0f}×"
    return f"{r:,.2f}×"


def main() -> None:
    df = pd.read_parquet(PARQUET)
    n_insider = int((df["is_insider"] == 1).sum())
    n_noninsider = int((df["is_insider"] == 0).sum())
    n_markets = df["market_slug"].nunique()
    n_wallets = df["wallet"].nunique()

    rows: list[dict] = []
    for col, label, unit in FEATURE_DISPLAY:
        ins = df.loc[df["is_insider"] == 1, col]
        non = df.loc[df["is_insider"] == 0, col]
        ins_mean = ins.mean()
        non_mean = non.mean()
        coverage = 1.0 - df[col].isna().mean()
        rows.append(
            {
                "Feature": label,
                "Insider": _fmt(ins_mean, unit),
                "Non-insider": _fmt(non_mean, unit),
                "Ratio (ins/non)": _ratio_label(ins_mean, non_mean),
                "Coverage": f"{coverage * 100:,.0f}%",
            }
        )

    table_df = pd.DataFrame(rows)
    print("\nMetadata feature separability — insider vs non-insider means\n")
    print(table_df.to_string(index=False))
    print(
        f"\nN = {len(df)} pairs ({n_insider} insider / {n_noninsider} "
        f"non-insider), {n_wallets} wallets, {n_markets} markets\n"
    )

    # --- figure ------------------------------------------------------
    fig_height = 1.6 + 0.32 * len(table_df) + 1.8
    fig, ax = plt.subplots(figsize=(11, fig_height))
    ax.axis("off")

    header = (
        f"Metadata features — insider vs non-insider comparison\n"
        f"N = {len(df):,} (wallet, market) pairs   "
        f"|   {n_insider:,} insider   |   {n_noninsider:,} non-insider   "
        f"|   {n_wallets:,} wallets   |   {n_markets:,} markets"
    )
    fig.suptitle(header, fontsize=12, y=0.985, ha="center")

    data_note = (
        "Source: cache/metadata_features.parquet, produced by "
        "metadata_layer.py from cache/training_data.parquet.\n"
        "As-of time per row = wallet's first TRADE on its market (from "
        "cache/<slug>.pkl → parsed['wallet_trades']).\n"
        "Data sources: Polymarket data-api /activity (TRADE/REDEEM), "
        "gamma-api (market start_ts),\n"
        "Etherscan v2 chainid=137 (USDC transfers on Polygon, via the "
        "Polygonscan API key).\n"
        "Means are computed only over rows where the feature is "
        "non-NaN; 'Coverage' shows that share."
    )
    fig.text(
        0.06, 0.92, data_note,
        fontsize=8.5, va="top", ha="left", color="#333333",
        family="DejaVu Sans Mono",
    )

    tbl = ax.table(
        cellText=table_df.values.tolist(),
        colLabels=table_df.columns.tolist(),
        loc="center",
        cellLoc="left",
        colLoc="left",
    )
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(9)
    tbl.scale(1.0, 1.35)

    # Style header row
    for j in range(len(table_df.columns)):
        c = tbl[0, j]
        c.set_facecolor("#222e3a")
        c.set_text_props(color="white", weight="bold")
    # Zebra body
    for i in range(1, len(table_df) + 1):
        for j in range(len(table_df.columns)):
            c = tbl[i, j]
            c.set_facecolor("#f6f7fb" if i % 2 == 0 else "white")
    # Highlight strongly-separating rows (ratio >= 3× or <= 0.33×)
    for i, r in enumerate(rows, start=1):
        ratio_text = r["Ratio (ins/non)"]
        try:
            val = float(ratio_text.rstrip("×").replace(",", ""))
            strong = (val >= 3.0) or (0 < val <= 1 / 3)
        except ValueError:
            strong = ratio_text in {"∞"}
        if strong:
            for j in range(len(table_df.columns)):
                tbl[i, j].set_facecolor("#fff4d6")

    plt.subplots_adjust(left=0.04, right=0.96, top=0.78, bottom=0.04)

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out = ROOT / f"metadata_separability_{ts}.png"
    fig.savefig(out, dpi=160, bbox_inches="tight")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
