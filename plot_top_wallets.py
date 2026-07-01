#!/usr/bin/env python
"""plot_top_wallets.py — render the top-N wallets' trades on a market's YES-price
curve, for a market that was scored by supercompute.py BEFORE the plotting
feature existed (e.g. the first d4vd run).

This is a thin, standalone re-do of *only* the plotting stage. It:
  1. Reads the per-wallet scores from an existing supercompute results CSV
     (``supercompute_<slug8>_<ts>_results.csv``) — so NOTHING is recomputed
     (no model load, no metadata, no API key needed).
  2. Loads the market + OrderFilled events from ``cache/events/<slug>.pkl``
     (already cached by the original run — works offline).
  3. Rebuilds the price history + per-wallet trades and renders the top-N
     highest-scored wallets, one PNG each, via the SAME code path
     supercompute.py now uses (``render_top_wallet_plots``).

Output: a folder ``<results-stem>_plots/`` next to the CSV, containing
``rank<NN>_<wallet>.png`` (rank-prefixed so a listing sorts by score).

Usage (on Insomnia, in the project dir with the venv active):
    python plot_top_wallets.py \
        --market will-d4vd-be-the-1-searched-person-on-google-this-year \
        --results-csv supercompute_willd4vd_20260630_XXXXXX_results.csv

If --results-csv is omitted, the newest matching results CSV in --out-dir is used.
"""

from __future__ import annotations

import argparse
import glob
import logging
import sys
import time
from pathlib import Path

import pandas as pd

# Importing supercompute sets a headless matplotlib backend and pulls in the
# exact plotting helpers (render_top_wallet_plots) and cache/event loaders, so
# this stays in lockstep with how the real runs produce plots.
import supercompute as sc


def _find_results_csv(out_dir: Path, slug8: str, explicit: str | None) -> Path:
    if explicit:
        p = Path(explicit)
        if not p.exists():
            raise SystemExit(f"results CSV not found: {p}")
        return p
    pattern = str(out_dir / f"supercompute_{slug8}_*_results.csv")
    matches = sorted(glob.glob(pattern))
    if not matches:
        raise SystemExit(
            f"no results CSV matching {pattern}. Pass one with --results-csv."
        )
    return Path(matches[-1])  # newest by timestamp in the name


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Plot top-N wallets' trades for an already-scored market."
    )
    p.add_argument("--market", required=True, help="Polymarket market slug")
    p.add_argument("--results-csv", default=None,
                   help="existing supercompute *_results.csv (default: newest match)")
    p.add_argument("--top-n", type=int, default=25,
                   help="how many top-scored wallets to plot (default: 25)")
    p.add_argument("--out-dir", default=".",
                   help="where to write the plots folder + log (default: .)")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    slug = args.market
    slug8 = slug.replace("-", "")[:24] or "market"

    log_path, ts = sc.configure_logging(out_dir, slug8)
    logging.info("=" * 60)
    logging.info(f"plot_top_wallets starting, log={log_path}")
    logging.info(f"market={slug}  top_n={args.top_n}")

    # --- scores from the existing CSV (no recompute) -------------------- #
    results_csv = _find_results_csv(out_dir, slug8, args.results_csv)
    logging.info(f"reading scores from {results_csv}")
    df = pd.read_csv(results_csv)
    for col in ("wallet", "pred_prob", "pred_prob_raw"):
        if col not in df.columns:
            raise SystemExit(
                f"results CSV {results_csv} missing required column '{col}'."
            )
    df = df.sort_values("pred_prob", ascending=False).reset_index(drop=True)
    logging.info(f"loaded {len(df)} scored wallets")

    # --- market + events from cache, rebuild history + trades ----------- #
    market, events = sc.load_market_and_events(slug)
    duration_days = (market.end_ts - market.start_ts) / 86400.0
    logging.info(f"market resolved: tokens={len(market.token_ids)} "
                 f"closed={market.closed} duration={duration_days:.1f}d  "
                 f"events={len(events)}")

    history_df = sc.build_history_df_from_orderfilled(
        events, yes_token=market.yes_token_id, no_token=market.no_token_id,
    )
    parsed = sc.parse_orderfilled_events(events, market.token_ids, market.outcomes)
    wallet_trades_map = parsed["wallet_trades"]
    logging.info(f"wallets with parsed trades: {len(wallet_trades_map)}")

    # --- render (same path supercompute uses) --------------------------- #
    plots_dir = out_dir / f"{results_csv.stem.replace('_results', '')}_plots"
    written = sc.render_top_wallet_plots(
        df, wallet_trades_map, history_df, market, plots_dir, top_n=args.top_n,
    )
    logging.info("=" * 60)
    logging.info(f"DONE: {written} plots written to {plots_dir}")
    logging.info(f"log saved to {log_path}")


if __name__ == "__main__":
    main()
