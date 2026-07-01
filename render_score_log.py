"""render_score_log.py — turn a supercompute *_results.csv into a ranked log.

The heavy supercompute run already computed and saved every wallet's score in
its ``*_results.csv``. This script re-renders those scores as a human-readable
log, ranked high->low, so you can see the per-wallet scores in log form without
re-running the (expensive) scan.

Usage:
    python render_score_log.py supercompute_<slug>_<ts>_results.csv
    python render_score_log.py <results.csv> --top-n 50 --threshold 0.15

Writes ``<results_basename>_ranked.log`` next to the CSV and also prints to
stdout.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import pandas as pd


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("results_csv", help="a supercompute *_results.csv file")
    p.add_argument("--top-n", type=int, default=0,
                   help="show only this many top-scored wallets (0 = all)")
    p.add_argument("--threshold", type=float, default=0.15,
                   help="mark wallets at/above this calibrated prob with '*'")
    args = p.parse_args()

    csv_path = Path(args.results_csv)
    df = pd.read_csv(csv_path)
    if "pred_prob" not in df.columns:
        raise SystemExit(f"{csv_path} has no 'pred_prob' column; is it a "
                         f"supercompute results CSV?")
    df = df.sort_values("pred_prob", ascending=False).reset_index(drop=True)

    info_cols = [c for c in ("winrate", "n_markets_traded",
                             "herfindahl_index_markets") if c in df.columns]
    n_log = len(df) if args.top_n <= 0 else min(args.top_n, len(df))
    n_flagged = int((df["pred_prob"] >= args.threshold).sum())

    lines = []
    lines.append("=" * 60)
    lines.append(f"ranked scores from {csv_path.name}")
    lines.append(f"{len(df)} wallets scored | {n_flagged} at prob >= "
                 f"{args.threshold} | showing top {n_log}")
    lines.append("=" * 60)
    for i, r in df.head(n_log).iterrows():
        ft = r.get("first_trade_ts")
        ft_str = (time.strftime("%Y-%m-%d %H:%M", time.gmtime(int(ft)))
                  if pd.notna(ft) else "n/a")
        flag = "*" if r["pred_prob"] >= args.threshold else " "
        raw = r.get("pred_prob_raw", float("nan"))
        vol = r.get("volume", float("nan"))
        ntr = r.get("n_trades", float("nan"))
        extra = "  ".join(
            f"{c}={r[c]:.3f}" if pd.notna(r[c]) else f"{c}=NaN"
            for c in info_cols
        )
        lines.append(
            f"{flag} #{i + 1:<5d} {r['wallet']}  prob={r['pred_prob']:.4f} "
            f"(raw={raw:.4f})  vol=${vol:,.0f}  "
            f"trades={int(ntr) if pd.notna(ntr) else 'n/a'}  "
            f"first_trade={ft_str}  {extra}"
        )
    lines.append("=" * 60)

    text = "\n".join(lines)
    print(text)
    out_path = csv_path.with_name(csv_path.stem.replace("_results", "") + "_ranked.log")
    out_path.write_text(text + "\n")
    print(f"\nwrote {out_path}")


if __name__ == "__main__":
    main()
