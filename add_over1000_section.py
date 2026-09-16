"""Back-fill the "wallets with volume > $1,000" section into supercompute logs.

supercompute.py now prints, just before its full per-wallet ranked dump, a
focused section listing only the wallets that traded > $1,000 (same fields and
format, prob-descending). This script adds that same section to logs that were
produced *before* that feature existed, reading the numbers from each log's
companion ``*_results.csv`` so nothing has to be re-run on the cluster.

For every ``supercompute_*.log`` that has a matching ``*_results.csv`` it:
  * builds the section lines in the exact log format (``HH:MM:SS [super] ...``),
    reusing the timestamp of the log's own ranked-section header,
  * inserts the section immediately before the
    ``per-wallet scores, ranked high->low`` line (falling back to end-of-file
    if that marker is absent),
  * is idempotent — a log that already contains the section is left untouched.

Usage:
  python add_over1000_section.py                 # every *.log in the cwd
  python add_over1000_section.py FILE.log ...    # specific logs
"""

from __future__ import annotations

import glob
import re
import sys
import time

import pandas as pd

VOL_THRESHOLD = 1000.0
INFO_COLS = ["winrate", "n_markets_traded", "herfindahl_index_markets"]
RANKED_MARKER = "per-wallet scores, ranked high->low"
SECTION_MARKER = "wallets with volume > $1,000, ranked high->low"
PREFIX_RE = re.compile(r"^(\d\d:\d\d:\d\d \[super\] )")


def _results_csv_for(log_path: str) -> str | None:
    stem = log_path[:-4] if log_path.endswith(".log") else log_path
    cand = f"{stem}_results.csv"
    import os
    return cand if os.path.exists(cand) else None


def _wallet_line(rank: int, r: pd.Series, info_cols: list[str]) -> str:
    ft = r["first_trade_ts"]
    ft_str = (time.strftime("%Y-%m-%d %H:%M", time.gmtime(int(ft)))
              if pd.notna(ft) else "n/a")
    flag = "*" if int(r["flagged"]) == 1 else " "
    extra = "  ".join(
        f"{c}={r[c]:.3f}" if pd.notna(r[c]) else f"{c}=NaN" for c in info_cols
    )
    return (
        f"{flag} #{rank + 1:<5d} {r['wallet']}  prob={r['pred_prob']:.4f} "
        f"(raw={r['pred_prob_raw']:.4f})  vol=${r['volume']:,.0f}  "
        f"trades={int(r['n_trades'])}  first_trade={ft_str}  {extra}"
    )


def build_section(csv_path: str, prefix: str) -> list[str]:
    df = pd.read_csv(csv_path)
    df = df.sort_values("pred_prob", ascending=False).reset_index(drop=True)
    info_cols = [c for c in INFO_COLS if c in df.columns]
    over = df[df["volume"] > VOL_THRESHOLD]
    lines = [f"{prefix}{SECTION_MARKER} ({len(over)} of {len(df)}):"]
    for i, r in over.iterrows():
        lines.append(prefix + _wallet_line(i, r, info_cols))
    lines.append(prefix + "=" * 60)
    return lines


def process(log_path: str) -> str:
    csv_path = _results_csv_for(log_path)
    if csv_path is None:
        return f"skip (no results.csv): {log_path}"

    with open(log_path) as f:
        lines = f.read().splitlines()

    if any(SECTION_MARKER in ln for ln in lines):
        return f"skip (already has section): {log_path}"

    # Anchor: the ranked-section header. Reuse its timestamp prefix so the
    # inserted lines look native, and insert the section right before it.
    anchor_idx = next((i for i, ln in enumerate(lines)
                       if RANKED_MARKER in ln), None)
    if anchor_idx is not None:
        m = PREFIX_RE.match(lines[anchor_idx])
        prefix = m.group(1) if m else ""
        section = build_section(csv_path, prefix)
        new_lines = lines[:anchor_idx] + section + lines[anchor_idx:]
        where = f"inserted before ranked section (line {anchor_idx + 1})"
    else:
        m = next((PREFIX_RE.match(ln) for ln in reversed(lines)
                  if PREFIX_RE.match(ln)), None)
        prefix = m.group(1) if m else ""
        section = build_section(csv_path, prefix)
        new_lines = lines + section
        where = "appended at end (no ranked marker found)"

    with open(log_path, "w") as f:
        f.write("\n".join(new_lines) + "\n")
    n = len(section) - 2  # minus header and divider
    return f"ok ({n} wallets, {where}): {log_path}"


def main() -> None:
    targets = sys.argv[1:] or sorted(glob.glob("supercompute_*.log"))
    if not targets:
        print("no logs found")
        return
    for log_path in targets:
        print(process(log_path))


if __name__ == "__main__":
    main()
