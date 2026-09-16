"""render_plots.py — regenerate top-wallet trade plots from a completed run.

A supercompute run writes a ``*_results.csv`` (every scored wallet, ranked
high->low) and, for runs made with the plotting code, a ``*_plots/`` folder
with one PNG per top wallet. This script re-renders those per-wallet plots
*offline* — from the results CSV plus the cached OrderFilled events
(``cache/events/<slug>.pkl``) — WITHOUT re-running the (expensive) scan.

Its main use is extending or backfilling a plots folder: e.g. a run that only
plotted the top 25 can get ranks 26-50 added with

    python render_plots.py supercompute_<slug8>_<ts>_results.csv \
        --rank-start 26 --rank-end 50

Ranks are 1-based positions in the CSV's pred_prob (calibrated) ordering,
which equals the raw-prob ordering (the Platt calibrator is monotonic). PNGs
are written as ``rank<NN>_<wallet>.png`` into the plots folder (by default the
sibling ``*_plots/`` folder next to the CSV), so new ranks slot in alongside
existing ones and a directory listing stays score-sorted. Markers sit at each
wallet's ACTUAL fill price (``use_trade_price=True``) to expose off-market
fills — identical style to supercompute's inline plotting.

The market slug (for the events cache) is auto-detected by matching the CSV's
embedded slug8 against ``cache/events/*.pkl``; override with ``--slug`` if the
guess is wrong or ambiguous.
"""

from __future__ import annotations

import argparse
import os
import pickle
import re
from pathlib import Path

# Headless backend before pyplot is pulled in via polycluster.viz.
os.environ.setdefault("MPLBACKEND", "Agg")

import pandas as pd

from polycluster.parsing import (
    build_history_df_from_orderfilled,
    map_trades_to_history,
    normalize_trades_to_yes_df,
    parse_orderfilled_events,
)
from polycluster.viz import plot_user_on_history

CACHE = Path("cache")
EVENTS_DIR = CACHE / "events"


def _slug8(s: str) -> str:
    """The same 24-char slug key supercompute embeds in output filenames."""
    return s.replace("-", "")[:24]


def _extract_slug8_from_csv(csv_path: Path) -> str | None:
    """Pull the ``slug8`` token out of a supercompute_<slug8>_<ts>_results.csv."""
    m = re.match(r"supercompute_(.+?)_\d{8}_\d{6}_results", csv_path.stem)
    return m.group(1) if m else None


def _resolve_events_pkl(csv_path: Path, slug: str | None) -> tuple[Path, str]:
    """Locate the cached events pkl for this run.

    If ``slug`` is given we use it directly. Otherwise we recover the slug8
    token from the CSV filename and match it against cache/events/*.pkl by
    recomputing each candidate's slug8.

    Returns (pkl_path, resolved_slug).
    """
    if slug:
        pkl = EVENTS_DIR / f"{slug}.pkl"
        if not pkl.exists():
            raise SystemExit(f"events cache not found: {pkl}")
        return pkl, slug

    target = _extract_slug8_from_csv(csv_path)
    if target is None:
        raise SystemExit(
            f"could not parse slug8 from {csv_path.name}; pass --slug explicitly"
        )
    candidates = [
        p for p in EVENTS_DIR.glob("*.pkl") if _slug8(p.stem) == target
    ]
    if not candidates:
        raise SystemExit(
            f"no cache/events/*.pkl matches slug8 {target!r}; pass --slug "
            f"explicitly (expected the market's full slug)"
        )
    if len(candidates) > 1:
        names = ", ".join(p.stem for p in candidates)
        raise SystemExit(
            f"multiple events caches match slug8 {target!r}: {names}. "
            f"Disambiguate with --slug."
        )
    return candidates[0], candidates[0].stem


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("results_csv", help="a supercompute *_results.csv file")
    p.add_argument("--slug", default=None,
                   help="market slug for cache/events/<slug>.pkl "
                        "(default: auto-detect from the CSV filename)")
    p.add_argument("--rank-start", type=int, default=1,
                   help="first rank to plot, 1-based inclusive (default 1)")
    p.add_argument("--rank-end", type=int, default=50,
                   help="last rank to plot, 1-based inclusive (default 50)")
    p.add_argument("--out-dir", default=None,
                   help="plots folder (default: sibling *_plots/ next to the CSV)")
    p.add_argument("--overwrite", action="store_true",
                   help="re-render even if a rank's PNG already exists")
    args = p.parse_args()

    csv_path = Path(args.results_csv)
    df = pd.read_csv(csv_path)
    if "pred_prob" not in df.columns or "wallet" not in df.columns:
        raise SystemExit(f"{csv_path} lacks 'pred_prob'/'wallet'; is it a "
                         f"supercompute results CSV?")
    df = df.sort_values("pred_prob", ascending=False).reset_index(drop=True)

    if args.out_dir is not None:
        plots_dir = Path(args.out_dir)
    else:
        base = re.sub(r"_results$", "", csv_path.stem)
        plots_dir = csv_path.with_name(f"{base}_plots")
    plots_dir.mkdir(parents=True, exist_ok=True)

    pkl, slug = _resolve_events_pkl(csv_path, args.slug)
    with open(pkl, "rb") as f:
        market, events = pickle.load(f)
    print(f"loaded {len(events)} events from {pkl}")

    history_df = build_history_df_from_orderfilled(
        events, yes_token=market.yes_token_id, no_token=market.no_token_id,
    )
    parsed = parse_orderfilled_events(events, market.token_ids, market.outcomes)
    wallet_trades_map = parsed["wallet_trades"]

    import matplotlib.pyplot as plt

    lo = max(1, args.rank_start)
    hi = min(args.rank_end, len(df))
    print(f"rendering ranks {lo}..{hi} of {len(df)} into {plots_dir}")

    written = skipped = failed = 0
    for i in range(lo - 1, hi):
        r = df.iloc[i]
        rank = i + 1
        wallet = str(r["wallet"])
        out_png = plots_dir / f"rank{rank:02d}_{wallet}.png"
        if out_png.exists() and not args.overwrite:
            skipped += 1
            continue
        try:
            trades = (wallet_trades_map.get(wallet.lower(), [])
                      or wallet_trades_map.get(wallet, []))
            trades_df = normalize_trades_to_yes_df(
                trades, yes_token=market.yes_token_id,
                no_token=market.no_token_id,
            )
            mapped_df = map_trades_to_history(trades_df, history_df)
            fig, ax = plot_user_on_history(
                mapped_df, history_df, use_trade_price=True, show=False,
            )
            ax.set_title(
                f"#{rank}  {wallet}  prob={r['pred_prob']:.4f} "
                f"(raw={r['pred_prob_raw']:.4f})  {slug}"
            )
            fig.savefig(out_png, dpi=120, bbox_inches="tight")
            plt.close(fig)
            written += 1
            print(f"wrote {out_png.name}  ({len(mapped_df)} trades)")
        except Exception as exc:  # noqa: BLE001
            failed += 1
            print(f"FAILED #{rank} {wallet}: {type(exc).__name__}: {exc}")

    print(f"\ndone: {written} written, {skipped} skipped (exist), "
          f"{failed} failed -> {plots_dir}")


if __name__ == "__main__":
    main()
