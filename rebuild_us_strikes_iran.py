"""One-shot recovery for the us-strikes-iran-by-feb-28 slug.

Re-runs the cell-5 (labeling) and cell-6 (features) logic from xgboost.ipynb,
but only for the long ``us-strikes-iran-by-february-28-2026-227-...-189-761``
slug. Now that the complete event pickle is in place under the new
deterministic slug_key, re-labeling will include the 8 insider wallets that
were previously dropped, and feature rows will be regenerated to match.

After running:
  * ``cache/labeled_pairs.parquet`` is rewritten with the slug's rows replaced.
  * ``cache/features/<slug_key>.parquet`` is regenerated with the new wallet set.
"""

from __future__ import annotations

import hashlib
import json
import pickle
import random
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from polycluster import (
    parse_orderfilled_events,
    build_history_df_from_orderfilled,
    compute_market_user_features,
)

RANDOM_SEED = 42
NEG_PER_POS = 6
TARGET_SLUG = (
    "us-strikes-iran-by-february-28-2026-227-967-547-688-589-491-592-418-452-"
    "924-384-915-464-672-196-157-993-596-269-535-381-391-471-256-988-997-296-"
    "225-762-973-292-827-345-182-558-215-794-879-189-761"
)

CACHE_DIR = Path("cache")
EVENTS_DIR = CACHE_DIR / "events"
FEATURES_DIR = CACHE_DIR / "features"


def slug_key(slug: str) -> str:
    if len(slug) <= 180:
        return slug
    return f"{slug[:170]}__{hashlib.md5(slug.encode()).hexdigest()[:8]}"


def main() -> int:
    key = slug_key(TARGET_SLUG)
    event_path = EVENTS_DIR / f"{key}.pkl"
    feat_path = FEATURES_DIR / f"{key}.parquet"
    labeled_path = CACHE_DIR / "labeled_pairs.parquet"
    pairs_path = CACHE_DIR / "known_insider_pairs.json"

    if not event_path.exists():
        print(f"FATAL: event pickle missing at {event_path}", file=sys.stderr)
        return 1

    print(f"slug_key = {key}")
    print(f"loading events from {event_path}")
    with event_path.open("rb") as f:
        market, events = pickle.load(f)
    print(f"  {len(events):,} events, market.closed={market.closed}")

    pairs = json.loads(pairs_path.read_text())
    all_insider_wallets = {p["wallet"].lower() for p in pairs}
    insiders_for_slug = sorted(
        {p["wallet"].lower() for p in pairs if p["market_slug"] == TARGET_SLUG}
    )
    print(f"  {len(insiders_for_slug)} known insiders for this slug")

    parsed = parse_orderfilled_events(events, market.token_ids, market.outcomes)
    wallet_volume = {
        w.lower(): sum(float(t["cash_amount"]) for t in trades)
        for w, trades in parsed["wallet_trades"].items()
    }
    traded = set(wallet_volume)
    insiders_present = [w for w in insiders_for_slug if w in traded]
    missing = [w for w in insiders_for_slug if w not in traded]
    print(f"  insiders present in events: {len(insiders_present)}")
    print(f"  insiders absent: {missing}")

    n_pos = len(insiders_present)
    n_want = NEG_PER_POS * n_pos
    raw_pool = sorted(traded - all_insider_wallets)
    insider_vols = np.array([wallet_volume[w] for w in insiders_present], dtype=float)
    vol_floor = float(np.percentile(insider_vols, 25))
    eligible_pool = [w for w in raw_pool if wallet_volume[w] >= vol_floor]
    rng = random.Random(f"{RANDOM_SEED}-{TARGET_SLUG}")
    if len(eligible_pool) < n_want:
        sampled = eligible_pool
        print(f"  WARNING: only {len(eligible_pool)} negatives available (wanted {n_want})")
    else:
        sampled = rng.sample(eligible_pool, n_want)
    print(f"  sampled {len(sampled)} negatives from eligible pool of {len(eligible_pool):,} "
          f"(≥ ${vol_floor:,.0f} floor)")

    new_rows = [
        {"wallet": w, "market_slug": TARGET_SLUG, "is_insider": 1}
        for w in insiders_present
    ] + [
        {"wallet": w, "market_slug": TARGET_SLUG, "is_insider": 0}
        for w in sampled
    ]
    new_df = pd.DataFrame(new_rows)

    existing = pd.read_parquet(labeled_path)
    kept = existing[existing.market_slug != TARGET_SLUG]
    rewritten = pd.concat([kept, new_df], ignore_index=True)
    rewritten.to_parquet(labeled_path, index=False)
    print(
        f"  rewrote {labeled_path}: kept {len(kept)} rows, added "
        f"{len(new_df)} ({n_pos} pos + {len(sampled)} neg). total now {len(rewritten)}"
    )

    final_outcome_yes = None
    if market.closed and market.outcome_prices:
        final_outcome_yes = 1 if float(market.outcome_prices[0]) >= 0.5 else 0

    history_df = build_history_df_from_orderfilled(
        events, yes_token=market.yes_token_id, no_token=market.no_token_id,
    )

    wallets_for_feat = new_df.wallet.tolist()
    total = len(wallets_for_feat)
    print(f"  computing features for {total} wallets", flush=True)
    rows = []
    import time
    t0 = time.time()
    for i, w in enumerate(wallets_for_feat, 1):
        wl = w.lower()
        trades = parsed["wallet_trades"].get(wl, [])
        if not trades:
            rows.append({"wallet": wl, "market_slug": TARGET_SLUG})
            print(f"    [{i}/{total}] {wl[:10]}… no trades (skipped)", flush=True)
            continue
        tw0 = time.time()
        feats = compute_market_user_features(
            parsed_trades=trades,
            history_df=history_df,
            market_start_time=market.start_ts,
            market_end_time=market.end_ts,
            final_outcome_yes=final_outcome_yes,
        )
        feats["wallet"] = wl
        feats["market_slug"] = TARGET_SLUG
        rows.append(feats)
        dt = time.time() - tw0
        elapsed = time.time() - t0
        eta = elapsed / i * (total - i)
        print(
            f"    [{i}/{total}] {wl[:10]}… {len(trades)} trades, {dt:.1f}s "
            f"(elapsed {elapsed/60:.1f}m, eta {eta/60:.1f}m)",
            flush=True,
        )
    feat_df = pd.DataFrame(rows)
    feat_df.to_parquet(feat_path, index=False)
    print(f"  wrote {feat_path}: {len(feat_df)} rows", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
