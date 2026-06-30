"""Second-stage metadata model: feature build pipeline.

This script consumes the labeled (wallet, market) pairs used by the
behavioral model and emits a parallel "metadata" feature matrix using
:mod:`polycluster.metadata`. The features are documented in
``metadata_features.txt``.

Output:
  cache/metadata_features.parquet
  cache/metadata/  (per-wallet / per-funder caches, populated incidentally)

The model itself (CatBoost/RF on this feature matrix) will follow in a
sibling script once the feature matrix is verified.

Usage:
  POLYGONSCAN_API_KEY=... python metadata_layer.py
"""

from __future__ import annotations

import json
import os
import pickle
import sys
import time
from pathlib import Path

import pandas as pd

from polycluster.markets import get_market_by_slug
from polycluster.metadata import (
    METADATA_FEATURE_COLUMNS,
    compute_metadata_features,
)

ROOT = Path(__file__).resolve().parent
CACHE = ROOT / "cache"
METADATA_CACHE = CACHE / "metadata"
MARKET_START_CACHE = METADATA_CACHE / "market_start_ts.json"
OUTPUT_PARQUET = CACHE / "metadata_features.parquet"

KNOWN_INSIDER_JSON = CACHE / "known_insider_pairs.json"
TRAINING_PARQUET = CACHE / "training_data.parquet"


def load_market_pickle(slug: str) -> dict | None:
    """Load the cached market dict for a slug, or None if missing.

    The pickle has keys: ``market``, ``events``, ``history``, ``parsed``.
    ``parsed['wallet_trades']`` maps wallet -> list[trade dict] with a
    ``timestamp`` field, which is the authoritative source of the
    wallet's first-trade timestamp on the market.
    """
    p = CACHE / f"{slug}.pkl"
    if not p.exists():
        return None
    with open(p, "rb") as f:
        return pickle.load(f)


def load_labeled_pairs() -> pd.DataFrame:
    """Assemble the (wallet, market_slug, is_insider) frame to score.

    Combines two sources:
      * ``cache/known_insider_pairs.json`` - confirmed positives.
      * ``cache/training_data.parquet`` - the full behavioral training
        set, which already contains positives + negatives + the
        per-pair time column.

    The training parquet is the primary source when available because
    it carries labels and the time column we use later for the
    train/val/test split. The JSON acts as a fallback so the pipeline
    still works if the parquet is missing.
    """
    if TRAINING_PARQUET.exists():
        df = pd.read_parquet(TRAINING_PARQUET)
        wallet_col = next(
            c for c in df.columns
            if c.lower() in {"wallet", "wallet_address", "address", "proxy_wallet"}
        )
        slug_col = next(
            c for c in df.columns if c.lower() in {"market_slug", "slug"}
        )
        label_col = next(
            c for c in df.columns
            if c.lower() in {"is_insider", "label", "y"}
        )
        out = df[[wallet_col, slug_col, label_col]].rename(
            columns={
                wallet_col: "wallet",
                slug_col: "market_slug",
                label_col: "is_insider",
            }
        )
        out["wallet"] = out["wallet"].str.lower()
        return out.drop_duplicates(subset=["wallet", "market_slug"]).reset_index(drop=True)

    with open(KNOWN_INSIDER_JSON) as f:
        pairs = json.load(f)
    return pd.DataFrame(
        [
            {
                "wallet": p["wallet"].lower(),
                "market_slug": p["market_slug"],
                "is_insider": 1,
            }
            for p in pairs
        ]
    ).drop_duplicates(subset=["wallet", "market_slug"]).reset_index(drop=True)


def load_flagged_wallets() -> set[str]:
    """All wallet addresses appearing in the known-insider list."""
    if not KNOWN_INSIDER_JSON.exists():
        return set()
    with open(KNOWN_INSIDER_JSON) as f:
        pairs = json.load(f)
    return {p["wallet"].lower() for p in pairs}


def build_market_start_ts_lookup():
    """Return a callable that maps slug -> start_ts, with a JSON disk cache.

    Avoids re-hitting the gamma-api for slugs we've already resolved.
    """
    METADATA_CACHE.mkdir(parents=True, exist_ok=True)
    cache: dict[str, int] = {}
    if MARKET_START_CACHE.exists():
        with open(MARKET_START_CACHE) as f:
            cache = {k: int(v) for k, v in json.load(f).items()}

    def lookup(slug: str) -> int:
        if slug in cache:
            return cache[slug]
        market = get_market_by_slug(slug)
        cache[slug] = int(market.start_ts)
        with open(MARKET_START_CACHE, "w") as f:
            json.dump(cache, f)
        return cache[slug]

    return lookup


def main() -> None:
    api_key = os.environ.get("POLYGONSCAN_API_KEY") or os.environ.get(
        "ETHERSCAN_API_KEY"
    )
    if not api_key:
        print(
            "[metadata_layer] WARNING: POLYGONSCAN_API_KEY not set; "
            "wallet_age / funder / sibling features will be NaN.",
            file=sys.stderr,
        )

    pairs_df = load_labeled_pairs()
    print(f"[metadata_layer] {len(pairs_df)} (wallet, market) pairs to score")
    flagged = load_flagged_wallets()
    print(f"[metadata_layer] {len(flagged)} flagged wallets loaded")

    market_start_lookup = build_market_start_ts_lookup()

    # Cache of the most recently loaded market pickle, keyed by slug.
    # Pairs are processed grouped by slug (training_data.parquet is
    # already roughly clustered), so this is effectively a one-pickle-
    # at-a-time LRU of size 1.
    pairs_df = pairs_df.sort_values("market_slug").reset_index(drop=True)
    current_slug: str | None = None
    current_first_trade_ts: dict[str, int] = {}

    rows: list[dict] = []
    t0 = time.time()
    for i, row in pairs_df.iterrows():
        wallet = row["wallet"]
        slug = row["market_slug"]
        try:
            if slug != current_slug:
                pkl = load_market_pickle(slug)
                if pkl is None:
                    current_first_trade_ts = {}
                else:
                    current_first_trade_ts = {
                        w.lower(): min(int(t["timestamp"]) for t in trades)
                        for w, trades in pkl["parsed"]["wallet_trades"].items()
                        if trades
                    }
                current_slug = slug

            as_of_ts = current_first_trade_ts.get(wallet)
            market_start_ts = market_start_lookup(slug)
            feats = compute_metadata_features(
                wallet,
                slug,
                market_start_ts,
                as_of_ts=as_of_ts,
                api_key=api_key,
                flagged_wallets=flagged,
            )
            err = None
        except Exception as exc:
            feats = {col: float("nan") for col in METADATA_FEATURE_COLUMNS}
            feats.update({"as_of_ts": None, "funder": None, "first_deposit_ts": None})
            err = f"{type(exc).__name__}: {exc}"

        out_row = {
            "wallet": wallet,
            "market_slug": slug,
            "is_insider": int(row["is_insider"]),
            **feats,
            "_error": err,
        }
        rows.append(out_row)
        elapsed = time.time() - t0
        suffix = f" ERR: {err}" if err else ""
        print(
            f"[metadata_layer] {i + 1}/{len(pairs_df)} "
            f"({wallet[:8]}…, {slug[:32]}) "
            f"elapsed={elapsed:.1f}s{suffix}",
            flush=True,
        )

    out = pd.DataFrame(rows)
    OUTPUT_PARQUET.parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(OUTPUT_PARQUET, index=False)
    print(f"[metadata_layer] wrote {OUTPUT_PARQUET} (shape={out.shape})")


if __name__ == "__main__":
    main()
