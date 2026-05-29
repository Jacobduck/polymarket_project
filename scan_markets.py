"""Scan multiple markets for insider-like behavior using the 30-feat XGBoost model.

For each market slug in ``MARKET_SLUGS``:

  1. Fetch OrderFilled events (recursive-split fetcher, cached at
     ``cache/events/<slug>.pkl``).
  2. Pick the top ``TOP_N_WALLETS`` wallets by lifetime cash volume.
  3. Compute only the 30 features the 30-feat model uses
     (``feature_filter`` in ``polycluster.features``) and cache at
     ``cache/features_30feat/<slug>.parquet``.
  4. Score every wallet with
     ``cache/models/xgb_insider_30feat_latest.json`` and flag those with
     ``pred_prob >= THRESHOLD``.

Logs go to both stdout and ``scan_markets_<timestamp>.log``.
"""

from __future__ import annotations

import json
import logging
import pickle
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb

from polycluster.events import get_market_orderfilled_events
from polycluster.features import compute_market_user_features
from polycluster.markets import get_market_by_slug
from polycluster.parsing import (
    build_history_df_from_orderfilled,
    parse_orderfilled_events,
)

CACHE = Path("cache")
EVENTS_DIR = CACHE / "events"
FEATURES_DIR = CACHE / "features_30feat"
MODEL_DIR = CACHE / "models"

MODEL_PATH = MODEL_DIR / "xgb_insider_30feat_latest.json"
META_PATH = MODEL_DIR / "xgb_insider_30feat_latest.meta.json"

THRESHOLD = 0.7
TOP_N_WALLETS = 100

MARKET_SLUGS: list[str] = [
    "will-russia-capture-all-of-hryshyne-by-april-30",
    "strait-of-hormuz-traffic-returns-to-normal-by-july-31",
    "us-iran-nuclear-deal-by-may-31-974",
    "us-iran-nuclear-deal-by-june-30",
    "will-the-us-reopen-its-embassy-in-iran-in-2026",
]


def configure_logging() -> str:
    ts = time.strftime("%Y%m%d_%H%M%S")
    log_path = f"scan_markets_{ts}.log"
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    for h in list(logger.handlers):
        logger.removeHandler(h)
    fmt = logging.Formatter("%(asctime)s [scan] %(message)s", datefmt="%H:%M:%S")
    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(fmt)
    logger.addHandler(sh)
    fh = logging.FileHandler(log_path)
    fh.setFormatter(fmt)
    logger.addHandler(fh)
    return log_path


def load_model() -> tuple[xgb.XGBClassifier, list[str]]:
    with open(META_PATH) as f:
        meta = json.load(f)
    features = meta["features"]
    model = xgb.XGBClassifier()
    model.load_model(str(MODEL_PATH))
    return model, features


def fetch_or_load_events(slug: str, market):
    EVENTS_DIR.mkdir(parents=True, exist_ok=True)
    pkl = EVENTS_DIR / f"{slug}.pkl"
    if pkl.exists():
        logging.info(f"  events: loading from cache {pkl}")
        with open(pkl, "rb") as f:
            _cached_market, events = pickle.load(f)
        logging.info(f"  events: {len(events)} loaded")
        return events

    logging.info("  events: fetching (recursive-split, both tokens, both sides)...")
    t0 = time.time()
    events = get_market_orderfilled_events(market, verbose=True)
    elapsed = time.time() - t0
    logging.info(f"  events: fetched {len(events)} in {elapsed:.1f}s")
    with open(pkl, "wb") as f:
        pickle.dump((market, events), f)
    logging.info(f"  events: cached to {pkl}")
    return events


def compute_features_for_market(
    slug: str,
    market,
    events: list[dict],
    model_features: list[str],
) -> pd.DataFrame:
    FEATURES_DIR.mkdir(parents=True, exist_ok=True)
    parquet = FEATURES_DIR / f"{slug}.parquet"
    if parquet.exists():
        logging.info(f"  features: loading from cache {parquet}")
        return pd.read_parquet(parquet)

    history_df = build_history_df_from_orderfilled(
        events,
        yes_token=market.yes_token_id,
        no_token=market.no_token_id,
    )
    parsed = parse_orderfilled_events(events, market.token_ids, market.outcomes)
    wallet_trades_map = parsed["wallet_trades"]

    wallets_by_volume = sorted(
        wallet_trades_map.items(),
        key=lambda kv: sum(float(t["cash_amount"]) for t in kv[1]),
        reverse=True,
    )[:TOP_N_WALLETS]
    logging.info(
        f"  wallets: {len(wallet_trades_map)} total, "
        f"taking top {len(wallets_by_volume)} by volume"
    )

    if market.closed and market.outcome_prices:
        final_outcome_yes = 1 if float(market.outcome_prices[0]) >= 0.5 else 0
    else:
        final_outcome_yes = None
    logging.info(
        f"  market: closed={market.closed}, final_outcome_yes={final_outcome_yes}"
    )

    filter_set = set(model_features)
    rows: list[dict] = []
    t_compute_start = time.time()
    for i, (wallet, trades) in enumerate(wallets_by_volume, 1):
        volume = sum(float(t["cash_amount"]) for t in trades)
        logging.info(
            f"  wallet {i:3d}/{len(wallets_by_volume)}: {wallet} "
            f"trades={len(trades)} volume=${volume:,.0f}"
        )
        if not trades:
            logging.info("    skipped (no trades)")
            continue
        t0 = time.time()
        feats = compute_market_user_features(
            parsed_trades=trades,
            history_df=history_df,
            market_start_time=market.start_ts,
            market_end_time=market.end_ts,
            final_outcome_yes=final_outcome_yes,
            feature_filter=filter_set,
        )
        elapsed = time.time() - t0
        row: dict = {"wallet": wallet, "volume": volume, "n_trades": len(trades)}
        for c in model_features:
            row[c] = feats.get(c, np.nan)
        rows.append(row)
        logging.info(f"    features computed in {elapsed:.2f}s")

    total_compute = time.time() - t_compute_start
    df = pd.DataFrame(rows)
    df.to_parquet(parquet, index=False)
    logging.info(
        f"  features: computed {len(df)} wallets in {total_compute:.1f}s, "
        f"saved to {parquet}"
    )
    return df


def score_and_flag(
    df: pd.DataFrame,
    model: xgb.XGBClassifier,
    features_30: list[str],
    slug: str,
) -> tuple[pd.DataFrame, int]:
    if len(df) == 0:
        logging.info("  no wallets to score")
        return df, 0
    X = df[features_30].astype(float)
    proba = model.predict_proba(X)[:, 1]
    df = df.copy()
    df["pred_prob"] = proba
    df["flagged"] = (proba >= THRESHOLD).astype(int)
    n_flagged = int(df["flagged"].sum())
    logging.info(
        f"  scoring: {n_flagged}/{len(df)} flagged at threshold {THRESHOLD}"
    )
    for _, r in df[df["flagged"] == 1].sort_values("pred_prob", ascending=False).iterrows():
        logging.info(
            f"    FLAGGED  {r['wallet']}  pred_prob={r['pred_prob']:.3f}  "
            f"volume=${r['volume']:,.0f}"
        )
    return df, n_flagged


def main() -> None:
    log_path = configure_logging()
    logging.info("=" * 60)
    logging.info(f"scan_markets starting, log={log_path}")
    logging.info(f"model: {MODEL_PATH}")
    logging.info(f"threshold: {THRESHOLD}, top_N: {TOP_N_WALLETS}")
    logging.info("=" * 60)

    if not MARKET_SLUGS:
        logging.info("MARKET_SLUGS is empty — edit scan_markets.py to add slugs.")
        return

    model, features_30 = load_model()
    logging.info(f"loaded 30-feat model with {len(features_30)} features")

    summary: list[tuple[str, int, int]] = []
    for i, slug in enumerate(MARKET_SLUGS, 1):
        logging.info("")
        logging.info("=" * 60)
        logging.info(f"market {i}/{len(MARKET_SLUGS)}: {slug}")
        logging.info("=" * 60)

        t_market = time.time()
        try:
            market = get_market_by_slug(slug)
            duration_days = (market.end_ts - market.start_ts) / 86400.0
            logging.info(
                f"  resolved: tokens={len(market.token_ids)} closed={market.closed} "
                f"duration={duration_days:.1f}d"
            )
            events = fetch_or_load_events(slug, market)
            df = compute_features_for_market(slug, market, events, features_30)
            df, n_flagged = score_and_flag(df, model, features_30, slug)
            elapsed = time.time() - t_market
            logging.info(f"  market done in {elapsed:.1f}s: {n_flagged} flagged")
            summary.append((slug, len(df), n_flagged))
        except Exception as exc:
            logging.exception(f"  ERROR processing {slug}: {exc}")
            summary.append((slug, 0, 0))

    logging.info("")
    logging.info("=" * 60)
    logging.info("SUMMARY")
    logging.info("=" * 60)
    total = 0
    for slug, n, n_flagged in summary:
        logging.info(f"  {slug}: scored={n}, flagged={n_flagged}")
        total += n_flagged
    logging.info(f"TOTAL FLAGGED: {total} users across {len(MARKET_SLUGS)} markets")
    logging.info(f"log saved to {log_path}")


if __name__ == "__main__":
    main()
