"""supercompute.py — score EVERY wallet in a Polymarket market with the best
model we have (xgb 20b+10m v7) and log the wallets flagged as insiders.

Built to run on Columbia's Insomnia HPC cluster so the CPU-heavy per-wallet
feature computation is parallelized across many cores. The model itself scores
in milliseconds; the expensive part is computing the 20 behavioral features for
every wallet that traded the market, which this script farms out to a process
pool (``--jobs``). The 10 metadata features are network/cache bound, so they
run in a thread pool (``--meta-jobs``).

Pipeline (single self-contained pass — "mode (a)"):
  1. Resolve the market by slug.
  2. Fetch OrderFilled events (needs internet) OR reuse cache/events/<slug>.pkl
     if already present (so the same file works offline once cached).
  3. Build the history frame + parse per-wallet trades.
  4. For EVERY wallet (or top-N by volume if --top-n given), compute v7's exact
     30-feature vector: 20 behavioral (process pool) + 10 metadata (thread pool).
  5. Score with xgb_insider_meta_v7, apply the saved Platt calibrator, flag
     wallets with calibrated prob >= --threshold (default 0.15, v7's value).
  6. Write a human-readable log + a results CSV (all wallets) + a flagged CSV.

The v7 feature list, threshold, and calibrator are read from the model's
artifacts in cache/models/, so this stays in sync with how v7 was trained.

Metadata API key (for the funder-graph features incl. funder_flagged_wallet_count):
  read from $POLYGONSCAN_API_KEY / $ETHERSCAN_API_KEY, or pass --api-key.
  On Insomnia, put `export POLYGONSCAN_API_KEY=...` in ~/.bashrc; SLURM's default
  --export=ALL propagates it to the compute node. Without a key those 3 features
  are NaN (v7 tolerates NaN natively). Never hard-code the key in this file.

Example SLURM batch script (scan.sh):
  #!/bin/bash
  #SBATCH --account=<your_account>
  #SBATCH --job-name=insider_scan
  #SBATCH --nodes=1
  #SBATCH --cpus-per-task=32
  #SBATCH --mem=64G
  #SBATCH --time=0-02:00
  #SBATCH --output=logs/scan_%j.out
  #SBATCH --error=logs/scan_%j.err
  module load anaconda
  conda activate ./envs/poly
  python supercompute.py --market <slug> --jobs ${SLURM_CPUS_PER_TASK:-32}
Then: sbatch scan.sh  (and rsync the supercompute_*.log / *_results.csv back).

Outputs (in --out-dir, default "."):
  supercompute_<slug8>_<ts>.log
  supercompute_<slug8>_<ts>_results.csv   (every scored wallet)
  supercompute_<slug8>_<ts>_flagged.csv   (flagged wallets only)
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import pickle
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from multiprocessing import Pool
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import xgboost as xgb

from polycluster.events import get_market_orderfilled_events
from polycluster.features import compute_market_user_features
from polycluster.markets import get_market_by_slug
from polycluster.metadata import (
    METADATA_FEATURE_COLUMNS,
    compute_metadata_features,
)
from polycluster.parsing import (
    build_history_df_from_orderfilled,
    parse_orderfilled_events,
)

CACHE = Path("cache")
EVENTS_DIR = CACHE / "events"
MODEL_DIR = CACHE / "models"
KNOWN_INSIDER_JSON = CACHE / "known_insider_pairs.json"

MODEL_TAG = "xgb_insider_meta_v7"
MODEL_PATH = MODEL_DIR / f"{MODEL_TAG}_latest.json"
META_PATH = MODEL_DIR / f"{MODEL_TAG}_latest.meta.json"
CALIBRATOR_PATH = MODEL_DIR / f"{MODEL_TAG}_latest.calibrator.joblib"

DEFAULT_THRESHOLD = 0.15


# --------------------------------------------------------------------------- #
# Logging
# --------------------------------------------------------------------------- #
def configure_logging(out_dir: Path, slug8: str) -> tuple[str, str]:
    ts = time.strftime("%Y%m%d_%H%M%S")
    log_path = str(out_dir / f"supercompute_{slug8}_{ts}.log")
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    for h in list(logger.handlers):
        logger.removeHandler(h)
    fmt = logging.Formatter("%(asctime)s [super] %(message)s", datefmt="%H:%M:%S")
    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(fmt)
    logger.addHandler(sh)
    fh = logging.FileHandler(log_path)
    fh.setFormatter(fmt)
    logger.addHandler(fh)
    return log_path, ts


# --------------------------------------------------------------------------- #
# Model / config loading
# --------------------------------------------------------------------------- #
def load_model_bundle() -> tuple[xgb.XGBClassifier, object | None, list[str],
                                  list[str], list[str], float]:
    """Return (model, calibrator, all_features, behavioral, metadata, threshold).

    Everything is read from v7's saved artifacts so feature order / threshold
    match how the model was trained.
    """
    if not META_PATH.exists() or not MODEL_PATH.exists():
        raise SystemExit(
            f"missing v7 artifacts: {MODEL_PATH} / {META_PATH}. "
            f"Stage cache/models/{MODEL_TAG}_latest.* onto the cluster."
        )
    with open(META_PATH) as f:
        meta = json.load(f)
    features = list(meta["features"])
    threshold = float(meta.get("threshold", DEFAULT_THRESHOLD))

    model = xgb.XGBClassifier()
    model.load_model(str(MODEL_PATH))

    calibrator = None
    if CALIBRATOR_PATH.exists():
        calibrator = joblib.load(CALIBRATOR_PATH)
    else:
        logging.warning(
            f"calibrator not found at {CALIBRATOR_PATH}; using RAW xgb "
            f"probabilities (threshold {threshold} was tuned on CALIBRATED probs)"
        )

    meta_set = set(METADATA_FEATURE_COLUMNS)
    behavioral = [c for c in features if c not in meta_set]
    metadata = [c for c in features if c in meta_set]
    return model, calibrator, features, behavioral, metadata, threshold


def apply_calibrator(calibrator, raw) -> np.ndarray:
    """Apply a Platt (LogisticRegression) calibrator to raw model probs.

    Computes the sigmoid directly from ``coef_``/``intercept_`` instead of
    calling ``predict_proba``. scikit-learn's ``predict_proba`` internals are
    version-coupled (the ``multi_class`` attribute was removed in 1.7+), so a
    calibrator pickled with one version can crash when loaded under another.
    The direct computation is equivalent for binary Platt scaling and is
    version-proof. Falls back to ``predict_proba`` if the linear params are
    unavailable.
    """
    raw_arr = np.asarray(raw, dtype=float).ravel()
    try:
        coef = np.asarray(calibrator.coef_, dtype=float).ravel()[0]
        intercept = float(np.asarray(calibrator.intercept_).ravel()[0])
        z = raw_arr * coef + intercept
        p1 = 1.0 / (1.0 + np.exp(-z))
        classes = getattr(calibrator, "classes_", np.array([0, 1]))
        if len(classes) == 2 and classes[1] == 0:
            p1 = 1.0 - p1  # positive-class column is classes_[1]
        return p1
    except AttributeError:
        return calibrator.predict_proba(raw_arr.reshape(-1, 1))[:, 1]


def load_flagged_wallets() -> set[str]:
    if not KNOWN_INSIDER_JSON.exists():
        return set()
    with open(KNOWN_INSIDER_JSON) as f:
        pairs = json.load(f)
    return {p["wallet"].lower() for p in pairs}


def resolve_api_key(cli_key: str | None) -> str | None:
    return (
        cli_key
        or os.environ.get("POLYGONSCAN_API_KEY")
        or os.environ.get("ETHERSCAN_API_KEY")
    )


def load_market_and_events(slug: str):
    """Resolve market + OrderFilled events, preferring the on-disk cache.

    If cache/events/<slug>.pkl exists we load BOTH the market and events from
    it, so a cached market needs no internet at all (works on an offline
    compute node). Only on a cache miss do we hit the network — first to
    resolve the market by slug, then to fetch + cache events.
    """
    EVENTS_DIR.mkdir(parents=True, exist_ok=True)
    pkl = EVENTS_DIR / f"{slug}.pkl"
    if pkl.exists():
        logging.info(f"events: loading market + events from cache {pkl}")
        with open(pkl, "rb") as f:
            market, events = pickle.load(f)
        logging.info(f"events: {len(events)} loaded from cache (offline-ok)")
        return market, events
    logging.info("events: cache miss -> resolving market + fetching (needs internet)...")
    market = get_market_by_slug(slug)
    t0 = time.time()
    events = get_market_orderfilled_events(market, verbose=True)
    logging.info(f"events: fetched {len(events)} in {time.time() - t0:.1f}s")
    with open(pkl, "wb") as f:
        pickle.dump((market, events), f)
    logging.info(f"events: cached to {pkl}")
    return market, events


# --------------------------------------------------------------------------- #
# Behavioral features (CPU-bound) -> process pool
# --------------------------------------------------------------------------- #
_CTX: dict = {}


def _init_behavioral_worker(history_df, m_start, m_end, final_yes, behav_filter):
    _CTX["history_df"] = history_df
    _CTX["m_start"] = m_start
    _CTX["m_end"] = m_end
    _CTX["final_yes"] = final_yes
    _CTX["behav_filter"] = set(behav_filter)


def _behavioral_one(item):
    wallet, trades = item
    feats = compute_market_user_features(
        parsed_trades=trades,
        history_df=_CTX["history_df"],
        market_start_time=_CTX["m_start"],
        market_end_time=_CTX["m_end"],
        final_outcome_yes=_CTX["final_yes"],
        feature_filter=_CTX["behav_filter"],
    )
    return wallet, {c: feats.get(c, np.nan) for c in _CTX["behav_filter"]}


def compute_behavioral(
    items: list[tuple[str, list]],
    history_df,
    m_start,
    m_end,
    final_yes,
    behav_filter: list[str],
    jobs: int,
) -> dict[str, dict]:
    out: dict[str, dict] = {}
    if jobs <= 1 or len(items) <= 1:
        _init_behavioral_worker(history_df, m_start, m_end, final_yes, behav_filter)
        for i, item in enumerate(items, 1):
            w, feats = _behavioral_one(item)
            out[w] = feats
            if i % 100 == 0 or i == len(items):
                logging.info(f"behavioral: {i}/{len(items)} wallets")
        return out

    logging.info(f"behavioral: {len(items)} wallets across {jobs} processes")
    t0 = time.time()
    with Pool(
        processes=jobs,
        initializer=_init_behavioral_worker,
        initargs=(history_df, m_start, m_end, final_yes, behav_filter),
    ) as pool:
        done = 0
        for w, feats in pool.imap_unordered(_behavioral_one, items, chunksize=8):
            out[w] = feats
            done += 1
            if done % 200 == 0 or done == len(items):
                logging.info(f"behavioral: {done}/{len(items)} wallets "
                             f"({time.time() - t0:.0f}s)")
    logging.info(f"behavioral: done in {time.time() - t0:.1f}s")
    return out


# --------------------------------------------------------------------------- #
# Metadata features (network/cache-bound) -> thread pool
# --------------------------------------------------------------------------- #
def _metadata_one(
    wallet: str,
    trades: list,
    slug: str,
    m_start: int,
    api_key: str | None,
    flagged: set[str],
    needed: list[str],
) -> tuple[str, dict, int | None]:
    trade_ts = [int(t["timestamp"]) for t in trades if t.get("timestamp") is not None]
    as_of = min(trade_ts) if trade_ts else None
    try:
        mf = compute_metadata_features(
            wallet.lower(), slug, int(m_start),
            as_of_ts=as_of, api_key=api_key, flagged_wallets=flagged,
        )
        return wallet, {c: mf.get(c, np.nan) for c in needed}, as_of
    except Exception as exc:  # noqa: BLE001
        logging.warning(f"metadata failed for {wallet}: {type(exc).__name__}: {exc}")
        return wallet, {c: np.nan for c in needed}, as_of


def compute_metadata(
    items: list[tuple[str, list]],
    slug: str,
    m_start: int,
    api_key: str | None,
    flagged: set[str],
    needed: list[str],
    meta_jobs: int,
) -> tuple[dict[str, dict], dict[str, int | None]]:
    feats: dict[str, dict] = {}
    as_of_map: dict[str, int | None] = {}
    if not needed:
        return feats, as_of_map
    logging.info(f"metadata: {len(items)} wallets across {meta_jobs} threads "
                 f"(api_key={'set' if api_key else 'MISSING -> funder feats NaN'})")
    t0 = time.time()
    with ThreadPoolExecutor(max_workers=max(1, meta_jobs)) as ex:
        futs = [
            ex.submit(_metadata_one, w, trades, slug, m_start, api_key, flagged, needed)
            for w, trades in items
        ]
        done = 0
        for fut in as_completed(futs):
            w, mf, as_of = fut.result()
            feats[w] = mf
            as_of_map[w] = as_of
            done += 1
            if done % 200 == 0 or done == len(items):
                logging.info(f"metadata: {done}/{len(items)} wallets "
                             f"({time.time() - t0:.0f}s)")
    logging.info(f"metadata: done in {time.time() - t0:.1f}s")
    return feats, as_of_map


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Score every wallet in a market with xgb v7.")
    p.add_argument("--market", required=True, help="Polymarket market slug")
    p.add_argument("--jobs", type=int, default=os.cpu_count() or 4,
                   help="processes for behavioral features (default: all cores)")
    p.add_argument("--meta-jobs", type=int, default=8,
                   help="threads for metadata (network) features (default: 8)")
    p.add_argument("--top-n", type=int, default=0,
                   help="score only top-N wallets by volume (0 = every wallet)")
    p.add_argument("--threshold", type=float, default=None,
                   help="flag cutoff on calibrated prob (default: v7's saved 0.15)")
    p.add_argument("--api-key", default=None,
                   help="Polygonscan/Etherscan key (else uses env var)")
    p.add_argument("--out-dir", default=".", help="where to write log + CSVs")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    slug = args.market
    slug8 = slug.replace("-", "")[:24] or "market"

    log_path, ts = configure_logging(out_dir, slug8)
    logging.info("=" * 60)
    logging.info(f"supercompute starting, log={log_path}")
    logging.info(f"market={slug}  jobs={args.jobs}  meta_jobs={args.meta_jobs}  "
                 f"top_n={'all' if args.top_n == 0 else args.top_n}")

    model, calibrator, features, behavioral, metadata_feats, model_thr = \
        load_model_bundle()
    threshold = args.threshold if args.threshold is not None else model_thr
    logging.info(f"model={MODEL_PATH.name}  features={len(features)} "
                 f"({len(behavioral)} behavioral + {len(metadata_feats)} metadata)  "
                 f"calibrator={'yes' if calibrator else 'NO'}  threshold={threshold}")

    flagged_wallets = load_flagged_wallets()
    api_key = resolve_api_key(args.api_key)
    logging.info(f"known-insider wallets for funder signal: {len(flagged_wallets)}")
    if not api_key:
        logging.warning("no API key; funder-graph metadata features will be NaN")

    # --- resolve market + events ---------------------------------------- #
    market, events = load_market_and_events(slug)
    duration_days = (market.end_ts - market.start_ts) / 86400.0
    logging.info(f"market resolved: tokens={len(market.token_ids)} "
                 f"closed={market.closed} duration={duration_days:.1f}d")

    history_df = build_history_df_from_orderfilled(
        events, yes_token=market.yes_token_id, no_token=market.no_token_id,
    )
    parsed = parse_orderfilled_events(events, market.token_ids, market.outcomes)
    wallet_trades_map = parsed["wallet_trades"]

    if market.closed and market.outcome_prices:
        final_yes = 1 if float(market.outcome_prices[0]) >= 0.5 else 0
    else:
        final_yes = None
    logging.info(f"market: closed={market.closed} final_outcome_yes={final_yes}  "
                 f"wallets that traded: {len(wallet_trades_map)}")

    # --- choose wallets -------------------------------------------------- #
    def _volume(trades) -> float:
        return sum(float(t["cash_amount"]) for t in trades)

    items = [(w, tr) for w, tr in wallet_trades_map.items() if tr]
    items.sort(key=lambda kv: _volume(kv[1]), reverse=True)
    if args.top_n and args.top_n > 0:
        items = items[: args.top_n]
    logging.info(f"scoring {len(items)} wallets")
    if not items:
        logging.info("no wallets to score; exiting")
        return

    volume_map = {w: _volume(tr) for w, tr in items}
    ntrades_map = {w: len(tr) for w, tr in items}

    # --- features -------------------------------------------------------- #
    behav = compute_behavioral(
        items, history_df, market.start_ts, market.end_ts, final_yes,
        behavioral, args.jobs,
    )
    meta_vals, as_of_map = compute_metadata(
        items, slug, int(market.start_ts), api_key, flagged_wallets,
        metadata_feats, args.meta_jobs,
    )

    # --- assemble feature matrix in EXACT model order ------------------- #
    rows: list[dict] = []
    for w, _tr in items:
        row = {"wallet": w, "volume": volume_map[w], "n_trades": ntrades_map[w],
               "first_trade_ts": as_of_map.get(w)}
        b = behav.get(w, {})
        m = meta_vals.get(w, {})
        for c in behavioral:
            row[c] = b.get(c, np.nan)
        for c in metadata_feats:
            row[c] = m.get(c, np.nan)
        rows.append(row)
    df = pd.DataFrame(rows)

    # --- score + calibrate ---------------------------------------------- #
    X = df[features].astype(float)
    raw = model.predict_proba(X)[:, 1]
    if calibrator is not None:
        proba = apply_calibrator(calibrator, raw)
    else:
        proba = raw
    df["pred_prob_raw"] = raw
    df["pred_prob"] = proba
    df["flagged"] = (proba >= threshold).astype(int)
    df = df.sort_values("pred_prob", ascending=False).reset_index(drop=True)

    n_flagged = int(df["flagged"].sum())
    logging.info("=" * 60)
    logging.info(f"RESULT: {n_flagged}/{len(df)} wallets flagged at "
                 f"calibrated prob >= {threshold}")
    logging.info("=" * 60)

    # human-readable info columns for the log
    info_cols = [c for c in ("winrate", "n_markets_traded", "herfindahl_index_markets")
                 if c in df.columns]
    for _, r in df[df["flagged"] == 1].iterrows():
        ft = r["first_trade_ts"]
        ft_str = (time.strftime("%Y-%m-%d %H:%M", time.gmtime(int(ft)))
                  if pd.notna(ft) else "n/a")
        extra = "  ".join(
            f"{c}={r[c]:.3f}" if pd.notna(r[c]) else f"{c}=NaN" for c in info_cols
        )
        logging.info(
            f"FLAGGED {r['wallet']}  prob={r['pred_prob']:.3f} "
            f"(raw={r['pred_prob_raw']:.3f})  vol=${r['volume']:,.0f}  "
            f"trades={int(r['n_trades'])}  first_trade={ft_str}  {extra}"
        )

    # --- write CSVs ------------------------------------------------------ #
    results_csv = out_dir / f"supercompute_{slug8}_{ts}_results.csv"
    flagged_csv = out_dir / f"supercompute_{slug8}_{ts}_flagged.csv"
    lead_cols = ["wallet", "pred_prob", "pred_prob_raw", "flagged", "volume",
                 "n_trades", "first_trade_ts"]
    ordered = lead_cols + [c for c in df.columns if c not in lead_cols]
    df[ordered].to_csv(results_csv, index=False)
    df[df["flagged"] == 1][ordered].to_csv(flagged_csv, index=False)
    logging.info(f"wrote {results_csv}")
    logging.info(f"wrote {flagged_csv}")
    logging.info(f"log saved to {log_path}")


if __name__ == "__main__":
    main()
