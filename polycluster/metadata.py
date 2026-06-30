"""Wallet-level metadata features for the second-stage insider model.

Where :mod:`polycluster.features` looks at *how* a wallet trades one
specific market, this module looks at the wallet's broader footprint:
how many markets it touches, its lifetime winrate, its bet sizing, the
age of the wallet relative to the market, and signals that come from
the EOA that funded the proxy (sibling wallets sharing a funder).

All features are evaluated at an *as-of time* equal to the wallet's
first TRADE on the market under question. Anything after that timestamp
is dropped to prevent leakage.

Top-level entry points:

* :func:`get_wallet_activity` - pull a wallet's TRADE/REDEEM history
  from the Polymarket data-api.
* :func:`get_wallet_first_usdc_deposit` - first incoming USDC transfer
  on Polygon (Polygonscan API). Returns the funder + timestamp.
* :func:`get_funder_outgoing_wallets` - distinct recipients of outgoing
  USDC transfers from the funder EOA (candidate sibling wallets).
* :func:`compute_metadata_features` - the orchestrator that returns the
  full per-(wallet, market) metadata feature dict.

External requirements: ``POLYGONSCAN_API_KEY`` env var for on-chain
features. Without it those features are returned as NaN.

The features are documented in ``metadata_features.txt`` at the repo
root.
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
import requests

DATA_API_BASE = "https://data-api.polymarket.com"
# Polygonscan's standalone v1 API (api.polygonscan.com) was retired in
# 2025; the same API key now works against Etherscan's unified v2
# endpoint with chainid=137 (Polygon mainnet).
ETHERSCAN_V2_BASE = "https://api.etherscan.io/v2/api"
POLYGON_CHAIN_ID = 137
USDC_POLYGON = "0x2791bca1f2de4661ed88a30c99a7a9449aa84174"

# The data-api /activity endpoint rejects offset > 3000 with HTTP 400,
# mirroring the /trades endpoint. Wallets with more activity get
# truncated at the oldest end.
DATA_API_MAX_OFFSET = 3_000
DATA_API_ACTIVITY_CAP = 10_000
FUNDER_SIBLING_CAP = 1_000

DEFAULT_CACHE_DIR = Path("cache") / "metadata"


# ---------------------------------------------------------------------------
# Polymarket data-api
# ---------------------------------------------------------------------------

def get_wallet_activity(
    wallet: str,
    *,
    types: tuple[str, ...] = ("TRADE", "REDEEM"),
    page_size: int = 500,
    max_rows: int = DATA_API_ACTIVITY_CAP,
    timeout: float = 30.0,
    verbose: bool = False,
    cache_dir: Path | str | None = DEFAULT_CACHE_DIR,
) -> pd.DataFrame:
    """Pull all of a wallet's activity rows from the Polymarket data-api.

    The data-api ``/activity`` endpoint returns TRADE and REDEEM rows
    newest-first. This wrapper paginates until either the endpoint runs
    out, ``max_rows`` is hit, or pagination plateaus.

    Args:
        wallet: 0x-prefixed wallet address (case-insensitive).
        types: Activity types to keep. Default ``("TRADE", "REDEEM")``.
        page_size: Per-request page size (capped to 1000 by the API).
        max_rows: Stop paginating once this many ROWS have been kept.
        timeout: HTTP timeout per request.
        verbose: If True, print pagination progress.
        cache_dir: Directory for the per-wallet parquet cache. Pass None
            to disable caching entirely.

    Returns:
        DataFrame sorted ascending by ``timestamp``. Columns include
        ``timestamp``, ``type``, ``slug``, ``side``, ``size``,
        ``usdcSize``, ``outcomeIndex``, ``transactionHash``, ``asset``.
        Empty if the wallet has no matching activity.
    """
    wallet = wallet.lower()
    page_size = max(1, min(page_size, 1000))

    cache_path: Path | None = None
    if cache_dir is not None:
        cache_path = Path(cache_dir) / f"activity_{wallet}.parquet"
        if cache_path.exists():
            return pd.read_parquet(cache_path)
        cache_path.parent.mkdir(parents=True, exist_ok=True)

    rows: list[dict] = []
    offset = 0
    while len(rows) < max_rows and offset < DATA_API_MAX_OFFSET:
        remaining = DATA_API_MAX_OFFSET - offset
        this_limit = min(page_size, remaining)
        resp = requests.get(
            f"{DATA_API_BASE}/activity",
            params={"user": wallet, "limit": this_limit, "offset": offset},
            timeout=timeout,
        )
        if resp.status_code == 400:
            # Past the offset cap; data-api returns 400.
            if verbose:
                print(f"[activity] stopped at offset={offset}: HTTP 400")
            break
        resp.raise_for_status()
        page = resp.json()
        if isinstance(page, dict) and "error" in page:
            if verbose:
                print(f"[activity] stopped at offset={offset}: {page['error']}")
            break
        if not page:
            break
        for row in page:
            if row.get("type") in types:
                rows.append(row)
        if verbose:
            print(f"[activity] offset={offset}, page={len(page)}, kept={len(rows)}")
        offset += len(page)
        if len(page) < this_limit:
            break

    df = pd.DataFrame(rows)
    if not df.empty:
        df["timestamp"] = pd.to_numeric(df["timestamp"], errors="coerce")
        df["usdcSize"] = pd.to_numeric(df["usdcSize"], errors="coerce")
        df["size"] = pd.to_numeric(df["size"], errors="coerce")
        df["price"] = pd.to_numeric(df["price"], errors="coerce")
        df = (
            df.dropna(subset=["timestamp"])
            .sort_values("timestamp")
            .reset_index(drop=True)
        )

    if cache_path is not None:
        try:
            df.to_parquet(cache_path, index=False)
        except Exception:
            pass  # caching is best-effort; never fail the caller

    return df


# ---------------------------------------------------------------------------
# Polygonscan
# ---------------------------------------------------------------------------

def _polygonscan_get(api_key: str, **params) -> dict:
    """Call the Polygon explorer via the Etherscan v2 endpoint.

    Polygonscan's standalone v1 API was retired in 2025; the same
    API key now works against Etherscan v2 with ``chainid=137``.
    Handles a single retry on rate-limit.
    """
    params["apikey"] = api_key
    params["chainid"] = POLYGON_CHAIN_ID
    for attempt in range(2):
        resp = requests.get(ETHERSCAN_V2_BASE, params=params, timeout=30.0)
        resp.raise_for_status()
        payload = resp.json()
        if (
            payload.get("status") == "0"
            and "rate limit" in str(payload.get("result", "")).lower()
            and attempt == 0
        ):
            time.sleep(1.2)
            continue
        return payload
    return payload


def _resolve_api_key(api_key: str | None) -> str | None:
    if api_key:
        return api_key
    return os.environ.get("POLYGONSCAN_API_KEY") or os.environ.get(
        "ETHERSCAN_API_KEY"
    )


def get_wallet_first_usdc_deposit(
    wallet: str,
    *,
    api_key: str | None = None,
    cache_dir: Path | str | None = DEFAULT_CACHE_DIR,
) -> dict | None:
    """Return ``{"from": funder, "timestamp": ts, "tx_hash": hash}``.

    The "first USDC deposit" is the earliest tokentx Transfer where
    ``to == wallet`` for the USDC contract on Polygon, sorted ascending.

    Args:
        wallet: Wallet address.
        api_key: Polygonscan API key. Falls back to ``POLYGONSCAN_API_KEY``
            (or ``ETHERSCAN_API_KEY``) env var. Returns None if neither
            is set.
        cache_dir: Directory for the per-wallet JSON cache.

    Returns:
        The dict described above, or None if the wallet has no incoming
        USDC, or if no API key is available.
    """
    wallet = wallet.lower()
    key = _resolve_api_key(api_key)
    if not key:
        return None

    cache_path: Path | None = None
    if cache_dir is not None:
        cache_path = Path(cache_dir) / f"deposit_{wallet}.json"
        if cache_path.exists():
            with open(cache_path) as f:
                cached = json.load(f)
            return cached or None
        cache_path.parent.mkdir(parents=True, exist_ok=True)

    payload = _polygonscan_get(
        key,
        module="account",
        action="tokentx",
        contractaddress=USDC_POLYGON,
        address=wallet,
        startblock=0,
        endblock=99999999,
        page=1,
        offset=100,
        sort="asc",
    )
    result = payload.get("result", [])
    out: dict | None = None
    if isinstance(result, list):
        for tx in result:
            if str(tx.get("to", "")).lower() == wallet:
                out = {
                    "from": str(tx["from"]).lower(),
                    "timestamp": int(tx["timeStamp"]),
                    "tx_hash": tx["hash"],
                }
                break

    if cache_path is not None:
        try:
            with open(cache_path, "w") as f:
                json.dump(out or {}, f)
        except Exception:
            pass

    return out


def get_funder_outgoing_wallets(
    funder: str,
    *,
    as_of_ts: int | None = None,
    api_key: str | None = None,
    cache_dir: Path | str | None = DEFAULT_CACHE_DIR,
    cap: int = FUNDER_SIBLING_CAP,
) -> list[str] | None:
    """List distinct USDC recipients of outgoing transfers from ``funder``.

    These are candidate sibling proxies for the human behind ``funder``.

    Args:
        funder: Funder EOA address.
        as_of_ts: If set, only count recipients whose first incoming
            transfer from this funder happened at or before this unix
            timestamp. This is what enforces the leakage cutoff at the
            funder graph level.
        api_key: Polygonscan API key (falls back to ``POLYGONSCAN_API_KEY``
            or ``ETHERSCAN_API_KEY`` env var).
        cache_dir: Directory for the per-funder JSON cache. The cache
            stores all transfers up to "now"; filtering by ``as_of_ts``
            is applied after reading from cache.
        cap: If more than ``cap`` distinct recipients are seen, return
            None — the funder is treated as a CEX/relayer and the
            sibling-wallet features should be NaN.

    Returns:
        Sorted list of lowercased recipient addresses, or None if the
        funder hit the cap or no API key was available.
    """
    funder = funder.lower()
    key = _resolve_api_key(api_key)
    if not key or not funder:
        return None

    cache_path: Path | None = None
    transfers: list[dict] | None = None
    if cache_dir is not None:
        cache_path = Path(cache_dir) / f"funder_{funder}.json"
        if cache_path.exists():
            with open(cache_path) as f:
                payload = json.load(f)
            if payload.get("hit_cap"):
                return None
            transfers = payload.get("transfers", [])
        cache_path.parent.mkdir(parents=True, exist_ok=True)

    if transfers is None:
        transfers = []
        page = 1
        per_page = 1000
        hit_cap = False
        seen_recipients: set[str] = set()
        while True:
            payload = _polygonscan_get(
                key,
                module="account",
                action="tokentx",
                contractaddress=USDC_POLYGON,
                address=funder,
                startblock=0,
                endblock=99999999,
                page=page,
                offset=per_page,
                sort="asc",
            )
            result = payload.get("result", [])
            if not isinstance(result, list) or not result:
                break
            for tx in result:
                if str(tx.get("from", "")).lower() != funder:
                    continue
                to = str(tx.get("to", "")).lower()
                if not to:
                    continue
                ts = int(tx["timeStamp"])
                if to not in seen_recipients:
                    seen_recipients.add(to)
                    transfers.append({"to": to, "ts": ts})
                if len(seen_recipients) > cap:
                    hit_cap = True
                    break
            if hit_cap or len(result) < per_page:
                break
            page += 1
            if page > 10:  # absolute safety cap (10k transfers)
                break

        if cache_path is not None:
            try:
                with open(cache_path, "w") as f:
                    json.dump(
                        {"transfers": transfers, "hit_cap": hit_cap}, f
                    )
            except Exception:
                pass
        if hit_cap:
            return None

    if as_of_ts is not None:
        recipients = sorted({t["to"] for t in transfers if t["ts"] <= as_of_ts})
    else:
        recipients = sorted({t["to"] for t in transfers})
    return recipients


# ---------------------------------------------------------------------------
# Per-market PnL (net cash flow)
# ---------------------------------------------------------------------------

def _per_market_pnl(activity: pd.DataFrame) -> pd.Series:
    """Net cash-flow PnL per market, summed over the given activity rows.

    pnl_market = sum(SELL usdcSize) + sum(REDEEM usdcSize)
                 - sum(BUY usdcSize)

    Args:
        activity: Activity rows already filtered to ``timestamp <= as_of_ts``.

    Returns:
        Series indexed by ``slug`` with the per-market PnL in USDC.
    """
    if activity.empty:
        return pd.Series(dtype=float)
    side = activity["side"].fillna("").str.upper()
    type_ = activity["type"].fillna("")
    signed = activity["usdcSize"].astype(float).copy()
    signed[(type_ == "TRADE") & (side == "BUY")] *= -1
    signed[(type_ == "TRADE") & (~side.isin(["BUY", "SELL"]))] = 0
    # REDEEM keeps its + sign (payout)
    return signed.groupby(activity["slug"]).sum()


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------

METADATA_FEATURE_COLUMNS: tuple[str, ...] = (
    "n_markets_traded",
    "wallet_age_seconds",
    "market_age_seconds",
    "wallet_to_market_age_ratio",
    "wallet_minus_market_age_seconds",
    "winrate",
    "avg_bet_size_usdc",
    "median_bet_size_usdc",
    "total_bet_size_usdc",
    "n_wallets_same_funder",
    "n_other_wallets_same_funder",
    "herfindahl_index_markets",
    "is_first_market",
    "time_since_first_deposit_to_first_trade_seconds",
    "lifetime_volume_share_on_market",
    "pnl_sharpe",
    "funder_flagged_wallet_count",
)


def compute_metadata_features(
    wallet: str,
    market_slug: str,
    market_start_ts: int,
    *,
    as_of_ts: int | None = None,
    activity: pd.DataFrame | None = None,
    api_key: str | None = None,
    flagged_wallets: Iterable[str] | None = None,
    cache_dir: Path | str | None = DEFAULT_CACHE_DIR,
) -> dict:
    """Compute the metadata feature vector for one (wallet, market) pair.

    The as-of time is the wallet's first TRADE on ``market_slug``. Pass
    it explicitly via ``as_of_ts`` (recommended — read from the cached
    market pickle's ``parsed['wallet_trades']`` so the value is exact
    even for high-volume wallets whose data-api activity gets truncated
    at the 3,000 offset cap). If omitted, the function falls back to
    deriving it from the wallet's data-api activity, which can fail
    when the on-market trades are older than the cap.

    The returned dict always contains every key in
    :data:`METADATA_FEATURE_COLUMNS`, plus ``as_of_ts``, ``funder``, and
    ``first_deposit_ts`` for downstream debugging. Numerical features
    are floats (NaN when unknown); ``is_first_market`` is 0/1.

    Args:
        wallet: 0x-prefixed wallet address.
        market_slug: Market slug under evaluation.
        market_start_ts: Unix timestamp of the market's open (from
            :func:`polycluster.markets.get_market_by_slug`).
        as_of_ts: Wallet's first-trade timestamp on this market. If
            None, derived from the wallet's data-api activity.
        activity: Pre-fetched activity DataFrame for the wallet. If
            None, this function fetches it via
            :func:`get_wallet_activity`.
        api_key: Polygonscan API key. Falls back to the
            ``POLYGONSCAN_API_KEY`` (or ``ETHERSCAN_API_KEY``) env var.
            Funder-dependent features are NaN if no key is available.
        flagged_wallets: Iterable of wallet addresses that are known
            insiders. Used for ``funder_flagged_wallet_count``.
        cache_dir: Directory for per-wallet activity / funder caches.

    Returns:
        Dict with every column in :data:`METADATA_FEATURE_COLUMNS` plus
        ``as_of_ts``, ``funder``, ``first_deposit_ts``.

    Raises:
        ValueError: If ``as_of_ts`` is None AND the wallet has no TRADE
            activity on ``market_slug`` in its (truncated) data-api
            history.
    """
    wallet = wallet.lower()
    flagged_set = {w.lower() for w in (flagged_wallets or [])}

    if activity is None:
        activity = get_wallet_activity(wallet, cache_dir=cache_dir)

    out: dict = {col: float("nan") for col in METADATA_FEATURE_COLUMNS}
    out.update({"as_of_ts": None, "funder": None, "first_deposit_ts": None})

    if as_of_ts is None:
        if activity.empty:
            raise ValueError(f"wallet {wallet} has no activity on the data-api")
        on_market = activity[
            (activity["slug"] == market_slug) & (activity["type"] == "TRADE")
        ]
        if on_market.empty:
            raise ValueError(
                f"wallet {wallet} has no TRADE activity on {market_slug!r} "
                f"in its data-api activity (may be truncated at offset cap)"
            )
        as_of_ts = int(on_market["timestamp"].min())
    else:
        as_of_ts = int(as_of_ts)
    out["as_of_ts"] = as_of_ts

    if activity.empty or "timestamp" not in activity.columns:
        pre = activity.iloc[0:0]
    else:
        pre = activity[activity["timestamp"] <= as_of_ts]
    pre_trades = pre[pre["type"] == "TRADE"] if not pre.empty else pre

    markets_traded = set(pre_trades["slug"].dropna().unique()) if not pre_trades.empty else set()
    n_markets_traded = len(markets_traded)
    out["n_markets_traded"] = float(n_markets_traded)

    if not pre_trades.empty:
        bet_sizes = pre_trades["usdcSize"].dropna()
    else:
        bet_sizes = pd.Series(dtype=float)
    if not bet_sizes.empty:
        out["avg_bet_size_usdc"] = float(bet_sizes.mean())
        out["median_bet_size_usdc"] = float(bet_sizes.median())
        out["total_bet_size_usdc"] = float(bet_sizes.sum())
    else:
        out["avg_bet_size_usdc"] = 0.0
        out["median_bet_size_usdc"] = 0.0
        out["total_bet_size_usdc"] = 0.0

    if not pre_trades.empty:
        per_market_vol = pre_trades.groupby("slug")["usdcSize"].sum()
    else:
        per_market_vol = pd.Series(dtype=float)
    total_vol = float(per_market_vol.sum())
    if total_vol > 0:
        shares = (per_market_vol / total_vol).to_numpy()
        out["herfindahl_index_markets"] = float(np.sum(shares ** 2))
        out["lifetime_volume_share_on_market"] = float(
            per_market_vol.get(market_slug, 0.0) / total_vol
        )
    else:
        out["herfindahl_index_markets"] = 0.0
        out["lifetime_volume_share_on_market"] = 0.0

    pnls = _per_market_pnl(pre)
    if not pnls.empty:
        out["winrate"] = float((pnls > 0).mean())
        if len(pnls) >= 2:
            arr = pnls.to_numpy(dtype=float)
            sd = float(arr.std(ddof=0))
            out["pnl_sharpe"] = float(arr.mean() / sd) if sd > 0 else float("nan")

    out["is_first_market"] = float(
        n_markets_traded == 1 and market_slug in markets_traded
    )

    out["market_age_seconds"] = float(as_of_ts - int(market_start_ts))

    deposit = get_wallet_first_usdc_deposit(
        wallet, api_key=api_key, cache_dir=cache_dir
    )
    if deposit is not None:
        first_deposit_ts = deposit["timestamp"]
        funder = deposit["from"]
        wallet_age = float(as_of_ts - first_deposit_ts)
        out["first_deposit_ts"] = first_deposit_ts
        out["funder"] = funder
        out["wallet_age_seconds"] = wallet_age
        out["time_since_first_deposit_to_first_trade_seconds"] = wallet_age
        out["wallet_to_market_age_ratio"] = wallet_age / max(
            out["market_age_seconds"], 1.0
        )
        out["wallet_minus_market_age_seconds"] = (
            wallet_age - out["market_age_seconds"]
        )

        siblings = get_funder_outgoing_wallets(
            funder,
            as_of_ts=as_of_ts,
            api_key=api_key,
            cache_dir=cache_dir,
        )
        if siblings is not None:
            n_siblings = len(siblings)
            out["n_wallets_same_funder"] = float(n_siblings)
            out["n_other_wallets_same_funder"] = float(max(0, n_siblings - 1))
            out["funder_flagged_wallet_count"] = float(
                sum(1 for w in siblings if w in flagged_set)
            )

    return out


def build_metadata_feature_dataframe(
    pairs: Iterable[tuple[str, str]],
    *,
    market_start_ts_lookup,
    api_key: str | None = None,
    flagged_wallets: Iterable[str] | None = None,
    cache_dir: Path | str | None = DEFAULT_CACHE_DIR,
    verbose: bool = False,
) -> pd.DataFrame:
    """Compute metadata features for many (wallet, market) pairs.

    Args:
        pairs: Iterable of ``(wallet, market_slug)`` tuples.
        market_start_ts_lookup: Callable ``slug -> int`` returning the
            market's ``start_ts``. Caller controls how slugs are resolved
            (gamma-api, cache, etc.).
        api_key: Polygonscan API key (falls back to env var).
        flagged_wallets: Set of insider wallet addresses for the funder
            graph signal.
        cache_dir: Per-wallet / per-funder cache directory.
        verbose: Print progress per pair.

    Returns:
        DataFrame with one row per pair, columns: ``wallet``,
        ``market_slug``, plus every key produced by
        :func:`compute_metadata_features`.
    """
    rows: list[dict] = []
    flagged_list = list(flagged_wallets or [])
    for wallet, slug in pairs:
        try:
            mst = int(market_start_ts_lookup(slug))
            feats = compute_metadata_features(
                wallet,
                slug,
                mst,
                api_key=api_key,
                flagged_wallets=flagged_list,
                cache_dir=cache_dir,
            )
        except Exception as exc:
            if verbose:
                print(f"[metadata] FAILED ({wallet}, {slug}): {exc}")
            feats = {col: float("nan") for col in METADATA_FEATURE_COLUMNS}
            feats.update({"as_of_ts": None, "funder": None, "first_deposit_ts": None})
            feats["_error"] = str(exc)
        feats = {"wallet": wallet.lower(), "market_slug": slug, **feats}
        rows.append(feats)
        if verbose:
            print(f"[metadata] ({wallet}, {slug}) done")
    return pd.DataFrame(rows)
