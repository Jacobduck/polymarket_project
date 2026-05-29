"""Per-(wallet, market) feature extraction.

Given a wallet's parsed trades and the market's price history, this
module produces a dict of features describing the wallet's behavior on
that market: entry timing, holding pattern, realized/unrealized PnL,
markout edges at multiple horizons, contrarian price-taking, and
inventory/day concentration. The features are designed to surface the
signatures of informed/insider trading.

Top-level entry points:

* :func:`get_user_features` - high-level wrapper that takes a wallet
  address (or list) and a market slug/object, fetches everything, and
  returns the feature dict.
* :func:`compute_market_user_features` - the orchestrator. Takes
  already-parsed trades + an already-built history dataframe and runs
  every per-aspect feature function.
* The per-aspect feature functions
  (:func:`compute_entry_features`, :func:`compute_holding_features`,
  :func:`compute_unrealized_features`,
  :func:`compute_path_alignment_features`,
  :func:`compute_realized_pnl_features`,
  :func:`compute_markout_features`, :func:`compute_markout_features1`,
  :func:`compute_adaptive_markout_features`,
  :func:`compute_contrarian_price_features`,
  :func:`compute_extreme_inventory_and_day_features`) and the major-
  move detector (:func:`detect_major_move`).
* :func:`compute_realized_pnl_fifo` - FIFO realized-PnL bookkeeping
  with per-trade timeline output (also used internally by
  :func:`compute_unrealized_features`).
* :func:`build_market_volume_df_from_orderfilled` - convert raw
  OrderFilled events into a market-wide volume table with YES-side
  prices.
"""

from __future__ import annotations

from collections import deque
from typing import Any

import numpy as np
import pandas as pd

from polycluster.events import get_market_orderfilled_events
from polycluster.markets import Market, get_market_by_slug
from polycluster.parsing import (
    build_history_df_from_orderfilled,
    normalize_trades_to_yes_dicts,
    parse_orderfilled_events,
)

USDC_DECIMALS = 1_000_000
TOKEN_DECIMALS = 1_000_000


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _prep_trades(trades_df: pd.DataFrame | list[dict]) -> pd.DataFrame:
    """Coerce a wallet's normalized trades into the columns features need.

    Adds ``signed_token``, ``signed_cash`` and ``cum_inventory`` after
    sorting by timestamp. Accepts either a list of normalized trade
    dicts or an already-built DataFrame.
    """
    if isinstance(trades_df, list):
        df = pd.DataFrame(trades_df)
    else:
        df = trades_df.copy()

    if df.empty:
        for col in [
            "timestamp", "token_amount", "cash_amount", "norm_side",
            "norm_price", "signed_token", "signed_cash", "cum_inventory",
        ]:
            if col not in df.columns:
                df[col] = pd.Series(dtype=float)
        return df

    df["timestamp"] = pd.to_numeric(df["timestamp"], errors="coerce")
    df["token_amount"] = pd.to_numeric(df["token_amount"], errors="coerce")
    df["cash_amount"] = pd.to_numeric(df["cash_amount"], errors="coerce")

    df = df.dropna(subset=["timestamp"]).sort_values("timestamp").reset_index(drop=True)
    df["norm_side"] = df["norm_side"].astype(str).str.upper()

    df["signed_token"] = np.where(
        df["norm_side"] == "BUY", df["token_amount"], -df["token_amount"]
    )
    df["signed_cash"] = np.where(
        df["norm_side"] == "BUY", -df["cash_amount"], df["cash_amount"]
    )
    df["cum_inventory"] = df["signed_token"].cumsum()
    return df


def _prep_history(history_df: pd.DataFrame) -> pd.DataFrame:
    """Validate and clean a YES-price history dataframe."""
    df = history_df.copy()
    required = ["timestamp", "yes_price"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"history_df missing columns: {missing}")

    df["timestamp"] = pd.to_numeric(df["timestamp"], errors="coerce")
    df["yes_price"] = pd.to_numeric(df["yes_price"], errors="coerce")
    df = df.dropna(subset=["timestamp", "yes_price"]).sort_values("timestamp").reset_index(drop=True)
    return df


def _nearest_price_at_time(history_df: pd.DataFrame, ts: float) -> float:
    """YES price at the history row closest to ``ts`` (asof-nearest)."""
    hist = history_df[["timestamp", "yes_price"]].sort_values("timestamp").reset_index(drop=True)
    row = pd.merge_asof(
        pd.DataFrame({"timestamp": [ts]}),
        hist,
        on="timestamp",
        direction="nearest",
    )
    return row.loc[0, "yes_price"]


def _price_at_or_after(history_df: pd.DataFrame, ts: float) -> float:
    """YES price at the first history row with ``timestamp >= ts``."""
    hist = history_df.sort_values("timestamp").reset_index(drop=True)
    idx = hist["timestamp"].searchsorted(ts, side="left")
    if idx >= len(hist):
        return hist.iloc[-1]["yes_price"]
    return hist.iloc[idx]["yes_price"]


def _price_at_or_before(history_df: pd.DataFrame, ts: float) -> float:
    """YES price at the last history row with ``timestamp <= ts``."""
    hist = history_df.sort_values("timestamp").reset_index(drop=True)
    idx = hist["timestamp"].searchsorted(ts, side="right") - 1
    if idx < 0:
        return hist.iloc[0]["yes_price"]
    return hist.iloc[idx]["yes_price"]


def _safe_time_position(
    ts: float, market_start_time: float, market_end_time: float
) -> float:
    """Relative position of ``ts`` along the market lifetime in ``[0, 1]``."""
    if pd.isna(ts) or pd.isna(market_start_time) or pd.isna(market_end_time):
        return np.nan
    denom = market_end_time - market_start_time
    if denom <= 0:
        return np.nan
    return (ts - market_start_time) / denom


def _safe_time_remaining_pct(
    ts: float, market_start_time: float, market_end_time: float
) -> float:
    """Fraction of market lifetime remaining at ``ts``."""
    pos = _safe_time_position(ts, market_start_time, market_end_time)
    return np.nan if pd.isna(pos) else 1 - pos


def _safe_relative_gap(
    t1: float, t2: float, market_start_time: float, market_end_time: float
) -> float:
    """``(t2 - t1)`` as a fraction of total market lifetime."""
    if (
        pd.isna(t1) or pd.isna(t2)
        or pd.isna(market_start_time) or pd.isna(market_end_time)
    ):
        return np.nan
    denom = market_end_time - market_start_time
    if denom <= 0:
        return np.nan
    return (t2 - t1) / denom


def _rolling_price_slope(hist_window: pd.DataFrame) -> float:
    """Least-squares slope of ``yes_price`` vs ``timestamp`` on a window."""
    if len(hist_window) < 3:
        return np.nan
    x = hist_window["timestamp"].to_numpy(dtype=float)
    y = hist_window["yes_price"].to_numpy(dtype=float)
    x = x - x.mean()
    denom = np.sum(x ** 2)
    if denom == 0:
        return np.nan
    return float(np.sum(x * (y - y.mean())) / denom)


def _snapshot_lots(lots: deque) -> list[dict[str, float]]:
    """Shallow-copy the FIFO lot deque as a plain list of dicts."""
    return [{"qty": float(lot["qty"]), "price": float(lot["price"])} for lot in lots]


def _compute_unrealized_edge_over_gross_lifetime_tokens(
    final_open_lots: list[dict[str, float]],
    gross_tokens_lifetime: float,
    final_outcome_yes: int | float | None,
    history: pd.DataFrame,
    eps: float = 0.0,
) -> tuple[float, float]:
    """Unrealized edge from open lots, normalized by gross lifetime tokens.

    Returns ``(linear_edge, sqrt_edge)``. ``linear_edge`` weights each
    open lot's edge by its size; ``sqrt_edge`` uses ``sqrt(|qty|)`` to
    dampen large lots.
    """
    if final_outcome_yes is None:
        return (np.nan, np.nan)

    hist = history.copy()
    price_range = hist["yes_price"].max() - hist["yes_price"].min()
    settle_price = float(final_outcome_yes)

    numer = 0.0
    numer_sqrt = 0.0
    for lot in final_open_lots:
        qty = float(lot["qty"])
        entry_price = float(lot["price"])
        if abs(qty) <= eps:
            continue
        numer += qty * ((settle_price - entry_price) / price_range)
        edge = np.sign(qty) * (settle_price - entry_price)
        numer_sqrt += np.sqrt(abs(qty)) * (edge / price_range)

    if gross_tokens_lifetime is None or gross_tokens_lifetime <= eps:
        return (0.0, 0.0)

    return (
        numer / gross_tokens_lifetime,
        numer_sqrt / np.sqrt(abs(gross_tokens_lifetime)),
    )


def _find_terminal_spike_start(
    history_df: pd.DataFrame,
    yes_wins: bool,
    terminal_last_time_frac: float = 0.05,
    progress_threshold: float = 0.25,
    eps: float = 1e-9,
) -> float:
    """Find the start of the terminal price spike in outcome-aligned space.

    Looks only at the last ``terminal_last_time_frac`` of market life.
    Returns the first timestamp in that window where the aligned price
    has completed at least ``progress_threshold`` of the post-window-min
    move toward the final aligned price. Returns NaN if no such time
    exists or the price never moves up after its window minimum.
    """
    hist = history_df.sort_values("timestamp").reset_index(drop=True).copy()

    start_ts = float(hist["timestamp"].min())
    end_ts = float(hist["timestamp"].max())
    lifetime = end_ts - start_ts

    if lifetime <= 0 or len(hist) < 2:
        return np.nan

    terminal_window_start = end_ts - terminal_last_time_frac * lifetime

    if yes_wins:
        hist["_aligned_price"] = hist["yes_price"]
    else:
        hist["_aligned_price"] = 1.0 - hist["yes_price"]

    search = hist[hist["timestamp"] >= terminal_window_start].copy()
    if len(search) == 0:
        return np.nan

    min_idx = search["_aligned_price"].idxmin()
    min_ts = float(search.loc[min_idx, "timestamp"])

    search_after_min = search[search["timestamp"] >= min_ts].copy()

    aligned_min = float(search_after_min["_aligned_price"].iloc[0])
    aligned_final = float(hist.iloc[-1]["_aligned_price"])

    total_move = aligned_final - aligned_min
    if total_move <= eps:
        return np.nan

    search_after_min["_progress"] = (
        search_after_min["_aligned_price"] - aligned_min
    ) / total_move
    hit = search_after_min[search_after_min["_progress"] >= progress_threshold]
    return float(hit.iloc[0]["timestamp"]) if len(hit) > 0 else np.nan


def _build_fifo_lots_and_matches(
    trades_df: pd.DataFrame,
    side_col: str = "norm_side",
    price_col: str = "norm_price",
    token_col: str = "token_amount",
) -> tuple[list[dict], list[dict]]:
    """Walk normalized trades and build long/short FIFO lots + matches.

    BUYs first cover open short lots (FIFO), leftover opens a new long
    lot; SELLs first close open long lots (FIFO), leftover opens a new
    short lot. Each lot tracks ``original_tokens``, ``remaining_tokens``,
    ``closed_tokens`` and a list of ``close_events``. Returns
    ``(all_lots, realized_matches)``.
    """
    trades = trades_df.sort_values("timestamp").reset_index(drop=True).copy()

    long_lots: deque = deque()
    short_lots: deque = deque()
    all_lots: list[dict] = []
    realized_matches: list[dict] = []
    lot_id = 0

    for _, row in trades.iterrows():
        side = row[side_col]
        ts = float(row["timestamp"])
        price = float(row[price_col])
        tokens_left = float(row[token_col])
        if tokens_left <= 0:
            continue

        if side == "BUY":
            while tokens_left > 0 and len(short_lots) > 0:
                lot = short_lots[0]
                matched = min(tokens_left, lot["remaining_tokens"])
                raw_pnl = lot["entry_price"] - price

                realized_matches.append({
                    "entry_lot_id": lot["lot_id"],
                    "entry_side": "SELL",
                    "entry_sign": -1,
                    "entry_time": lot["entry_time"],
                    "entry_price": lot["entry_price"],
                    "exit_time": ts,
                    "exit_price": price,
                    "matched_tokens": matched,
                    "raw_realized_pnl_per_token": raw_pnl,
                })

                lot["remaining_tokens"] -= matched
                lot["closed_tokens"] += matched
                lot["close_events"].append({
                    "time": ts,
                    "price": price,
                    "tokens": matched,
                    "remaining_after": lot["remaining_tokens"],
                })
                tokens_left -= matched
                if lot["remaining_tokens"] <= 1e-12:
                    short_lots.popleft()

            if tokens_left > 0:
                new_lot = {
                    "lot_id": lot_id,
                    "side": "BUY",
                    "side_sign": +1,
                    "entry_time": ts,
                    "entry_price": price,
                    "original_tokens": tokens_left,
                    "remaining_tokens": tokens_left,
                    "closed_tokens": 0.0,
                    "close_events": [],
                }
                lot_id += 1
                long_lots.append(new_lot)
                all_lots.append(new_lot)

        elif side == "SELL":
            while tokens_left > 0 and len(long_lots) > 0:
                lot = long_lots[0]
                matched = min(tokens_left, lot["remaining_tokens"])
                raw_pnl = price - lot["entry_price"]

                realized_matches.append({
                    "entry_lot_id": lot["lot_id"],
                    "entry_side": "BUY",
                    "entry_sign": +1,
                    "entry_time": lot["entry_time"],
                    "entry_price": lot["entry_price"],
                    "exit_time": ts,
                    "exit_price": price,
                    "matched_tokens": matched,
                    "raw_realized_pnl_per_token": raw_pnl,
                })

                lot["remaining_tokens"] -= matched
                lot["closed_tokens"] += matched
                lot["close_events"].append({
                    "time": ts,
                    "price": price,
                    "tokens": matched,
                    "remaining_after": lot["remaining_tokens"],
                })
                tokens_left -= matched
                if lot["remaining_tokens"] <= 1e-12:
                    long_lots.popleft()

            if tokens_left > 0:
                new_lot = {
                    "lot_id": lot_id,
                    "side": "SELL",
                    "side_sign": -1,
                    "entry_time": ts,
                    "entry_price": price,
                    "original_tokens": tokens_left,
                    "remaining_tokens": tokens_left,
                    "closed_tokens": 0.0,
                    "close_events": [],
                }
                lot_id += 1
                short_lots.append(new_lot)
                all_lots.append(new_lot)

    return all_lots, realized_matches


def _find_lot_derisk_time(lot: dict, threshold: float) -> float:
    """First timestamp at which a lot is reduced by >= ``threshold`` of original tokens."""
    original = float(lot["original_tokens"])
    if original <= 0:
        return np.nan
    cum_closed = 0.0
    for ev in lot["close_events"]:
        cum_closed += float(ev["tokens"])
        if cum_closed >= threshold * original - 1e-12:
            return float(ev["time"])
    return np.nan


# ---------------------------------------------------------------------------
# Major-move detection
# ---------------------------------------------------------------------------

def detect_major_move(
    history_df: pd.DataFrame,
    lookahead_sec: int = 3600,
    abs_move_threshold: float = 0.15,
    upward_only: bool = False,
) -> dict[str, float]:
    """Find the earliest timestamp where the future move exceeds threshold.

    For each row, computes ``future_price`` at ``ts + lookahead_sec``
    (using :func:`_price_at_or_after`) and ``delta = future - current``.
    Returns the first row whose ``|delta|`` (or ``delta`` if
    ``upward_only``) meets ``abs_move_threshold``. All four fields are
    NaN if no row qualifies.

    Args:
        history_df: YES-price history with ``timestamp``, ``yes_price``.
        lookahead_sec: Forward horizon in seconds.
        abs_move_threshold: Minimum absolute (or signed) price move.
        upward_only: If True, only positive moves count.

    Returns:
        Dict with ``major_move_time``, ``major_move_start_price``,
        ``major_move_future_price``, ``major_move_delta``.
    """
    hist = _prep_history(history_df).copy()
    future_ts = hist["timestamp"] + lookahead_sec
    hist["future_price"] = [_price_at_or_after(hist, ts) for ts in future_ts]
    hist["delta"] = hist["future_price"] - hist["yes_price"]

    mask = (
        hist["delta"] >= abs_move_threshold
        if upward_only
        else hist["delta"].abs() >= abs_move_threshold
    )

    if not mask.any():
        return {
            "major_move_time": np.nan,
            "major_move_start_price": np.nan,
            "major_move_future_price": np.nan,
            "major_move_delta": np.nan,
        }

    row = hist.loc[mask].iloc[0]
    return {
        "major_move_time": row["timestamp"],
        "major_move_start_price": row["yes_price"],
        "major_move_future_price": row["future_price"],
        "major_move_delta": row["delta"],
    }


# ---------------------------------------------------------------------------
# Entry-phase features
# ---------------------------------------------------------------------------

def compute_entry_features(
    trades_df: pd.DataFrame,
    history_df: pd.DataFrame,
    market_start_time: float,
    market_end_time: float,
    major_move_time: float | None = None,
    pre_entry_quiet_window_sec: int = 6 * 3600,
    entry_side: str = "BUY",
) -> dict[str, float]:
    """Features describing how/when the wallet first entered.

    Computes entry-price percentile vs market history, position of first
    entry along market lifetime, gap to major move, hourly HHI of
    entry cash, entry span, and pre-entry price activity (std and mean
    abs return) over a quiet-window before the first entry.
    """
    trades = _prep_trades(trades_df)
    hist = _prep_history(history_df)

    entry_trades = trades[trades["norm_side"] == entry_side].copy()
    if len(entry_trades) == 0:
        return {
            "entry_price_percentile": np.nan,
            "first_entry_time_position_pct": np.nan,
            "first_entry_time_remaining_pct": np.nan,
            "time_to_major_move_pct_of_market": np.nan,
            "entry_concentration_hhi": np.nan,
            "entry_span_pct_of_market": np.nan,
            "pre_entry_price_std": np.nan,
            "pre_entry_abs_return_mean": np.nan,
        }

    first_entry_time = entry_trades["timestamp"].min()
    entry_prices = entry_trades["norm_price"].dropna()
    avg_entry_price = (
        entry_prices.mean()
        if len(entry_prices) > 0
        else _nearest_price_at_time(hist, first_entry_time)
    )
    entry_price_percentile = (hist["yes_price"] <= avg_entry_price).mean()

    first_entry_time_position_pct = _safe_time_position(
        first_entry_time, market_start_time, market_end_time
    )
    first_entry_time_remaining_pct = _safe_time_remaining_pct(
        first_entry_time, market_start_time, market_end_time
    )

    if major_move_time is None or pd.isna(major_move_time):
        time_to_major_move_pct_of_market = np.nan
    else:
        time_to_major_move_pct_of_market = _safe_relative_gap(
            first_entry_time, major_move_time, market_start_time, market_end_time
        )

    temp = entry_trades.copy()
    rel_sec = temp["timestamp"] - first_entry_time
    temp["bin"] = (rel_sec // 3600).astype(int)
    cash_by_bin = temp.groupby("bin")["cash_amount"].sum()
    if cash_by_bin.sum() > 0:
        shares = cash_by_bin / cash_by_bin.sum()
        entry_concentration_hhi = float((shares ** 2).sum())
    else:
        entry_concentration_hhi = np.nan

    entry_span_sec = entry_trades["timestamp"].max() - entry_trades["timestamp"].min()
    market_duration = market_end_time - market_start_time
    entry_span_pct_of_market = (
        entry_span_sec / market_duration if market_duration > 0 else np.nan
    )

    win = hist[
        (hist["timestamp"] >= first_entry_time - pre_entry_quiet_window_sec)
        & (hist["timestamp"] <= first_entry_time)
    ].copy()
    if len(win) >= 2:
        win["ret"] = win["yes_price"].diff()
        pre_entry_price_std = win["yes_price"].std()
        pre_entry_abs_return_mean = win["ret"].abs().mean()
    else:
        pre_entry_price_std = np.nan
        pre_entry_abs_return_mean = np.nan

    return {
        "entry_price_percentile": entry_price_percentile,
        "first_entry_time_position_pct": first_entry_time_position_pct,
        "first_entry_time_remaining_pct": first_entry_time_remaining_pct,
        "time_to_major_move_pct_of_market": time_to_major_move_pct_of_market,
        "entry_concentration_hhi": entry_concentration_hhi,
        "entry_span_pct_of_market": entry_span_pct_of_market,
        "pre_entry_price_std": pre_entry_price_std,
        "pre_entry_abs_return_mean": pre_entry_abs_return_mean,
    }


# ---------------------------------------------------------------------------
# Holding / conviction features
# ---------------------------------------------------------------------------

def compute_holding_features(
    trades_df: pd.DataFrame,
    final_outcome_yes: int | float | None = None,
) -> dict[str, float]:
    """Inventory aggregates and outcome-alignment of the net position.

    Returns directional consistency (net/gross), buy/sell/net/gross
    token totals, max/min/max-abs inventory, and outcome alignment of
    the net position (sign-only and token-share variants). Alignment
    fields are NaN when ``final_outcome_yes`` is None.
    """
    trades = _prep_trades(trades_df)

    buy_tokens = trades.loc[trades["norm_side"] == "BUY", "token_amount"].sum()
    sell_tokens = trades.loc[trades["norm_side"] == "SELL", "token_amount"].sum()
    net_tokens = buy_tokens - sell_tokens
    gross_tokens = buy_tokens + sell_tokens

    net_directional_consistency = (
        net_tokens / gross_tokens if gross_tokens > 0 else np.nan
    )
    max_inventory = trades["cum_inventory"].max() if len(trades) > 0 else np.nan
    min_inventory = trades["cum_inventory"].min() if len(trades) > 0 else np.nan
    max_abs_inventory = (
        trades["cum_inventory"].abs().max() if len(trades) > 0 else np.nan
    )

    if final_outcome_yes is None:
        outcome_alignment = np.nan
        outcome_alignment_token = np.nan
    else:
        correct_sign = 1 if int(final_outcome_yes) == 1 else -1
        outcome_alignment = float(np.sign(net_tokens) * correct_sign)
        outcome_alignment_token = (
            (net_tokens * correct_sign) / gross_tokens
            if gross_tokens > 0
            else np.nan
        )

    return {
        "net_directional_consistency": net_directional_consistency,
        "buy_tokens_total": buy_tokens,
        "sell_tokens_total": sell_tokens,
        "net_tokens": net_tokens,
        "gross_tokens": gross_tokens,
        "max_inventory": max_inventory,
        "min_inventory": min_inventory,
        "max_abs_inventory": max_abs_inventory,
        "outcome_alignment": outcome_alignment,
        "outcome_alignment_token": outcome_alignment_token,
    }


# ---------------------------------------------------------------------------
# Unrealized features
# ---------------------------------------------------------------------------

def compute_unrealized_features(
    parsed_trades: list[dict],
    final_outcome_yes: int | float | None,
    history: pd.DataFrame,
) -> dict[str, float]:
    """Unrealized-edge features computed from final FIFO open lots.

    Runs FIFO PnL bookkeeping (:func:`compute_realized_pnl_fifo`),
    takes the residual open lots, and computes
    :func:`_compute_unrealized_edge_over_gross_lifetime_tokens` against
    ``final_outcome_yes`` as the settlement price. Returns linear and
    sqrt-weighted unrealized edges plus the open-position quantity.
    """
    fifo_res = compute_realized_pnl_fifo(parsed_trades)
    yes_dicts = normalize_trades_to_yes_dicts(parsed_trades)
    trades_df = pd.DataFrame(yes_dicts) if yes_dicts else pd.DataFrame()
    trades = _prep_trades(trades_df)

    buy_tokens = trades.loc[trades["norm_side"] == "BUY", "token_amount"].sum()
    sell_tokens = trades.loc[trades["norm_side"] == "SELL", "token_amount"].sum()
    gross_tokens = buy_tokens + sell_tokens

    edge_linear, edge_sqrt = _compute_unrealized_edge_over_gross_lifetime_tokens(
        final_open_lots=fifo_res["final_open_lots"],
        final_outcome_yes=final_outcome_yes,
        gross_tokens_lifetime=gross_tokens,
        history=history,
    )

    return {
        "unrealized_edge_per_open_token": edge_linear,
        "unrealized_edge_per_open_token_sqrt": edge_sqrt,
        "open_position_qty": fifo_res["position_qty"],
    }


# ---------------------------------------------------------------------------
# Path-alignment features
# ---------------------------------------------------------------------------

def compute_path_alignment_features(
    trades_df: pd.DataFrame,
    final_outcome_yes: int | float | None = None,
    market_start_time: float | None = None,
    market_end_time: float | None = None,
    early_time_frac: float = 0.5,
) -> dict[str, float]:
    """Outcome alignment of the position path (cumulative inventory).

    Computes ``sum(position_t * correct_sign) / sum(|position_t|)``
    over all trades, plus an early-time variant restricted to
    ``[market_start, market_start + early_time_frac * lifetime]``.
    Also returns avg/max abs position, final position, and the count
    and rate of inventory sign flips.
    """
    trades = _prep_trades(trades_df)
    out: dict[str, float] = {
        "path_alignment": np.nan,
        "early_path_alignment": np.nan,
        "avg_abs_position_tokens": np.nan,
        "max_abs_position_tokens": np.nan,
        "final_position_tokens": np.nan,
        "position_flips": np.nan,
        "position_flip_rate": np.nan,
    }

    if len(trades) == 0 or final_outcome_yes is None:
        return out

    correct_sign = 1 if bool(final_outcome_yes) else -1
    df = trades.sort_values("timestamp").copy()
    df["position_tokens"] = df["signed_token"].cumsum()

    pos = df["position_tokens"].to_numpy(dtype=float)
    abs_pos = np.abs(pos)
    denom = abs_pos.sum()
    if denom > 0:
        out["path_alignment"] = float(np.sum(pos * correct_sign) / denom)

    if market_start_time is not None and market_end_time is not None:
        early_cutoff = (
            market_start_time + early_time_frac * (market_end_time - market_start_time)
        )
        early_df = df[df["timestamp"] <= early_cutoff].copy()
        if len(early_df) > 0:
            early_pos = early_df["position_tokens"].to_numpy(dtype=float)
            early_abs = np.abs(early_pos)
            early_denom = early_abs.sum()
            if early_denom > 0:
                out["early_path_alignment"] = float(
                    np.sum(early_pos * correct_sign) / early_denom
                )

    out["avg_abs_position_tokens"] = float(abs_pos.mean()) if len(abs_pos) > 0 else np.nan
    out["max_abs_position_tokens"] = float(abs_pos.max()) if len(abs_pos) > 0 else np.nan
    out["final_position_tokens"] = float(pos[-1]) if len(pos) > 0 else np.nan

    pos_sign = np.sign(pos)
    pos_sign_nz = pos_sign[pos_sign != 0]
    if len(pos_sign_nz) >= 2:
        flips = int(np.sum(pos_sign_nz[1:] != pos_sign_nz[:-1]))
        out["position_flips"] = flips
        out["position_flip_rate"] = flips / (len(pos_sign_nz) - 1)
    else:
        out["position_flips"] = 0.0
        out["position_flip_rate"] = 0.0

    return out


# ---------------------------------------------------------------------------
# Realized PnL features
# ---------------------------------------------------------------------------

def compute_realized_pnl_features(
    trades_df: pd.DataFrame,
    market_start_time: float | None = None,
    market_end_time: float | None = None,
    early_time_frac: float = 0.5,
) -> dict[str, float]:
    """Symmetric FIFO realized PnL in YES-space, overall and early-period.

    For each trade, closes prior opposite-side inventory FIFO and
    records realized PnL piecewise. Returns total realized PnL, per-
    closing-trade and per-matched-token averages, count of closing
    trades, total matched tokens, and win-rate of closing trades. An
    "early" copy of every metric is computed over trades up to
    ``market_start + early_time_frac * lifetime`` if both bounds are
    provided.
    """
    trades = _prep_trades(trades_df)
    out: dict[str, float] = {
        "realized_pnl_total": np.nan,
        "realized_pnl_per_closing_trade": np.nan,
        "realized_pnl_per_matched_token": np.nan,
        "n_closing_trades_with_realization": np.nan,
        "matched_tokens_realized": np.nan,
        "realized_win_rate_closing_trades": np.nan,
        "early_realized_pnl_total": np.nan,
        "early_realized_pnl_per_closing_trade": np.nan,
        "early_realized_pnl_per_matched_token": np.nan,
        "early_n_closing_trades_with_realization": np.nan,
        "early_matched_tokens_realized": np.nan,
        "early_realized_win_rate_closing_trades": np.nan,
    }

    if len(trades) == 0:
        return out

    df = trades.sort_values("timestamp").copy()
    early_cutoff = (
        market_start_time + early_time_frac * (market_end_time - market_start_time)
        if market_start_time is not None and market_end_time is not None
        else None
    )

    open_long_lots: list[dict] = []
    open_short_lots: list[dict] = []

    realized_pnl_total = 0.0
    matched_tokens_total = 0.0
    closing_trade_pnls: list[float] = []

    early_realized_pnl_total = 0.0
    early_matched_tokens_total = 0.0
    early_closing_trade_pnls: list[float] = []

    for _, row in df.iterrows():
        side = row["norm_side"]
        px = float(row["norm_price"])
        qty = float(row["token_amount"])
        ts = row["timestamp"]
        if qty <= 0 or pd.isna(px):
            continue

        this_trade_realized = 0.0
        this_trade_matched = 0.0
        qty_left = qty

        if side == "BUY":
            while qty_left > 0 and len(open_short_lots) > 0:
                lot = open_short_lots[0]
                matched = min(qty_left, lot["remaining"])
                pnl_piece = (lot["norm_price"] - px) * matched
                this_trade_realized += pnl_piece
                this_trade_matched += matched
                realized_pnl_total += pnl_piece
                matched_tokens_total += matched
                lot["remaining"] -= matched
                qty_left -= matched
                if lot["remaining"] <= 1e-12:
                    open_short_lots.pop(0)
            if qty_left > 0:
                open_long_lots.append(
                    {"remaining": qty_left, "norm_price": px, "timestamp": ts}
                )
        elif side == "SELL":
            while qty_left > 0 and len(open_long_lots) > 0:
                lot = open_long_lots[0]
                matched = min(qty_left, lot["remaining"])
                pnl_piece = (px - lot["norm_price"]) * matched
                this_trade_realized += pnl_piece
                this_trade_matched += matched
                realized_pnl_total += pnl_piece
                matched_tokens_total += matched
                lot["remaining"] -= matched
                qty_left -= matched
                if lot["remaining"] <= 1e-12:
                    open_long_lots.pop(0)
            if qty_left > 0:
                open_short_lots.append(
                    {"remaining": qty_left, "norm_price": px, "timestamp": ts}
                )

        if this_trade_matched > 0:
            closing_trade_pnls.append(this_trade_realized)
            if early_cutoff is not None and ts <= early_cutoff:
                early_realized_pnl_total += this_trade_realized
                early_matched_tokens_total += this_trade_matched
                early_closing_trade_pnls.append(this_trade_realized)

    n_closing = len(closing_trade_pnls)
    if n_closing > 0:
        out["realized_pnl_total"] = realized_pnl_total
        out["realized_pnl_per_closing_trade"] = realized_pnl_total / n_closing
        out["n_closing_trades_with_realization"] = n_closing
        out["realized_win_rate_closing_trades"] = float(
            np.mean(np.array(closing_trade_pnls) > 0)
        )
    if matched_tokens_total > 0:
        out["matched_tokens_realized"] = matched_tokens_total
        out["realized_pnl_per_matched_token"] = realized_pnl_total / matched_tokens_total

    early_n_closing = len(early_closing_trade_pnls)
    if early_n_closing > 0:
        out["early_realized_pnl_total"] = early_realized_pnl_total
        out["early_realized_pnl_per_closing_trade"] = (
            early_realized_pnl_total / early_n_closing
        )
        out["early_n_closing_trades_with_realization"] = early_n_closing
        out["early_realized_win_rate_closing_trades"] = float(
            np.mean(np.array(early_closing_trade_pnls) > 0)
        )
    if early_matched_tokens_total > 0:
        out["early_matched_tokens_realized"] = early_matched_tokens_total
        out["early_realized_pnl_per_matched_token"] = (
            early_realized_pnl_total / early_matched_tokens_total
        )

    return out


# ---------------------------------------------------------------------------
# FIFO realized PnL (full timeline output)
# ---------------------------------------------------------------------------

def compute_realized_pnl_fifo(
    parsed_trades: list[dict],
    allow_short: bool = True,
    eps: float = 0.0,
) -> dict[str, Any]:
    """FIFO realized-PnL bookkeeping with full per-trade timeline.

    Normalizes raw parsed trades to YES-space, sorts by ``(timestamp,
    trade_id)``, and walks them: BUYs cover existing short lots first
    (FIFO), SELLs close existing long lots (FIFO). Returns realized
    PnL, current open lots, per-trade PnL/inventory rows, and a
    snapshot of open lots after each trade.

    Args:
        parsed_trades: Raw wallet trades from
            :func:`polycluster.parsing.parse_orderfilled_events`
            (i.e. ``parsed["wallet_trades"][address]``). Already
            decimal-scaled, with ``side``/``outcome``/``norm_*`` not yet
            present.
        allow_short: If False, raises when a SELL would exceed long
            inventory (i.e. require opening a short lot).
        eps: Numerical tolerance for treating qty as zero.

    Returns:
        Dict with ``realized_pnl``, ``position_qty``,
        ``position_cost_basis``, ``avg_entry_price``,
        ``normalized_trades``, ``final_open_lots``, ``trade_pnl``,
        ``pnl_timeline``, ``inventory_timeline`` and
        ``open_lots_after_each_trade``.

    Raises:
        ValueError: If ``allow_short`` is False and a SELL would open a
            short, or if any normalized side is not BUY/SELL.
    """
    trades = normalize_trades_to_yes_dicts(parsed_trades)

    lots: deque = deque()
    realized_pnl = 0.0
    trade_pnl: list[dict] = []
    pnl_timeline: list[dict] = []
    inventory_timeline: list[dict] = []
    open_lots_after_each_trade: list[dict] = []

    for tr in trades:
        qty = float(tr["token_amount"])
        px = float(tr["norm_price"])
        side = tr["norm_side"]

        pnl_this_trade = 0.0
        remaining = qty

        if side == "BUY":
            while remaining > eps and lots and lots[0]["qty"] < 0:
                short_lot = lots[0]
                cover_qty = min(remaining, -short_lot["qty"])
                pnl_piece = (short_lot["price"] - px) * cover_qty
                pnl_this_trade += pnl_piece
                realized_pnl += pnl_piece
                short_lot["qty"] += cover_qty
                remaining -= cover_qty
                if abs(short_lot["qty"]) <= eps:
                    lots.popleft()
            if remaining > eps:
                lots.append({"qty": remaining, "price": px})

        elif side == "SELL":
            while remaining > eps and lots and lots[0]["qty"] > 0:
                long_lot = lots[0]
                close_qty = min(remaining, long_lot["qty"])
                pnl_piece = (px - long_lot["price"]) * close_qty
                pnl_this_trade += pnl_piece
                realized_pnl += pnl_piece
                long_lot["qty"] -= close_qty
                remaining -= close_qty
                if long_lot["qty"] <= eps:
                    lots.popleft()
            if remaining > eps:
                if not allow_short:
                    raise ValueError(
                        f"Normalized SELL exceeds long inventory at "
                        f"trade_id={tr.get('trade_id')} timestamp={tr.get('timestamp')}. "
                        f"Set allow_short=True if short inventory is possible."
                    )
                lots.append({"qty": -remaining, "price": px})
        else:
            raise ValueError(f"Unexpected normalized side: {side}")

        position_qty = sum(lot["qty"] for lot in lots)
        position_cost_basis = sum(lot["qty"] * lot["price"] for lot in lots)
        avg_entry_price = (
            None if abs(position_qty) <= eps else position_cost_basis / position_qty
        )

        trade_pnl.append({
            "trade_id": tr.get("trade_id"),
            "timestamp": int(tr.get("timestamp")),
            "norm_side": side,
            "norm_price": px,
            "qty": qty,
            "realized_pnl_this_trade": pnl_this_trade,
            "cumulative_realized_pnl": realized_pnl,
            "position_qty_after_trade": position_qty,
            "avg_entry_price_after_trade": avg_entry_price,
        })
        pnl_timeline.append({
            "trade_id": tr.get("trade_id"),
            "timestamp": int(tr.get("timestamp")),
            "realized_pnl_this_trade": pnl_this_trade,
            "cumulative_realized_pnl": realized_pnl,
        })
        inventory_timeline.append({
            "trade_id": tr.get("trade_id"),
            "timestamp": int(tr.get("timestamp")),
            "position_qty": position_qty,
            "position_cost_basis": position_cost_basis,
            "avg_entry_price": avg_entry_price,
        })
        open_lots_after_each_trade.append({
            "trade_id": tr.get("trade_id"),
            "timestamp": int(tr.get("timestamp")),
            "open_lots": _snapshot_lots(lots),
        })

    final_position_qty = sum(lot["qty"] for lot in lots)
    final_position_cost_basis = sum(lot["qty"] * lot["price"] for lot in lots)
    final_avg_entry_price = (
        None
        if abs(final_position_qty) <= eps
        else final_position_cost_basis / final_position_qty
    )

    return {
        "realized_pnl": realized_pnl,
        "position_qty": final_position_qty,
        "position_cost_basis": final_position_cost_basis,
        "avg_entry_price": final_avg_entry_price,
        "normalized_trades": trades,
        "final_open_lots": list(lots),
        "trade_pnl": trade_pnl,
        "pnl_timeline": pnl_timeline,
        "inventory_timeline": inventory_timeline,
        "open_lots_after_each_trade": open_lots_after_each_trade,
    }


# ---------------------------------------------------------------------------
# Markout features
# ---------------------------------------------------------------------------

def compute_markout_features(
    trades_df: pd.DataFrame,
    history_df: pd.DataFrame,
    horizon_fracs: tuple[float, ...] = (0.01, 0.03, 0.07, 0.15, 0.20),
    lookback_fracs: tuple[float, ...] = (0.01, 0.03, 0.07, 0.15, 0.20),
    momentum_penalty_lambda: float = 0.10,
    contrarian_reward_lambda: float = 0.30,
    price_scale_alpha: float = 1.5,
    max_price_weight: float = 8.0,
    eps: float = 0.0,
    feature_filter: set[str] | None = None,
) -> dict[str, float]:
    """Adjusted markout in YES-space over a grid of horizons / lookbacks.

    Only trades whose ``inventory_before`` is in the same direction as
    the trade are considered (BUY with non-negative prior inventory,
    SELL with non-positive prior inventory). Each trade gets a score
    that combines the post-trade move with a momentum penalty / contrarian
    reward applied to the pre-trade trend, and is weighted by a
    nonlinear price weight (favoring extreme prices). For each
    ``(horizon, lookback)`` pair, returns mean/median over qualifying
    trades and a token-weighted version normalized by gross lifetime
    tokens.
    """
    trades = _prep_trades(trades_df)
    hist = _prep_history(history_df)

    out: dict[str, float] = {}
    if len(trades) == 0 or len(hist) < 2:
        return out

    market_lifetime = hist["timestamp"].max() - hist["timestamp"].min()
    if market_lifetime <= 0:
        return out

    trades = trades.copy()
    trades["inventory_before"] = trades["cum_inventory"] - trades["signed_token"]

    gross_tokens = trades["token_amount"].sum()
    if gross_tokens <= 0:
        gross_tokens = np.nan

    buy_trades = trades[
        (trades["norm_side"] == "BUY") & (trades["inventory_before"] >= 0)
    ].copy()
    sell_trades = trades[
        (trades["norm_side"] == "SELL") & (trades["inventory_before"] <= 0)
    ].copy()

    for h_frac in horizon_fracs:
        horizon_sec = h_frac * market_lifetime
        for lb_frac in lookback_fracs:
            lookback_sec = lb_frac * market_lifetime
            tag = f"h{h_frac:.3f}_lb{lb_frac:.3f}".replace(".", "p")

            if feature_filter is not None:
                pair_keys = {
                    f"buy_markout_adj_mean_{tag}",
                    f"buy_markout_adj_median_{tag}",
                    f"buy_markout_adj_grosswt_{tag}",
                    f"sell_markout_adj_mean_{tag}",
                    f"sell_markout_adj_median_{tag}",
                    f"sell_markout_adj_grosswt_{tag}",
                }
                if not (pair_keys & feature_filter):
                    continue

            buy_scores: list[float] = []
            buy_tokens: list[float] = []
            for _, row in buy_trades.iterrows():
                current_price = float(row["norm_price"])
                past_price = _price_at_or_before(hist, row["timestamp"] - lookback_sec)
                future_price = _price_at_or_after(hist, row["timestamp"] + horizon_sec)
                post_move = future_price - current_price
                pretrend = current_price - past_price
                pos_pretrend = max(pretrend, 0.0)
                neg_pretrend = max(-pretrend, 0.0)
                price_weight = 1.0 / (max(current_price, eps) ** price_scale_alpha)
                price_weight = min(price_weight, max_price_weight)
                score = (
                    price_weight * post_move
                    - momentum_penalty_lambda * pos_pretrend
                    + contrarian_reward_lambda * neg_pretrend
                )
                buy_scores.append(score)
                buy_tokens.append(float(row["token_amount"]))

            sell_scores: list[float] = []
            sell_tokens: list[float] = []
            for _, row in sell_trades.iterrows():
                current_price = float(row["norm_price"])
                past_price = _price_at_or_before(hist, row["timestamp"] - lookback_sec)
                future_price = _price_at_or_after(hist, row["timestamp"] + horizon_sec)
                post_move = current_price - future_price
                pretrend = current_price - past_price
                pos_pretrend = max(pretrend, 0.0)
                neg_pretrend = max(-pretrend, 0.0)
                price_weight = 1.0 / (max(1.0 - current_price, eps) ** price_scale_alpha)
                price_weight = min(price_weight, max_price_weight)
                score = (
                    price_weight * post_move
                    - momentum_penalty_lambda * neg_pretrend
                    + contrarian_reward_lambda * pos_pretrend
                )
                sell_scores.append(score)
                sell_tokens.append(float(row["token_amount"]))

            out[f"buy_markout_adj_mean_{tag}"] = (
                float(np.mean(buy_scores)) if buy_scores else np.nan
            )
            out[f"buy_markout_adj_median_{tag}"] = (
                float(np.median(buy_scores)) if buy_scores else np.nan
            )
            out[f"sell_markout_adj_mean_{tag}"] = (
                float(np.mean(sell_scores)) if sell_scores else np.nan
            )
            out[f"sell_markout_adj_median_{tag}"] = (
                float(np.median(sell_scores)) if sell_scores else np.nan
            )

            if buy_scores and pd.notna(gross_tokens):
                out[f"buy_markout_adj_grosswt_{tag}"] = float(
                    np.sum(np.array(buy_scores) * np.array(buy_tokens) / gross_tokens)
                )
            else:
                out[f"buy_markout_adj_grosswt_{tag}"] = (
                    0.0 if pd.notna(gross_tokens) else np.nan
                )
            if sell_scores and pd.notna(gross_tokens):
                out[f"sell_markout_adj_grosswt_{tag}"] = float(
                    np.sum(np.array(sell_scores) * np.array(sell_tokens) / gross_tokens)
                )
            else:
                out[f"sell_markout_adj_grosswt_{tag}"] = (
                    0.0 if pd.notna(gross_tokens) else np.nan
                )

    return out


def compute_markout_features1(
    trades_df: pd.DataFrame,
    history_df: pd.DataFrame,
    horizons_sec: tuple[int, ...] = (300, 3600, 21600, 86400),
) -> dict[str, float]:
    """Plain (unadjusted) markouts at fixed wall-clock horizons.

    For every BUY (resp. SELL) trade, computes ``future_price -
    current_price`` (resp. ``current_price - future_price``) at each
    horizon in ``horizons_sec``. Returns mean and median per side per
    horizon.
    """
    trades = _prep_trades(trades_df)
    hist = _prep_history(history_df)
    out: dict[str, float] = {}

    buy_trades = trades[trades["norm_side"] == "BUY"].copy()
    sell_trades = trades[trades["norm_side"] == "SELL"].copy()

    for h in horizons_sec:
        buy_markouts = [
            _price_at_or_after(hist, row["timestamp"] + h) - row["norm_price"]
            for _, row in buy_trades.iterrows()
        ]
        sell_markouts = [
            row["norm_price"] - _price_at_or_after(hist, row["timestamp"] + h)
            for _, row in sell_trades.iterrows()
        ]
        out[f"buy_markout_mean_{h}s"] = (
            float(np.mean(buy_markouts)) if buy_markouts else np.nan
        )
        out[f"buy_markout_median_{h}s"] = (
            float(np.median(buy_markouts)) if buy_markouts else np.nan
        )
        out[f"sell_markout_mean_{h}s"] = (
            float(np.mean(sell_markouts)) if sell_markouts else np.nan
        )
        out[f"sell_markout_median_{h}s"] = (
            float(np.median(sell_markouts)) if sell_markouts else np.nan
        )

    return out


def compute_adaptive_markout_features(
    trades_df: pd.DataFrame,
    history_df: pd.DataFrame,
    yes_wins: bool,
    horizon_fracs: tuple[float, ...] = (0.01, 0.03, 0.07, 0.15, 0.20),
    derisk_thresholds: tuple[float, ...] = (0.25, 0.375, 0.50, 0.75, 1.00),
    realized_alpha_grid: tuple[float, ...] = (0.25, 0.50, 1.00),
    late_spike_lambda: float = 1.0,
    terminal_last_time_frac: float = 0.05,
    terminal_progress_threshold: float = 0.25,
    side_col: str = "norm_side",
    price_col: str = "norm_price",
    token_col: str = "token_amount",
    eps: float = 0.0,
) -> dict[str, float]:
    """Adaptive lot-level markout features with terminal-spike adjustment.

    Builds FIFO lots, then for each open lot computes a markout where
    the future timestamp is ``entry + horizon_frac * lifetime``, but
    capped by per-lot derisk-25% and derisk-50% times so the markout
    isn't measured past the trader's own derisking. Markouts are
    normalized by the YES-price range in history. A separate
    "late-spike" component sums positive markouts for lots opened after
    the detected terminal-spike start, and the adjusted markout
    subtracts ``late_spike_lambda`` times that component. Realized edge
    (per match) is computed too and combined with the adjusted markout
    using a grid of mixing weights.
    """
    out: dict[str, float] = {}

    if trades_df is None or history_df is None:
        return out
    if len(trades_df) == 0 or len(history_df) < 2:
        return out

    trades = trades_df.sort_values("timestamp").reset_index(drop=True).copy()
    hist = history_df.sort_values("timestamp").reset_index(drop=True).copy()

    market_start_ts = float(hist["timestamp"].min())
    market_end_ts = float(hist["timestamp"].max())
    market_lifetime = market_end_ts - market_start_ts
    if market_lifetime <= 0:
        return out

    price_min = float(hist["yes_price"].min())
    price_max = float(hist["yes_price"].max())
    price_range = max(price_max - price_min, eps)

    gross_tokens = float(trades[token_col].sum())
    if gross_tokens <= 0:
        return out

    terminal_start_ts = _find_terminal_spike_start(
        history_df=hist,
        yes_wins=yes_wins,
        terminal_last_time_frac=terminal_last_time_frac,
        progress_threshold=terminal_progress_threshold,
        eps=eps if eps > 0 else 1e-9,
    )

    out["terminal_start_ts"] = terminal_start_ts
    out["terminal_start_time_position"] = (
        (terminal_start_ts - market_start_ts) / market_lifetime
        if pd.notna(terminal_start_ts)
        else np.nan
    )

    lots, realized_matches = _build_fifo_lots_and_matches(
        trades_df=trades,
        side_col=side_col,
        price_col=price_col,
        token_col=token_col,
    )

    realized_edge_grosswt = 0.0
    realized_raw_pnl_grosswt = 0.0
    realized_token_sum = 0.0
    for m in realized_matches:
        tokens = float(m["matched_tokens"])
        raw_pnl = float(m["raw_realized_pnl_per_token"])
        normalized_edge = raw_pnl / price_range
        realized_edge_grosswt += normalized_edge * tokens / gross_tokens
        realized_raw_pnl_grosswt += raw_pnl * tokens / gross_tokens
        realized_token_sum += tokens

    out["realized_edge_grosswt"] = realized_edge_grosswt
    out["realized_raw_pnl_grosswt"] = realized_raw_pnl_grosswt
    out["realized_token_frac"] = realized_token_sum / gross_tokens

    for thr in derisk_thresholds:
        times: list[float] = []
        weights: list[float] = []
        for lot in lots:
            t = _find_lot_derisk_time(lot, thr)
            if pd.notna(t):
                times.append((t - market_start_ts) / market_lifetime)
                weights.append(float(lot["original_tokens"]))

        thr_tag = str(thr).replace(".", "p")
        if len(times) > 0:
            t_arr = np.array(times, dtype=float)
            w_arr = np.array(weights, dtype=float)
            out[f"derisk_{thr_tag}_time_position_mean"] = float(np.mean(t_arr))
            out[f"derisk_{thr_tag}_time_position_tokenwt"] = float(
                np.average(t_arr, weights=w_arr)
            )
            out[f"derisk_{thr_tag}_token_frac"] = float(w_arr.sum() / gross_tokens)
        else:
            out[f"derisk_{thr_tag}_time_position_mean"] = np.nan
            out[f"derisk_{thr_tag}_time_position_tokenwt"] = np.nan
            out[f"derisk_{thr_tag}_token_frac"] = 0.0

    for h_frac in horizon_fracs:
        tag = f"h{h_frac:.3f}".replace(".", "p")
        raw_markout_grosswt = 0.0
        raw_markout_abs_grosswt = 0.0
        late_spike_edge_grosswt = 0.0
        late_spike_token_frac = 0.0
        held_to_end_token_frac = 0.0

        for lot in lots:
            side_sign = int(lot["side_sign"])
            entry_time = float(lot["entry_time"])
            entry_price = float(lot["entry_price"])
            open_tokens = float(lot["remaining_tokens"])
            if open_tokens <= 1e-12:
                continue

            token_weight = open_tokens / gross_tokens
            t_pct = entry_time + h_frac * market_lifetime

            t_derisk_25 = _find_lot_derisk_time(lot, 0.25)
            t_derisk_75 = _find_lot_derisk_time(lot, 0.50)

            if pd.isna(t_derisk_25):
                t_future = market_end_ts
                held_to_end_token_frac += token_weight
            else:
                t_cap = t_derisk_75 if pd.notna(t_derisk_75) else market_end_ts
                t_future = max(t_derisk_25, min(t_pct, t_cap))

            future_price = _price_at_or_after(hist, t_future)
            raw_post_move = (
                future_price - entry_price if side_sign == +1 else entry_price - future_price
            )
            scaled_post_move = raw_post_move / price_range

            raw_markout_grosswt += token_weight * scaled_post_move
            raw_markout_abs_grosswt += token_weight * abs(scaled_post_move)

            if pd.notna(terminal_start_ts) and entry_time >= terminal_start_ts:
                late_spike_token_frac += token_weight
                late_spike_edge_grosswt += token_weight * max(scaled_post_move, 0.0)

        adjusted_markout_grosswt = (
            raw_markout_grosswt - late_spike_lambda * late_spike_edge_grosswt
        )

        out[f"adaptive_markout_raw_grosswt_{tag}"] = raw_markout_grosswt
        out[f"adaptive_markout_abs_grosswt_{tag}"] = raw_markout_abs_grosswt
        out[f"adaptive_late_spike_edge_grosswt_{tag}"] = late_spike_edge_grosswt
        out[f"adaptive_late_spike_token_frac_{tag}"] = late_spike_token_frac
        out[f"adaptive_held_to_end_token_frac_{tag}"] = held_to_end_token_frac
        out[f"adaptive_markout_latespike_adj_grosswt_{tag}"] = adjusted_markout_grosswt

        for alpha in realized_alpha_grid:
            alpha_tag = str(alpha).replace(".", "p")
            out[f"adaptive_plus_realized_alpha{alpha_tag}_{tag}"] = (
                adjusted_markout_grosswt + alpha * realized_edge_grosswt
            )
            out[f"adaptive_minus_realized_alpha{alpha_tag}_{tag}"] = (
                adjusted_markout_grosswt - alpha * realized_edge_grosswt
            )

    return out


# ---------------------------------------------------------------------------
# Contrarian price features
# ---------------------------------------------------------------------------

def compute_contrarian_price_features(
    trades_df: pd.DataFrame,
    history_df: pd.DataFrame,
    low_price_quantile: float = 0.25,
    recent_trend_lookback_sec: int = 21600,
    flat_move_threshold: float = 0.03,
) -> dict[str, float]:
    """Cash-share features for trading at extreme prices and against trend.

    Returns the fraction of cash deployed at low (resp. high) YES
    prices (using ``low_price_quantile`` from history), and the
    fraction of cash that was BUY-during-flat/down or SELL-during-
    flat/up using a rolling-slope trend label over
    ``recent_trend_lookback_sec``.
    """
    trades = _prep_trades(trades_df)
    hist = _prep_history(history_df)

    if len(trades) == 0 or len(hist) == 0:
        return {
            "cash_at_low_price_frac": np.nan,
            "buy_cash_at_low_price_frac": np.nan,
            "sell_cash_at_high_price_frac": np.nan,
            "cash_against_recent_trend_frac": np.nan,
            "buy_cash_against_recent_downtrend_frac": np.nan,
            "sell_cash_against_recent_uptrend_frac": np.nan,
            "recent_trend_mean_for_buys": np.nan,
            "recent_trend_mean_for_sells": np.nan,
        }

    total_cash = trades["cash_amount"].sum()
    buy_cash = trades.loc[trades["norm_side"] == "BUY", "cash_amount"].sum()
    sell_cash = trades.loc[trades["norm_side"] == "SELL", "cash_amount"].sum()

    low_price_threshold = hist["yes_price"].quantile(low_price_quantile)
    high_price_threshold = hist["yes_price"].quantile(1 - low_price_quantile)

    cash_at_low_price = trades.loc[
        trades["norm_price"] <= low_price_threshold, "cash_amount"
    ].sum()
    buy_cash_at_low_price = trades.loc[
        (trades["norm_side"] == "BUY") & (trades["norm_price"] <= low_price_threshold),
        "cash_amount",
    ].sum()
    sell_cash_at_high_price = trades.loc[
        (trades["norm_side"] == "SELL") & (trades["norm_price"] >= high_price_threshold),
        "cash_amount",
    ].sum()

    cash_at_low_price_frac = cash_at_low_price / total_cash if total_cash > 0 else np.nan
    buy_cash_at_low_price_frac = (
        buy_cash_at_low_price / buy_cash if buy_cash > 0 else np.nan
    )
    sell_cash_at_high_price_frac = (
        sell_cash_at_high_price / sell_cash if sell_cash > 0 else np.nan
    )

    trend_values: list[float] = []
    trend_labels: list[Any] = []
    for _, row in trades.iterrows():
        t = row["timestamp"]
        win = hist[
            (hist["timestamp"] >= t - recent_trend_lookback_sec)
            & (hist["timestamp"] <= t)
        ].copy()
        slope = _rolling_price_slope(win)
        if pd.isna(slope):
            implied_move = np.nan
            label = np.nan
        else:
            implied_move = slope * recent_trend_lookback_sec
            if implied_move > flat_move_threshold:
                label = "up"
            elif implied_move < -flat_move_threshold:
                label = "down"
            else:
                label = "flat"
        trend_values.append(implied_move)
        trend_labels.append(label)

    trades = trades.copy()
    trades["recent_trend_value"] = trend_values
    trades["recent_trend_label"] = trend_labels

    against_mask = (
        ((trades["norm_side"] == "BUY") & (trades["recent_trend_label"].isin(["flat", "down"])))
        | ((trades["norm_side"] == "SELL") & (trades["recent_trend_label"].isin(["flat", "up"])))
    )
    cash_against_recent_trend = trades.loc[against_mask, "cash_amount"].sum()
    buy_cash_against_recent_downtrend = trades.loc[
        (trades["norm_side"] == "BUY")
        & (trades["recent_trend_label"].isin(["flat", "down"])),
        "cash_amount",
    ].sum()
    sell_cash_against_recent_uptrend = trades.loc[
        (trades["norm_side"] == "SELL")
        & (trades["recent_trend_label"].isin(["flat", "up"])),
        "cash_amount",
    ].sum()

    cash_against_recent_trend_frac = (
        cash_against_recent_trend / total_cash if total_cash > 0 else np.nan
    )
    buy_cash_against_recent_downtrend_frac = (
        buy_cash_against_recent_downtrend / buy_cash if buy_cash > 0 else np.nan
    )
    sell_cash_against_recent_uptrend_frac = (
        sell_cash_against_recent_uptrend / sell_cash if sell_cash > 0 else np.nan
    )

    recent_trend_mean_for_buys = trades.loc[
        trades["norm_side"] == "BUY", "recent_trend_value"
    ].mean()
    recent_trend_mean_for_sells = trades.loc[
        trades["norm_side"] == "SELL", "recent_trend_value"
    ].mean()

    return {
        "cash_at_low_price_frac": cash_at_low_price_frac,
        "buy_cash_at_low_price_frac": buy_cash_at_low_price_frac,
        "sell_cash_at_high_price_frac": sell_cash_at_high_price_frac,
        "cash_against_recent_trend_frac": cash_against_recent_trend_frac,
        "buy_cash_against_recent_downtrend_frac": buy_cash_against_recent_downtrend_frac,
        "sell_cash_against_recent_uptrend_frac": sell_cash_against_recent_uptrend_frac,
        "recent_trend_mean_for_buys": recent_trend_mean_for_buys,
        "recent_trend_mean_for_sells": recent_trend_mean_for_sells,
    }


# ---------------------------------------------------------------------------
# Extreme inventory and day features
# ---------------------------------------------------------------------------

def compute_extreme_inventory_and_day_features(
    trades_df: pd.DataFrame,
    history_df: pd.DataFrame,
    market_start_day: pd.Timestamp,
    market_end_day: pd.Timestamp,
    low_extreme_threshold: float = 0.10,
    high_extreme_threshold: float = 0.90,
    dynamic_extreme_quantile: float = 0.10,
) -> dict[str, float]:
    """Cash fractions at extreme prices (fixed + percentile) and day patterns.

    Returns:
        Cash fractions for low-price BUYs and high-price SELLs under a
        fixed threshold and under a percentile threshold of history,
        with non-flattening masks (BUY when not short / SELL when not
        long). Plus day-level distinct-active-day counts, HHIs of cash
        per day, and active-day fraction over the market window.
    """
    trades = _prep_trades(trades_df)
    hist = _prep_history(history_df)

    if len(trades) == 0:
        return {
            "low_buy_cash_frac_leq_0p1": 0,
            "high_sell_cash_frac_geq_0p9": 0,
            "low_buy_building_cash_frac": 0,
            "high_sell_nonflattening_cash_frac": 0,
            "extreme_trade_nonflattening_cash_frac": 0,
            "low_buy_building_cash_frac_pctl10": 0,
            "high_sell_nonflattening_cash_frac_pctl90": 0,
            "extreme_trade_nonflattening_cash_frac_pctl": 0,
            "n_low_buy_building_trades_pctl10": 0,
            "n_high_sell_nonflattening_trades_pctl90": 0,
            "n_low_buy_building_trades": 0,
            "n_high_sell_nonflattening_trades": 0,
            "n_distinct_buy_days": 0,
            "n_distinct_sell_days": 0,
            "n_distinct_active_days": 0,
            "buy_day_concentration_hhi": 0,
            "sell_day_concentration_hhi": 0,
            "active_day_concentration_hhi": 0,
            "active_day_frac": 0,
            "low_buy_high_sell_pctl": 0,
        }

    trades = trades.copy()
    trades["trade_day"] = pd.to_datetime(trades["timestamp"], unit="s", utc=True).dt.floor("D")

    total_cash = trades["cash_amount"].sum()
    buy_cash = trades.loc[trades["norm_side"] == "BUY", "cash_amount"].sum()
    sell_cash = trades.loc[trades["norm_side"] == "SELL", "cash_amount"].sum()

    trades["inventory_after"] = trades["cum_inventory"]
    trades["inventory_before"] = trades["cum_inventory"] - trades["signed_token"]

    low_buy_mask = (
        (trades["norm_side"] == "BUY") & (trades["norm_price"] <= low_extreme_threshold)
    )
    high_sell_mask = (
        (trades["norm_side"] == "SELL") & (trades["norm_price"] >= high_extreme_threshold)
    )

    low_buy_cash_frac_leq_0p1 = (
        trades.loc[low_buy_mask, "cash_amount"].sum() / buy_cash if buy_cash > 0 else 0.0
    )
    high_sell_cash_frac_geq_0p9 = (
        trades.loc[high_sell_mask, "cash_amount"].sum() / sell_cash if sell_cash > 0 else 0.0
    )

    low_buy_building_mask = low_buy_mask & (trades["inventory_before"] >= 0)
    high_sell_nonflattening_mask = high_sell_mask & (trades["inventory_before"] <= 0)
    extreme_nonflattening_mask = low_buy_building_mask | high_sell_nonflattening_mask

    low_buy_building_cash_frac = (
        trades.loc[low_buy_building_mask, "cash_amount"].sum() / buy_cash
        if buy_cash > 0
        else 0.0
    )
    high_sell_nonflattening_cash_frac = (
        trades.loc[high_sell_nonflattening_mask, "cash_amount"].sum() / sell_cash
        if sell_cash > 0
        else 0.0
    )
    extreme_trade_nonflattening_cash_frac = (
        trades.loc[extreme_nonflattening_mask, "cash_amount"].sum() / total_cash
        if total_cash > 0
        else 0.0
    )

    n_low_buy_building_trades = int(low_buy_building_mask.sum())
    n_high_sell_nonflattening_trades = int(high_sell_nonflattening_mask.sum())

    low_price_threshold_pctl = hist["yes_price"].quantile(dynamic_extreme_quantile)
    high_price_threshold_pctl = hist["yes_price"].quantile(1 - dynamic_extreme_quantile)

    low_buy_mask_pctl = (
        (trades["norm_side"] == "BUY") & (trades["norm_price"] <= low_price_threshold_pctl)
    )
    high_sell_mask_pctl = (
        (trades["norm_side"] == "SELL") & (trades["norm_price"] >= high_price_threshold_pctl)
    )

    low_buy_building_mask_pctl = low_buy_mask_pctl & (trades["inventory_before"] >= 0)
    high_sell_nonflattening_mask_pctl = high_sell_mask_pctl & (trades["inventory_before"] <= 0)
    extreme_nonflattening_mask_pctl = (
        low_buy_building_mask_pctl | high_sell_nonflattening_mask_pctl
    )

    low_buy_building_cash_frac_pctl10 = (
        trades.loc[low_buy_building_mask_pctl, "cash_amount"].sum() / buy_cash
        if buy_cash > 0
        else 0.0
    )
    high_sell_nonflattening_cash_frac_pctl90 = (
        trades.loc[high_sell_nonflattening_mask_pctl, "cash_amount"].sum() / sell_cash
        if sell_cash > 0
        else 0.0
    )
    extreme_trade_nonflattening_cash_frac_pctl = (
        trades.loc[extreme_nonflattening_mask_pctl, "cash_amount"].sum() / total_cash
        if total_cash > 0
        else 0.0
    )

    n_low_buy_building_trades_pctl10 = int(low_buy_building_mask_pctl.sum())
    n_high_sell_nonflattening_trades_pctl90 = int(high_sell_nonflattening_mask_pctl.sum())

    buy_trades = trades[trades["norm_side"] == "BUY"]
    sell_trades = trades[trades["norm_side"] == "SELL"]
    all_trades = trades

    n_distinct_buy_days = buy_trades["trade_day"].nunique() if len(buy_trades) > 0 else 0
    n_distinct_sell_days = sell_trades["trade_day"].nunique() if len(sell_trades) > 0 else 0
    n_distinct_active_days = all_trades["trade_day"].nunique() if len(all_trades) > 0 else 0

    def _day_hhi(df_side: pd.DataFrame) -> float:
        if len(df_side) == 0:
            return np.nan
        cash_by_day = df_side.groupby("trade_day")["cash_amount"].sum()
        total = cash_by_day.sum()
        if total <= 0:
            return np.nan
        shares = cash_by_day / total
        return float((shares ** 2).sum())

    buy_day_concentration_hhi = _day_hhi(buy_trades)
    sell_day_concentration_hhi = _day_hhi(sell_trades)
    active_day_concentration_hhi = _day_hhi(all_trades)

    market_duration_days = max(1, int((market_end_day - market_start_day).days + 1))
    active_day_frac = n_distinct_active_days / market_duration_days

    return {
        "low_buy_cash_frac_leq_0p1": low_buy_cash_frac_leq_0p1,
        "high_sell_cash_frac_geq_0p9": high_sell_cash_frac_geq_0p9,
        "low_buy_building_cash_frac": low_buy_building_cash_frac,
        "high_sell_nonflattening_cash_frac": high_sell_nonflattening_cash_frac,
        "extreme_trade_nonflattening_cash_frac": extreme_trade_nonflattening_cash_frac,
        "low_buy_building_cash_frac_pctl10": low_buy_building_cash_frac_pctl10,
        "high_sell_nonflattening_cash_frac_pctl90": high_sell_nonflattening_cash_frac_pctl90,
        "extreme_trade_nonflattening_cash_frac_pctl": extreme_trade_nonflattening_cash_frac_pctl,
        "n_low_buy_building_trades_pctl10": n_low_buy_building_trades_pctl10,
        "n_high_sell_nonflattening_trades_pctl90": n_high_sell_nonflattening_trades_pctl90,
        "n_low_buy_building_trades": n_low_buy_building_trades,
        "n_high_sell_nonflattening_trades": n_high_sell_nonflattening_trades,
        "n_distinct_buy_days": n_distinct_buy_days,
        "n_distinct_sell_days": n_distinct_sell_days,
        "n_distinct_active_days": n_distinct_active_days,
        "buy_day_concentration_hhi": buy_day_concentration_hhi,
        "sell_day_concentration_hhi": sell_day_concentration_hhi,
        "active_day_concentration_hhi": active_day_concentration_hhi,
        "active_day_frac": active_day_frac,
        "low_buy_high_sell_pctl": (
            high_sell_nonflattening_cash_frac_pctl90 + low_buy_building_cash_frac_pctl10
        ) / 2,
    }


# ---------------------------------------------------------------------------
# Market-wide volume helper
# ---------------------------------------------------------------------------

def build_market_volume_df_from_orderfilled(
    orderfilled_df: pd.DataFrame,
    yes_token: str,
    no_token: str,
    usdc_asset_ids: list[str] | set[str] | None = None,
    token_decimals: int = TOKEN_DECIMALS,
    usdc_decimals: int = USDC_DECIMALS,
) -> pd.DataFrame:
    """Convert raw OrderFilled events into a market-wide volume table.

    For each cash↔token fill, emits one row with ``timestamp``,
    ``cash_amount``, ``token_amount``, ``outcome`` (``"YES"`` or
    ``"NO"``), and ``yes_price`` (the YES-equivalent price, mirroring
    NO fills as ``1 - raw_price``). Fills that don't have one cash side
    and one token side are dropped. ``usdc_asset_ids`` defaults to
    "anything that is not yes_token/no_token".

    Args:
        orderfilled_df: DataFrame of raw events with ``timestamp``,
            ``makerAssetId``, ``takerAssetId``, ``makerAmountFilled``,
            ``takerAmountFilled``.
        yes_token: CLOB token id of YES.
        no_token: CLOB token id of NO.
        usdc_asset_ids: Optional explicit USDC asset ids; inferred
            otherwise.
        token_decimals: Decimal places for outcome tokens.
        usdc_decimals: Decimal places for USDC.

    Returns:
        A DataFrame sorted by ``timestamp`` with the columns above.

    Raises:
        ValueError: If required columns are missing or USDC cannot be
            inferred.
    """
    df = orderfilled_df.copy()

    required = [
        "timestamp",
        "makerAssetId",
        "takerAssetId",
        "makerAmountFilled",
        "takerAmountFilled",
    ]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"orderfilled_df missing columns: {missing}")

    df["timestamp"] = pd.to_numeric(df["timestamp"], errors="coerce")
    df["makerAssetId"] = df["makerAssetId"].astype(str)
    df["takerAssetId"] = df["takerAssetId"].astype(str)
    df["makerAmountFilled"] = pd.to_numeric(df["makerAmountFilled"], errors="coerce")
    df["takerAmountFilled"] = pd.to_numeric(df["takerAmountFilled"], errors="coerce")
    df = df.dropna(subset=["timestamp", "makerAmountFilled", "takerAmountFilled"]).copy()

    yes_token = str(yes_token)
    no_token = str(no_token)

    if usdc_asset_ids is None:
        token_set = {yes_token, no_token}
        asset_values = set(df["makerAssetId"]).union(set(df["takerAssetId"]))
        inferred = list(asset_values - token_set)
        if len(inferred) == 0:
            raise ValueError("Could not infer USDC asset id")
        usdc_asset_ids = set(inferred)
    else:
        usdc_asset_ids = {str(x) for x in usdc_asset_ids}

    def parse_row(row: pd.Series) -> pd.Series:
        maker_asset = row["makerAssetId"]
        taker_asset = row["takerAssetId"]
        maker_amt = row["makerAmountFilled"]
        taker_amt = row["takerAmountFilled"]

        if maker_asset in {yes_token, no_token} and taker_asset in usdc_asset_ids:
            token_asset = maker_asset
            token_amount = maker_amt / token_decimals
            cash_amount = taker_amt / usdc_decimals
        elif maker_asset in usdc_asset_ids and taker_asset in {yes_token, no_token}:
            token_asset = taker_asset
            token_amount = taker_amt / token_decimals
            cash_amount = maker_amt / usdc_decimals
        else:
            return pd.Series({
                "cash_amount": np.nan,
                "token_amount": np.nan,
                "outcome": np.nan,
                "yes_price": np.nan,
            })

        outcome = "YES" if token_asset == yes_token else "NO"
        raw_price = (
            cash_amount / token_amount if token_amount and token_amount > 0 else np.nan
        )
        yes_price = (
            raw_price
            if outcome == "YES"
            else (1 - raw_price if pd.notna(raw_price) else np.nan)
        )
        return pd.Series({
            "cash_amount": cash_amount,
            "token_amount": token_amount,
            "outcome": outcome,
            "yes_price": yes_price,
        })

    parsed = df.apply(parse_row, axis=1)
    out = pd.concat(
        [df[["timestamp"]].reset_index(drop=True), parsed.reset_index(drop=True)],
        axis=1,
    )
    return out.dropna(subset=["cash_amount"]).sort_values("timestamp").reset_index(drop=True)


# ---------------------------------------------------------------------------
# Per-(wallet, market) orchestrator
# ---------------------------------------------------------------------------

_MARKOUTS1_FEATURE_NAMES = frozenset(
    f"{side}_markout_{stat}_{h}s"
    for side in ("buy", "sell")
    for stat in ("mean", "median")
    for h in (300, 3600, 21600, 86400)
)

_CONTRARIAN_FEATURE_NAMES = frozenset({
    "cash_at_low_price_frac",
    "buy_cash_at_low_price_frac",
    "sell_cash_at_high_price_frac",
    "cash_against_recent_trend_frac",
    "buy_cash_against_recent_downtrend_frac",
    "sell_cash_against_recent_uptrend_frac",
    "recent_trend_mean_for_buys",
    "recent_trend_mean_for_sells",
})


def compute_market_user_features(
    parsed_trades: list[dict],
    history_df: pd.DataFrame,
    market_start_time: float,
    market_end_time: float,
    *,
    final_outcome_yes: int | float | None = None,
    major_move_time: float | None = None,
    major_move_kwargs: dict[str, Any] | None = None,
    feature_filter: set[str] | None = None,
    low_price_quantile: float = 0.25,
    recent_trend_lookback_sec: int = 21600,
    flat_move_threshold: float = 0.03,
    low_extreme_threshold: float = 0.10,
    high_extreme_threshold: float = 0.90,
    early_time_frac: float = 0.5,
) -> dict[str, Any]:
    """Compute every per-aspect feature for one (wallet, market) pair.

    Takes already-parsed wallet trades plus the market's price history
    and runs every feature function in this module. Major-move
    detection is invoked automatically if ``major_move_time`` isn't
    provided.

    Args:
        parsed_trades: Raw trades for the wallet, e.g.
            ``parse_orderfilled_events(...)["wallet_trades"][address]``.
            Will be normalized to YES-space internally.
        history_df: YES-price history (output of
            :func:`polycluster.parsing.build_history_df_from_orderfilled`).
        market_start_time: Unix timestamp of market start.
        market_end_time: Unix timestamp of market end.
        final_outcome_yes: 1 if YES won, 0 if NO won, None if unknown.
            Used by alignment-, unrealized-edge-, and adaptive-markout
            features. If None, those features are NaN-filled (or in the
            case of adaptive markouts, computed assuming yes_wins=False).
        major_move_time: If provided, skip detection and use this
            timestamp directly.
        major_move_kwargs: Optional kwargs forwarded to
            :func:`detect_major_move` when ``major_move_time`` is None.
        low_price_quantile: For
            :func:`compute_contrarian_price_features`.
        recent_trend_lookback_sec: For
            :func:`compute_contrarian_price_features`.
        flat_move_threshold: For
            :func:`compute_contrarian_price_features`.
        low_extreme_threshold: For
            :func:`compute_extreme_inventory_and_day_features`.
        high_extreme_threshold: For
            :func:`compute_extreme_inventory_and_day_features`.
        early_time_frac: Cutoff for early-period variants of path-
            alignment and realized-PnL features.

    Returns:
        A flat dict of feature name -> value covering every aspect.
    """
    hist = _prep_history(history_df)
    market_start_day = pd.to_datetime(hist["timestamp"].min(), unit="s", utc=True).floor("D")
    market_end_day = pd.to_datetime(hist["timestamp"].max(), unit="s", utc=True).floor("D")

    yes_dicts = normalize_trades_to_yes_dicts(parsed_trades)
    trades_df = pd.DataFrame(yes_dicts) if yes_dicts else pd.DataFrame()

    if major_move_time is None:
        major_move_kwargs = major_move_kwargs or {}
        mm = detect_major_move(hist, **major_move_kwargs)
        major_move_time = mm["major_move_time"]
    else:
        mm = {
            "major_move_time": major_move_time,
            "major_move_start_price": _nearest_price_at_time(hist, major_move_time),
            "major_move_future_price": np.nan,
            "major_move_delta": np.nan,
        }

    if "major_move_time" in mm:
        mm["major_move_time_position_pct"] = _safe_time_position(
            mm["major_move_time"], market_start_time, market_end_time
        )
        mm["major_move_time_remaining_pct"] = _safe_time_remaining_pct(
            mm["major_move_time"], market_start_time, market_end_time
        )
        del mm["major_move_time"]

    feats: dict[str, Any] = {}
    feats.update(mm)

    feats.update(compute_entry_features(
        trades_df=trades_df,
        history_df=hist,
        market_start_time=market_start_time,
        market_end_time=market_end_time,
        major_move_time=major_move_time,
    ))

    feats.update(compute_unrealized_features(
        parsed_trades=parsed_trades,
        final_outcome_yes=final_outcome_yes,
        history=hist,
    ))

    feats.update(compute_holding_features(
        trades_df=trades_df,
        final_outcome_yes=final_outcome_yes,
    ))

    feats.update(compute_markout_features(
        trades_df=trades_df,
        history_df=hist,
        feature_filter=feature_filter,
    ))
    if feature_filter is None or (_MARKOUTS1_FEATURE_NAMES & feature_filter):
        feats.update(compute_markout_features1(trades_df=trades_df, history_df=hist))

    feats.update(compute_realized_pnl_features(
        trades_df=trades_df,
        market_start_time=market_start_time,
        market_end_time=market_end_time,
        early_time_frac=early_time_frac,
    ))

    feats.update(compute_path_alignment_features(
        trades_df=trades_df,
        final_outcome_yes=final_outcome_yes,
        market_start_time=market_start_time,
        market_end_time=market_end_time,
        early_time_frac=early_time_frac,
    ))

    if feature_filter is None or (_CONTRARIAN_FEATURE_NAMES & feature_filter):
        feats.update(compute_contrarian_price_features(
            trades_df=trades_df,
            history_df=hist,
            low_price_quantile=low_price_quantile,
            recent_trend_lookback_sec=recent_trend_lookback_sec,
            flat_move_threshold=flat_move_threshold,
        ))

    feats.update(compute_extreme_inventory_and_day_features(
        trades_df=trades_df,
        history_df=hist,
        market_start_day=market_start_day,
        market_end_day=market_end_day,
        low_extreme_threshold=low_extreme_threshold,
        high_extreme_threshold=high_extreme_threshold,
    ))

    yes_wins = (final_outcome_yes is not None) and (int(final_outcome_yes) == 1)
    feats.update(compute_adaptive_markout_features(
        trades_df=trades_df,
        history_df=hist,
        yes_wins=yes_wins,
    ))

    return feats


# ---------------------------------------------------------------------------
# High-level wrapper
# ---------------------------------------------------------------------------

def get_user_features(
    wallet: str | list[str],
    market: Market | str,
    *,
    timeout: float = 30.0,
    page_size: int = 100,
    min_window: int = 60,
    checkpointed: bool = False,
    initial_chunk_seconds: int = 1800,
    verbose: bool = False,
    final_outcome_yes: int | float | None = None,
    major_move_time: float | None = None,
    major_move_kwargs: dict[str, Any] | None = None,
    feature_filter: set[str] | None = None,
    low_price_quantile: float = 0.25,
    recent_trend_lookback_sec: int = 21600,
    flat_move_threshold: float = 0.03,
    low_extreme_threshold: float = 0.10,
    high_extreme_threshold: float = 0.90,
    early_time_frac: float = 0.5,
) -> dict[str, Any] | dict[str, dict[str, Any]]:
    """Fetch one or more wallets' features for a market, end-to-end.

    Convenience wrapper that:

    1. Resolves the market via
       :func:`~polycluster.markets.get_market_by_slug` if a slug is
       passed.
    2. Fetches every OrderFilled event once via
       :func:`~polycluster.events.get_market_orderfilled_events`.
    3. Builds the YES-price history with
       :func:`~polycluster.parsing.build_history_df_from_orderfilled`.
    4. Buckets events by wallet via
       :func:`~polycluster.parsing.parse_orderfilled_events`.
    5. Calls :func:`compute_market_user_features` per wallet.

    Wallet matching is case-insensitive. Passing a list of wallets is
    cheap because the fetch + parse happens exactly once.

    If ``final_outcome_yes`` is None, it is auto-derived from
    ``market.outcome_prices`` when the market is closed (1 if YES
    settled at >= 0.5, 0 otherwise); for open markets it stays None.

    Args:
        wallet: A single wallet address, or a list of them.
        market: A :class:`~polycluster.markets.Market` or a slug.
        timeout: HTTP timeout in seconds.
        page_size: Subgraph page size.
        min_window: Smallest sub-window for the recursive fetch (only
            used when ``checkpointed=False``).
        checkpointed: Use the checkpointed fetcher instead of the
            recursive one.
        initial_chunk_seconds: Chunk size for the checkpointed fetcher.
        verbose: Print progress while fetching events.
        final_outcome_yes: Override the auto-derived outcome.
        major_move_time: See :func:`compute_market_user_features`.
        major_move_kwargs: See :func:`compute_market_user_features`.
        low_price_quantile: See :func:`compute_market_user_features`.
        recent_trend_lookback_sec: See :func:`compute_market_user_features`.
        flat_move_threshold: See :func:`compute_market_user_features`.
        low_extreme_threshold: See :func:`compute_market_user_features`.
        high_extreme_threshold: See :func:`compute_market_user_features`.
        early_time_frac: See :func:`compute_market_user_features`.

    Returns:
        If ``wallet`` is a string: a feature dict for that wallet.
        If ``wallet`` is a list: a ``dict`` mapping each input address
        (preserving its original casing) to its feature dict.
    """
    if isinstance(market, str):
        market = get_market_by_slug(market, timeout=timeout)

    if final_outcome_yes is None and market.closed and market.outcome_prices:
        final_outcome_yes = 1 if float(market.outcome_prices[0]) >= 0.5 else 0

    events = get_market_orderfilled_events(
        market,
        page_size=page_size,
        timeout=timeout,
        min_window=min_window,
        checkpointed=checkpointed,
        initial_chunk_seconds=initial_chunk_seconds,
        verbose=verbose,
    )
    if isinstance(events, dict):
        events = events["rows"]

    history_df = build_history_df_from_orderfilled(
        events,
        yes_token=market.yes_token_id,
        no_token=market.no_token_id,
    )
    parsed = parse_orderfilled_events(events, market.token_ids, market.outcomes)

    def _features_for(w: str) -> dict[str, Any]:
        user_trades = parsed["wallet_trades"].get(w.lower(), [])
        return compute_market_user_features(
            parsed_trades=user_trades,
            history_df=history_df,
            market_start_time=market.start_ts,
            market_end_time=market.end_ts,
            final_outcome_yes=final_outcome_yes,
            major_move_time=major_move_time,
            major_move_kwargs=major_move_kwargs,
            feature_filter=feature_filter,
            low_price_quantile=low_price_quantile,
            recent_trend_lookback_sec=recent_trend_lookback_sec,
            flat_move_threshold=flat_move_threshold,
            low_extreme_threshold=low_extreme_threshold,
            high_extreme_threshold=high_extreme_threshold,
            early_time_frac=early_time_frac,
        )

    if isinstance(wallet, str):
        return _features_for(wallet)
    return {w: _features_for(w) for w in wallet}
