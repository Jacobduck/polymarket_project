"""Parse raw OrderFilled events into wallet-level trades and price history.

This module turns the raw subgraph events returned by
:mod:`polycluster.events` into the structured frames the rest of the
pipeline needs:

* :func:`parse_orderfilled_events` - bucket every event by wallet, with
  per-wallet rankings and binary (YES/NO) normalization.
* :func:`build_history_df_from_orderfilled` - condense events into a
  YES-token price history dataframe.
* :func:`normalize_trades_to_yes_dicts` /
  :func:`normalize_trades_to_yes_df` - re-express a wallet's trades in
  YES-equivalent terms (a sell of NO becomes a buy of YES, etc.) in
  either dict or DataFrame form, depending on the downstream consumer.
* :func:`map_trades_to_history` - align a wallet's trades to the price
  history for plotting and feature extraction.
"""

from __future__ import annotations

from collections import defaultdict
from typing import Any, Literal

import pandas as pd

from polycluster.events import get_market_orderfilled_events
from polycluster.markets import (
    Market,
    _normalize_outcomes,
    _normalize_token_ids,
    get_market_by_slug,
)


def parse_orderfilled_events(
    trades: list[dict],
    token_ids: list[str] | str,
    outcome_labels: list[str] | str | None = None,
    token_decimals: int = 6,
    cash_decimals: int = 6,
) -> dict[str, Any]:
    """Bucket OrderFilled events by wallet, with summaries and rankings.

    Walks every event and attributes a ``BUY``/``SELL`` of a particular
    outcome to both the maker and taker wallet. Cash-side events
    (``makerAssetId == 0`` or ``takerAssetId == 0``) are interpreted as
    USDC↔outcome-token trades; pure token↔token trades are also handled
    (rare on Polymarket but possible).

    For binary markets (exactly two outcomes) it additionally produces a
    "binary normalized" view that collapses YES and NO into a single
    YES-axis: a NO sell counts as a YES buy, and vice versa.

    Args:
        trades: Raw OrderFilled event dicts (the rows returned by
            :func:`polycluster.events.get_orderfilled_events`).
        token_ids: Outcome token IDs in the same order as
            ``outcome_labels``. Accepts a list, a JSON-encoded list, or
            a single id.
        outcome_labels: Human-readable outcome labels parallel to
            ``token_ids``. Defaults to ``["Outcome_0", "Outcome_1", ...]``.
        token_decimals: Decimal places for outcome-token amounts (6 on
            Polymarket).
        cash_decimals: Decimal places for cash/USDC amounts (6 on
            Polymarket).

    Returns:
        A dict with the following keys:

        * ``wallet_counts``: ``{wallet -> {f"BUY_{outcome}": int, ...}}``
          counts and amounts of each side/outcome combination.
        * ``wallet_trades``: ``{wallet -> [trade_dict, ...]}`` every
          trade the wallet participated in, with role/side/outcome.
        * ``wallet_summary``: per-wallet aggregates including totals and
          per-outcome ``net_*_amount`` and ``gross_*_amount``.
        * ``rank_by_total``: wallets sorted by total trade-count
          activity (descending).
        * ``rank_by_total_amount``: wallets sorted by total token amount
          traded.
        * ``rank_by_abs_net_amount``: wallets sorted by absolute net
          amount across outcomes - useful for finding directional whales.
        * ``binary_normalized``, ``rank_by_min_norm``,
          ``rank_by_total_norm``, ``rank_by_total_norm_amount``,
          ``rank_by_abs_net_norm_amount``: binary-market views; ``None``
          for non-binary markets.
        * ``token_to_outcome``: mapping of token id to outcome label.

    Raises:
        ValueError: If ``outcome_labels`` is provided but its length
            doesn't match ``token_ids``.
    """
    token_ids = _normalize_token_ids(token_ids)

    if outcome_labels is None:
        outcome_labels = [f"Outcome_{i}" for i in range(len(token_ids))]
    else:
        outcome_labels = _normalize_outcomes(outcome_labels)
        if len(outcome_labels) != len(token_ids):
            raise ValueError(
                "outcome_labels must have the same length as token_ids"
            )

    token_to_outcome = dict(zip(token_ids, outcome_labels))

    def make_count_dict() -> dict[str, float]:
        d: dict[str, float] = {}
        for outcome in outcome_labels:
            d[f"BUY_{outcome}"] = 0
            d[f"SELL_{outcome}"] = 0
            d[f"BUY_{outcome}_amount"] = 0.0
            d[f"SELL_{outcome}_amount"] = 0.0
        return d

    wallet_counts: dict[str, dict[str, float]] = defaultdict(make_count_dict)
    wallet_trades: dict[str, list[dict]] = defaultdict(list)

    for t in trades:
        maker = str(t["maker"]).lower()
        taker = str(t["taker"]).lower()

        maker_asset = str(t["makerAssetId"])
        taker_asset = str(t["takerAssetId"])

        maker_amt_raw = float(t.get("makerAmountFilled", 0))
        taker_amt_raw = float(t.get("takerAmountFilled", 0))

        timestamp = t.get("timestamp")
        tx_hash = t.get("transactionHash")
        order_hash = t.get("orderHash")
        trade_id = t.get("id")
        fee = t.get("fee")

        outcome = None
        maker_side = None
        taker_side = None
        token_amount_raw = None
        cash_amount_raw = None
        matched_token_id = None

        if maker_asset == "0" and taker_asset in token_to_outcome:
            matched_token_id = taker_asset
            outcome = token_to_outcome[taker_asset]
            maker_side = "BUY"
            taker_side = "SELL"
            token_amount_raw = taker_amt_raw
            cash_amount_raw = maker_amt_raw

        elif taker_asset == "0" and maker_asset in token_to_outcome:
            matched_token_id = maker_asset
            outcome = token_to_outcome[maker_asset]
            maker_side = "SELL"
            taker_side = "BUY"
            token_amount_raw = maker_amt_raw
            cash_amount_raw = taker_amt_raw

        elif maker_asset in token_to_outcome:
            matched_token_id = maker_asset
            outcome = token_to_outcome[maker_asset]
            maker_side = "SELL"
            taker_side = "BUY"
            token_amount_raw = maker_amt_raw
            cash_amount_raw = taker_amt_raw

        elif taker_asset in token_to_outcome:
            matched_token_id = taker_asset
            outcome = token_to_outcome[taker_asset]
            maker_side = "BUY"
            taker_side = "SELL"
            token_amount_raw = taker_amt_raw
            cash_amount_raw = maker_amt_raw

        else:
            continue

        token_amount = token_amount_raw / (10 ** token_decimals)
        cash_amount = cash_amount_raw / (10 ** cash_decimals)

        wallet_counts[maker][f"{maker_side}_{outcome}"] += 1
        wallet_counts[taker][f"{taker_side}_{outcome}"] += 1

        wallet_counts[maker][f"{maker_side}_{outcome}_amount"] += token_amount
        wallet_counts[taker][f"{taker_side}_{outcome}_amount"] += token_amount

        wallet_trades[maker].append({
            "trade_id": trade_id,
            "timestamp": timestamp,
            "transactionHash": tx_hash,
            "orderHash": order_hash,
            "role": "maker",
            "side": maker_side,
            "outcome": outcome,
            "token_id": matched_token_id,
            "token_amount": token_amount,
            "cash_amount": cash_amount,
            "makerAssetId": maker_asset,
            "takerAssetId": taker_asset,
            "fee": fee,
            "counterparty": taker,
        })

        wallet_trades[taker].append({
            "trade_id": trade_id,
            "timestamp": timestamp,
            "transactionHash": tx_hash,
            "orderHash": order_hash,
            "role": "taker",
            "side": taker_side,
            "outcome": outcome,
            "token_id": matched_token_id,
            "token_amount": token_amount,
            "cash_amount": cash_amount,
            "makerAssetId": maker_asset,
            "takerAssetId": taker_asset,
            "fee": fee,
            "counterparty": maker,
        })

    wallet_summary: dict[str, dict[str, float]] = {}

    for wallet, c in wallet_counts.items():
        total_activity = sum(v for k, v in c.items() if not k.endswith("_amount"))
        total_amount = sum(v for k, v in c.items() if k.endswith("_amount"))

        summary = dict(c)
        summary["n_trades"] = len(wallet_trades[wallet])
        summary["total_activity"] = total_activity
        summary["total_amount"] = total_amount

        for outcome in outcome_labels:
            summary[f"net_{outcome}_amount"] = (
                summary[f"BUY_{outcome}_amount"] - summary[f"SELL_{outcome}_amount"]
            )
            summary[f"gross_{outcome}_amount"] = (
                summary[f"BUY_{outcome}_amount"] + summary[f"SELL_{outcome}_amount"]
            )

        wallet_summary[wallet] = summary

    rank_by_total = sorted(
        wallet_summary.items(),
        key=lambda x: x[1]["total_activity"],
        reverse=True,
    )

    rank_by_total_amount = sorted(
        wallet_summary.items(),
        key=lambda x: (x[1]["total_amount"], x[1]["total_activity"]),
        reverse=True,
    )

    rank_by_abs_net_amount = sorted(
        wallet_summary.items(),
        key=lambda x: (
            sum(abs(x[1][f"net_{outcome}_amount"]) for outcome in outcome_labels),
            x[1]["total_amount"],
        ),
        reverse=True,
    )

    binary_normalized: dict[str, dict[str, float]] | None = None
    rank_by_min_norm = None
    rank_by_total_norm = None
    rank_by_total_norm_amount = None
    rank_by_abs_net_norm_amount = None

    if len(token_ids) == 2 and len(outcome_labels) == 2:
        first = outcome_labels[0]
        second = outcome_labels[1]

        binary_normalized = {}

        for wallet, c in wallet_counts.items():
            buy_first_norm = c[f"BUY_{first}"] + c[f"SELL_{second}"]
            sell_first_norm = c[f"SELL_{first}"] + c[f"BUY_{second}"]

            buy_first_norm_amount = (
                c[f"BUY_{first}_amount"] + c[f"SELL_{second}_amount"]
            )
            sell_first_norm_amount = (
                c[f"SELL_{first}_amount"] + c[f"BUY_{second}_amount"]
            )

            binary_normalized[wallet] = {
                **dict(c),
                f"BUY_{first}_norm": buy_first_norm,
                f"SELL_{first}_norm": sell_first_norm,
                f"BUY_{first}_norm_amount": buy_first_norm_amount,
                f"SELL_{first}_norm_amount": sell_first_norm_amount,
                "min_side": min(buy_first_norm, sell_first_norm),
                "total_activity_norm": buy_first_norm + sell_first_norm,
                "min_side_amount": min(buy_first_norm_amount, sell_first_norm_amount),
                "total_activity_norm_amount": (
                    buy_first_norm_amount + sell_first_norm_amount
                ),
                "net_norm_amount": buy_first_norm_amount - sell_first_norm_amount,
                "n_trades": len(wallet_trades[wallet]),
            }

        rank_by_min_norm = sorted(
            binary_normalized.items(),
            key=lambda x: (x[1]["min_side"], x[1]["total_activity_norm"]),
            reverse=True,
        )

        rank_by_total_norm = sorted(
            binary_normalized.items(),
            key=lambda x: (x[1]["total_activity_norm"], x[1]["min_side"]),
            reverse=True,
        )

        rank_by_total_norm_amount = sorted(
            binary_normalized.items(),
            key=lambda x: (
                x[1]["total_activity_norm_amount"],
                x[1]["total_activity_norm"],
            ),
            reverse=True,
        )

        rank_by_abs_net_norm_amount = sorted(
            binary_normalized.items(),
            key=lambda x: (
                abs(x[1]["net_norm_amount"]),
                x[1]["total_activity_norm_amount"],
            ),
            reverse=True,
        )

    return {
        "wallet_counts": dict(wallet_counts),
        "wallet_trades": dict(wallet_trades),
        "wallet_summary": wallet_summary,
        "rank_by_total": rank_by_total,
        "rank_by_total_amount": rank_by_total_amount,
        "rank_by_abs_net_amount": rank_by_abs_net_amount,
        "binary_normalized": binary_normalized,
        "rank_by_min_norm": rank_by_min_norm,
        "rank_by_total_norm": rank_by_total_norm,
        "rank_by_total_norm_amount": rank_by_total_norm_amount,
        "rank_by_abs_net_norm_amount": rank_by_abs_net_norm_amount,
        "token_to_outcome": token_to_outcome,
    }


_HISTORY_COLUMNS = [
    "trade_id",
    "timestamp",
    "yes_price",
    "token_id",
    "token_amount",
    "cash_amount",
]


def build_history_df_from_orderfilled(
    orderfilled_events: list[dict],
    yes_token: str,
    no_token: str,
    token_decimals: int = 6,
    cash_decimals: int = 6,
    convert_no_to_yes: bool = True,
    drop_duplicates: bool = True,
    aggregate_same_timestamp: Literal["last", "first", "mean"] | None = "last",
) -> pd.DataFrame:
    """Build a YES-price history dataframe from raw OrderFilled events.

    Each cash↔token fill is converted into a single YES-equivalent
    price observation (NO fills are mirrored as ``1 - no_price`` when
    ``convert_no_to_yes=True``). Optionally aggregates fills that share
    a timestamp.

    Returns the same six columns regardless of branch (empty result,
    no aggregation, or any aggregation strategy):
    ``[trade_id, timestamp, yes_price, token_id, token_amount, cash_amount]``.

    Args:
        orderfilled_events: Raw OrderFilled events.
        yes_token: CLOB token id for the YES outcome.
        no_token: CLOB token id for the NO outcome.
        token_decimals: Decimal places for outcome-token amounts.
        cash_decimals: Decimal places for cash amounts.
        convert_no_to_yes: If True, NO fills produce YES-equivalent
            rows via ``1 - no_price``. If False, NO fills are dropped.
        drop_duplicates: If True, deduplicate by event ``id``.
        aggregate_same_timestamp: How to collapse multiple fills that
            share a timestamp. ``"last"`` / ``"first"`` keep the
            corresponding row; ``"mean"`` averages the numeric columns
            (``yes_price``, ``token_amount``, ``cash_amount``) and
            keeps the last ``trade_id`` / ``token_id``. ``None`` keeps
            every row.

    Returns:
        A DataFrame with the columns listed above, sorted by
        ``timestamp`` ascending.

    Raises:
        ValueError: If ``aggregate_same_timestamp`` is not one of the
            supported values.
    """
    rows: list[dict] = []
    seen_ids: set = set()

    yes_token = str(yes_token)
    no_token = str(no_token)

    for ev in orderfilled_events:
        ev_id = ev.get("id")
        if drop_duplicates and ev_id in seen_ids:
            continue
        if drop_duplicates:
            seen_ids.add(ev_id)

        maker_asset = str(ev.get("makerAssetId"))
        taker_asset = str(ev.get("takerAssetId"))

        maker_amt = float(ev.get("makerAmountFilled", 0))
        taker_amt = float(ev.get("takerAmountFilled", 0))

        token_id = None
        token_amount_raw = None
        cash_amount_raw = None

        if maker_asset == "0" and taker_asset in {yes_token, no_token}:
            cash_amount_raw = maker_amt
            token_amount_raw = taker_amt
            token_id = taker_asset
        elif taker_asset == "0" and maker_asset in {yes_token, no_token}:
            cash_amount_raw = taker_amt
            token_amount_raw = maker_amt
            token_id = maker_asset
        else:
            continue

        if token_amount_raw == 0:
            continue

        cash_amount = cash_amount_raw / (10 ** cash_decimals)
        token_amount = token_amount_raw / (10 ** token_decimals)

        if token_amount == 0:
            continue

        raw_price = cash_amount / token_amount

        if token_id == yes_token:
            yes_price = raw_price
        elif token_id == no_token:
            if not convert_no_to_yes:
                continue
            yes_price = 1 - raw_price
        else:
            continue

        rows.append({
            "trade_id": ev_id,
            "timestamp": int(ev["timestamp"]),
            "yes_price": float(yes_price),
            "token_id": str(token_id),
            "token_amount": float(token_amount),
            "cash_amount": float(cash_amount),
        })

    if not rows:
        return pd.DataFrame(columns=_HISTORY_COLUMNS)

    history_df = (
        pd.DataFrame(rows)
        .sort_values("timestamp")
        .reset_index(drop=True)
    )

    if aggregate_same_timestamp is None:
        return history_df[_HISTORY_COLUMNS]

    if aggregate_same_timestamp in ("last", "first"):
        agg_method = aggregate_same_timestamp
        history_df = (
            history_df.groupby("timestamp", as_index=False)
            .agg(agg_method)
        )
    elif aggregate_same_timestamp == "mean":
        history_df = history_df.groupby("timestamp", as_index=False).agg({
            "trade_id": "last",
            "token_id": "last",
            "yes_price": "mean",
            "token_amount": "mean",
            "cash_amount": "mean",
        })
    else:
        raise ValueError(
            "aggregate_same_timestamp must be one of: 'last', 'first', 'mean', None"
        )

    return (
        history_df[_HISTORY_COLUMNS]
        .sort_values("timestamp")
        .reset_index(drop=True)
    )


def _normalize_trade_to_yes(trade: dict) -> dict:
    """Re-express a single parsed trade in YES-equivalent terms.

    Adds ``price`` (raw price), ``norm_side`` (``BUY``/``SELL`` from
    YES perspective), ``norm_outcome`` (always ``"Yes"``),
    ``norm_price``, and ``signed_yes_amount`` (positive for buys,
    negative for sells).

    Raises:
        ValueError: If ``token_amount`` is non-positive, or if
            ``side``/``outcome`` are not in the expected vocabulary.
    """
    t = dict(trade)

    qty = float(t["token_amount"])
    cash = float(t["cash_amount"])
    if qty <= 0:
        raise ValueError(f"token_amount must be > 0, got {qty}")

    price = cash / qty
    side = str(t["side"]).upper()
    outcome = str(t["outcome"]).capitalize()

    if outcome not in {"Yes", "No"}:
        raise ValueError(f"Unexpected outcome: {outcome}")
    if side not in {"BUY", "SELL"}:
        raise ValueError(f"Unexpected side: {side}")

    if outcome == "Yes":
        norm_side = side
        norm_price = price
    else:
        norm_side = "SELL" if side == "BUY" else "BUY"
        norm_price = 1.0 - price

    t["price"] = price
    t["norm_side"] = norm_side
    t["norm_outcome"] = "Yes"
    t["norm_price"] = norm_price
    t["signed_yes_amount"] = qty if norm_side == "BUY" else -qty
    return t


def normalize_trades_to_yes_dicts(trades: list[dict]) -> list[dict]:
    """Re-express a wallet's trades in YES-equivalent terms (list-of-dicts form).

    Strict variant: relies on the ``outcome`` field already being
    ``"Yes"`` or ``"No"`` (as produced by :func:`parse_orderfilled_events`)
    and raises on anything else. Each output dict preserves the input
    fields and adds ``price``, ``norm_side``, ``norm_outcome`` (always
    ``"Yes"``), ``norm_price`` and ``signed_yes_amount``.

    The returned list is sorted by ``(timestamp, trade_id)`` ascending.
    Use this form when feeding downstream functions that expect
    list-of-dicts (e.g. FIFO PnL, feature extraction).

    Args:
        trades: Parsed trades for one wallet, e.g.
            ``parsed["wallet_trades"][address]``.

    Returns:
        A new list of trade dicts in YES-equivalent terms.
    """
    out = [_normalize_trade_to_yes(tr) for tr in trades]
    out.sort(key=lambda x: (int(x["timestamp"]), str(x.get("trade_id", ""))))
    return out


def normalize_trades_to_yes_df(
    trades: list[dict],
    yes_token: str,
    no_token: str,
    token_decimals: int = 6,
    cash_decimals: int = 6,
) -> pd.DataFrame:
    """Re-express a wallet's trades in YES-equivalent terms (DataFrame form).

    Tolerant variant: keys off the ``token_id`` field (falling back to
    the ``outcome`` text), silently skips trades with zero
    ``token_amount`` or unrecognized tokens, and emits a DataFrame
    suitable for plotting / asof-merging onto a price history.

    The output frame has columns:
    ``[trade_id, timestamp, transactionHash, orderHash, original_side,
    original_outcome, token_amount, cash_amount, yes_trade_price,
    yes_action, signed_yes_qty]`` and is sorted by ``timestamp``.

    ``yes_action`` is one of ``"BUY_YES"`` / ``"SELL_YES"``.
    ``signed_yes_qty`` is positive for buys, negative for sells.

    Note that the ``token_decimals`` / ``cash_decimals`` arguments are
    accepted for parity with related decoders but are unused here:
    ``token_amount`` and ``cash_amount`` on the input trade dicts are
    already decimal-scaled by :func:`parse_orderfilled_events`.

    Args:
        trades: Parsed trades for one wallet (e.g.
            ``parsed["wallet_trades"][address]``).
        yes_token: CLOB token id of the YES outcome.
        no_token: CLOB token id of the NO outcome.
        token_decimals: Accepted for API parity. Unused.
        cash_decimals: Accepted for API parity. Unused.

    Returns:
        A DataFrame with one row per non-skipped trade.
    """
    del token_decimals, cash_decimals  # accepted for API parity

    rows: list[dict] = []

    for tr in trades:
        token_id = str(tr["token_id"])
        side = str(tr["side"]).upper()
        outcome = str(tr["outcome"]).strip().lower()

        token_amount = float(tr["token_amount"])
        cash_amount = float(tr["cash_amount"])

        if token_amount == 0:
            continue

        raw_price = cash_amount / token_amount

        if token_id == yes_token or outcome == "yes":
            yes_price = raw_price
            is_yes = True
        elif token_id == no_token or outcome == "no":
            yes_price = 1 - raw_price
            is_yes = False
        else:
            continue

        if is_yes:
            yes_action = "BUY_YES" if side == "BUY" else "SELL_YES"
        else:
            yes_action = "SELL_YES" if side == "BUY" else "BUY_YES"

        rows.append({
            "trade_id": tr["trade_id"],
            "timestamp": int(tr["timestamp"]),
            "transactionHash": tr.get("transactionHash"),
            "orderHash": tr.get("orderHash"),
            "original_side": side,
            "original_outcome": tr.get("outcome"),
            "token_amount": token_amount,
            "cash_amount": cash_amount,
            "yes_trade_price": yes_price,
            "yes_action": yes_action,
            "signed_yes_qty": (
                token_amount if yes_action == "BUY_YES" else -token_amount
            ),
        })

    df = pd.DataFrame(rows, columns=[
        "trade_id", "timestamp", "transactionHash", "orderHash",
        "original_side", "original_outcome", "token_amount",
        "cash_amount", "yes_trade_price", "yes_action", "signed_yes_qty",
    ])
    if not df.empty:
        df = df.sort_values("timestamp").reset_index(drop=True)
    return df


def map_trades_to_history(
    trades_df: pd.DataFrame,
    history_df: pd.DataFrame,
    tolerance: int | None = None,
) -> pd.DataFrame:
    """As-of-merge a wallet's trades onto a YES-price history.

    For each row in ``trades_df``, finds the nearest history row by
    timestamp and attaches its YES price as ``history_yes_price``.
    Uses :func:`pandas.merge_asof` with ``direction="nearest"``.

    Args:
        trades_df: Output of :func:`normalize_trades_to_yes_df`. Must
            contain a ``timestamp`` column.
        history_df: Output of :func:`build_history_df_from_orderfilled`,
            or any DataFrame with ``timestamp`` and ``yes_price``.
        tolerance: Maximum permissible distance, in unix seconds,
            between a trade timestamp and the matched history
            timestamp. Trades farther than ``tolerance`` from any
            history row receive ``NaN`` for the matched fields.
            ``None`` means no tolerance limit.

    Returns:
        A DataFrame with ``trades_df``'s columns plus the
        history columns (with ``yes_price`` renamed to
        ``history_yes_price``).
    """
    if trades_df.empty:
        return trades_df.copy()

    trades = trades_df.copy()
    hist = history_df.copy()

    trades["timestamp"] = pd.to_numeric(
        trades["timestamp"], errors="coerce"
    ).astype("int64")
    hist["timestamp"] = pd.to_numeric(
        hist["timestamp"], errors="coerce"
    ).astype("int64")

    trades = trades.sort_values("timestamp").reset_index(drop=True)
    hist = hist.sort_values("timestamp").reset_index(drop=True)

    merged = pd.merge_asof(
        trades,
        hist,
        on="timestamp",
        direction="nearest",
        tolerance=tolerance,
    )

    return merged.rename(columns={"yes_price": "history_yes_price"})


def get_user_trades_df(
    wallet: str | list[str],
    market: Market | str,
    *,
    timeout: float = 30.0,
    page_size: int = 100,
    min_window: int = 60,
    checkpointed: bool = False,
    initial_chunk_seconds: int = 1800,
    verbose: bool = False,
) -> pd.DataFrame | dict[str, pd.DataFrame]:
    """Fetch one or more wallets' trades for a market, end-to-end.

    Convenience wrapper that chains the four pipeline steps:

    1. :func:`~polycluster.markets.get_market_by_slug` (if a slug is
       passed) to resolve the market's token IDs and outcomes.
    2. :func:`~polycluster.events.get_market_orderfilled_events` to
       fetch every OrderFilled event for the market.
    3. :func:`parse_orderfilled_events` to bucket events by wallet.
    4. :func:`normalize_trades_to_yes_df` to re-express each wallet's
       trades on a single YES axis (``BUY_YES`` / ``SELL_YES``).

    The events are fetched and parsed exactly once regardless of how
    many wallets are requested, so passing a list of wallets is much
    cheaper than calling this function in a loop.

    Wallet address matching is case-insensitive (the parser lowercases
    addresses internally).

    Args:
        wallet: A single wallet address, or a list of them.
        market: Either a :class:`~polycluster.markets.Market` (if you
            already fetched it) or a slug string.
        timeout: HTTP timeout in seconds for both the gamma-api and
            subgraph requests.
        page_size: Subgraph page size.
        min_window: Smallest sub-window for the recursive fetch (only
            used when ``checkpointed=False``).
        checkpointed: If True, use the resumable checkpointed fetcher.
            Useful for long-running markets that time out the one-shot
            recursive fetch.
        initial_chunk_seconds: Chunk size for the checkpointed path.
        verbose: If True, print progress lines while fetching the
            underlying events.

    Returns:
        If ``wallet`` is a string: a DataFrame with one row per trade
        for that wallet, in the format produced by
        :func:`normalize_trades_to_yes_df`. Empty if the wallet has no
        trades on this market.

        If ``wallet`` is a list: a ``dict`` mapping each input address
        (preserving its original casing) to that wallet's trades
        DataFrame.
    """
    if isinstance(market, str):
        market = get_market_by_slug(market, timeout=timeout)

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

    parsed = parse_orderfilled_events(
        events,
        market.token_ids,
        market.outcomes,
    )

    def _df_for(w: str) -> pd.DataFrame:
        user_trades = parsed["wallet_trades"].get(w.lower(), [])
        return normalize_trades_to_yes_df(
            user_trades,
            yes_token=market.yes_token_id,
            no_token=market.no_token_id,
        )

    if isinstance(wallet, str):
        return _df_for(wallet)

    return {w: _df_for(w) for w in wallet}
