"""Fetch OrderFilled events from the Polymarket orderbook subgraph.

The CLOB exchange emits an ``OrderFilled`` event for every fill on
Polymarket. These are indexed by Goldsky and exposed via a public
GraphQL endpoint. This module provides:

* :func:`get_orderfilled_events` - one-shot fetch with recursive window
  splitting, suitable for short / medium-volume markets.
* :func:`get_orderfilled_events_checkpointed` - chunked, resumable
  fetcher for long-running or high-volume markets that would time the
  one-shot version out.
* :func:`get_market_orderfilled_events` - convenience wrapper that takes
  a :class:`~polycluster.markets.Market` (or slug) and dispatches to one
  of the two above.

Each returned row is a dict with the GraphQL fields plus two provenance
tags: ``query_side`` (``"maker"`` or ``"taker"``, indicating which half
of the trade matched our token-id filter) and ``queried_token_id`` (the
token ID that was filtered for in this query).
"""

from __future__ import annotations

import time
from typing import Iterable, Literal, Sequence

import requests

from polycluster.markets import Market, _normalize_token_ids, get_market_by_slug

#: Goldsky subgraph endpoint hosting Polymarket's CLOB orderbook events.
SUBGRAPH_URL = (
    "https://api.goldsky.com/api/public/project_cl6mb8i9h0003e201j6li0diw/"
    "subgraphs/orderbook-subgraph/0.0.1/gn"
)

#: Polymarket's user-facing Data API. Used as a fallback when the Goldsky
#: subgraph above is stale or unavailable. Each ``/trades`` record is the
#: per-user view of one fill (the maker's counterparty is not exposed), so
#: events synthesized from this source carry a placeholder maker wallet.
DATA_API_BASE = "https://data-api.polymarket.com"

#: The Data API caps pagination at this offset (a server-side limit, not a
#: client choice). Only the most recent ~3000 trades per market are reachable.
DATA_API_MAX_OFFSET = 3000

#: Synthetic maker wallet stamped on events recovered from the Data API.
#: Surfaces in :func:`parse_orderfilled_events`'s ``wallet_trades`` dict as a
#: phantom wallet that owns one half of every trade — harmless as long as
#: nothing queries it as if it were a real participant.
DATA_API_PLACEHOLDER_WALLET = "0x" + "0" * 40

Side = Literal["maker", "taker"]
Source = Literal["auto", "subgraph", "data_api"]


def _log(verbose: bool, msg: str) -> None:
    """Print a progress line to stdout when verbose, with flush so the
    line shows up immediately in notebooks instead of being buffered."""
    if verbose:
        print(msg, flush=True)


def _is_timeout_error(exc: BaseException) -> bool:
    """Heuristically detect whether an exception came from a server timeout.

    The subgraph reports timeouts both as HTTP-level timeouts and as
    in-response error messages (e.g. ``"canceling statement due to
    statement timeout"``), so we match on string markers in the
    exception text.
    """
    msg = str(exc).lower()
    return any(
        marker in msg
        for marker in ("statement timeout", "timed out", "timeout")
    )


def _query_orderfilled_window(
    token_ids: Sequence[str],
    start_ts: int,
    end_ts: int,
    side: Side = "maker",
    page_size: int = 1000,
    timeout: float = 30.0,
    after_id: str | None = None,
) -> list[dict]:
    """Fetch a single page of OrderFilled events from the subgraph.

    Issues a GraphQL ``orderFilledEvents`` query filtered by
    ``[start_ts, end_ts]`` and the given asset side (maker or taker).
    Results are ordered by ``id`` ascending so that ``after_id`` can be
    used as a stable cursor; callers re-sort by timestamp once the full
    set is collected.

    Args:
        token_ids: Token IDs to filter by, applied to either
            ``makerAssetId`` or ``takerAssetId`` depending on ``side``.
        start_ts: Inclusive lower bound on event ``timestamp`` (unix
            seconds).
        end_ts: Inclusive upper bound on event ``timestamp`` (unix
            seconds).
        side: Which side of the trade to filter by.
        page_size: Maximum rows per page (subgraph caps at 1000).
        timeout: HTTP timeout in seconds.
        after_id: Optional cursor; only events with ``id > after_id``
            are returned. Used by checkpointed pagination.

    Returns:
        A list of raw event dicts, each tagged with ``query_side``.

    Raises:
        RuntimeError: If the GraphQL response contains an ``errors`` array.
        requests.HTTPError: On non-2xx HTTP status.
    """
    asset_field = "makerAssetId_in" if side == "maker" else "takerAssetId_in"
    tokens_str = ", ".join(f'"{t}"' for t in token_ids)

    where_lines = [
        f'timestamp_gte: "{start_ts}"',
        f'timestamp_lte: "{end_ts}"',
        f"{asset_field}: [{tokens_str}]",
    ]
    if after_id is not None:
        where_lines.append(f'id_gt: "{after_id}"')
    where_block = "\n          ".join(where_lines)

    query = f"""
    {{
      orderFilledEvents(
        first: {page_size},
        orderBy: id,
        orderDirection: asc,
        where: {{
          {where_block}
        }}
      ) {{
        id
        timestamp
        transactionHash
        orderHash
        maker
        taker
        makerAssetId
        takerAssetId
        makerAmountFilled
        takerAmountFilled
        fee
      }}
    }}
    """
#response = requests.post(SUBGRAPH_URL, json={"query": query}, timeout=timeout)
    response = requests.post(SUBGRAPH_URL, json={"query": query})
    response.raise_for_status()
    payload = response.json()

    if "errors" in payload:
        raise RuntimeError(payload["errors"])

    rows = payload["data"]["orderFilledEvents"]
    for r in rows:
        r["query_side"] = side
    return rows


def _dedupe_rows(rows: Iterable[dict]) -> list[dict]:
    """Drop exact duplicate event rows from an iterable, preserving order.

    The same OrderFilled event can appear in multiple sub-queries
    (maker- and taker-side, overlapping time windows). We dedupe on the
    full identity tuple - including the provenance tags - so that
    side-tagged copies of the same event are treated as distinct only
    when they truly differ.
    """
    deduped = []
    seen = set()
    for row in rows:
        key = (
            row.get("id"),
            row.get("timestamp"),
            row.get("transactionHash"),
            row.get("orderHash"),
            row.get("maker"),
            row.get("taker"),
            row.get("makerAssetId"),
            row.get("takerAssetId"),
            row.get("makerAmountFilled"),
            row.get("takerAmountFilled"),
            row.get("query_side"),
            row.get("queried_token_id"),
        )
        if key not in seen:
            seen.add(key)
            deduped.append(row)
    return deduped


def _sort_rows(rows: list[dict]) -> list[dict]:
    """Stable sort rows by (timestamp, transactionHash, orderHash, id)."""
    return sorted(
        rows,
        key=lambda x: (
            int(x["timestamp"]),
            str(x.get("transactionHash", "")),
            str(x.get("orderHash", "")),
            str(x.get("id", "")),
        ),
    )


def _fetch_orderfilled_side(
    token_ids: Sequence[str],
    start_ts: int,
    end_ts: int,
    side: Side = "maker",
    page_size: int = 1000,
    timeout: float = 30.0,
    min_window: int = 60,
    max_depth: int = 30,
    verbose: bool = False,
) -> list[dict]:
    """Recursively fetch every event for one side by halving saturated windows.

    The subgraph caps results per query at ``page_size``. When a window
    returns exactly that many rows it may be truncated, so we recurse on
    its halves. We also recurse on server-side timeouts so smaller
    windows can succeed where the original failed.

    This is the building block for :func:`get_orderfilled_events`. For
    long-running markets, prefer the checkpointed variant.

    When ``verbose=True``, prints a line for every successful query and
    every saturation-triggered split so progress is visible even outside
    timeout events.
    """

    def helper(s: int, e: int, depth: int = 0) -> list[dict]:
        if s > e:
            return []
        if depth > max_depth:
            raise RuntimeError(
                f"Exceeded max recursion depth on window [{s}, {e}]"
            )

        try:
            t_q = time.time()
            rows = _query_orderfilled_window(
                token_ids=token_ids,
                start_ts=s,
                end_ts=e,
                side=side,
                page_size=page_size,
                timeout=timeout,
            )
            dt = time.time() - t_q

            if len(rows) < page_size:
                _log(
                    verbose,
                    f"[polycluster]     d={depth} side={side} "
                    f"[{s}, {e}] ({e - s}s): {len(rows)} rows in {dt:.2f}s "
                    f"-> done",
                )
                return rows

            if e - s <= min_window:
                _log(
                    verbose,
                    f"[polycluster]     d={depth} side={side} "
                    f"[{s}, {e}] ({e - s}s): {len(rows)} rows in {dt:.2f}s "
                    f"-> at min_window, returning (may be truncated)",
                )
                return rows

            mid = (s + e) // 2
            _log(
                verbose,
                f"[polycluster]     d={depth} side={side} "
                f"[{s}, {e}] ({e - s}s): {len(rows)} rows in {dt:.2f}s "
                f"-> saturated, splitting at {mid}",
            )
            return helper(s, mid, depth + 1) + helper(mid + 1, e, depth + 1)

        except Exception as exc:
            if _is_timeout_error(exc):
                if e - s <= min_window:
                    print(
                        f"[WARN] timeout on minimal window [{s}, {e}] "
                        f"for side={side}; skipping split"
                    )
                    raise
                mid = (s + e) // 2
                print(
                    f"[INFO] timeout on [{s}, {e}] for side={side}; "
                    f"splitting into [{s}, {mid}] and [{mid + 1}, {e}]"
                )
                return helper(s, mid, depth + 1) + helper(mid + 1, e, depth + 1)
            raise

    rows = helper(start_ts, end_ts)
    return _sort_rows(_dedupe_rows(rows))


def get_orderfilled_events(
    token_ids: list[str] | str,
    start_ts: int,
    end_ts: int,
    page_size: int = 1000,
    timeout: float = 30.0,
    min_window: int = 60,
    verbose: bool = False,
) -> list[dict]:
    """Fetch all OrderFilled events for one or more CLOB tokens in a window.

    For each token ID, queries both maker- and taker-side events,
    dedupes the union, and returns rows sorted by
    ``(timestamp, transactionHash, orderHash, id)``.

    Each row is tagged with ``queried_token_id`` and ``query_side`` to
    preserve the provenance of the row when a single event matches via
    multiple sub-queries.

    Use this for shorter or smaller markets. For long-running markets
    that time out the recursive split, use
    :func:`get_orderfilled_events_checkpointed`.

    Args:
        token_ids: A single CLOB token ID, a list of them, or a
            JSON-encoded list (as returned by the gamma API).
        start_ts: Inclusive lower bound on event timestamp (unix
            seconds).
        end_ts: Inclusive upper bound on event timestamp (unix seconds).
        page_size: Subgraph page size.
        timeout: HTTP timeout per request, in seconds.
        min_window: Smallest sub-window (in seconds) the recursive split
            will produce; below this the split halts even if the window
            looks saturated.
        verbose: If True, print a progress line at the start and end of
            each ``(token_id, side)`` pair, including the row count and
            elapsed seconds. Useful for tracking long-running fetches in
            notebooks.

    Returns:
        A sorted, deduped list of event dicts.
    """
    token_ids = _normalize_token_ids(token_ids)
    all_rows: list[dict] = []

    n_iterations = len(token_ids) * 2
    iteration = 0
    t_total = time.time()
    _log(
        verbose,
        f"[polycluster] get_orderfilled_events: {len(token_ids)} token(s), "
        f"window=[{start_ts}, {end_ts}] ({end_ts - start_ts}s)",
    )

    for token_idx, token_id in enumerate(token_ids):
        for side in ("maker", "taker"):
            iteration += 1
            t0 = time.time()
            _log(
                verbose,
                f"[polycluster]   ({iteration}/{n_iterations}) "
                f"token {token_idx + 1}/{len(token_ids)} side={side}: starting...",
            )
            side_rows = _fetch_orderfilled_side(
                [token_id],
                start_ts,
                end_ts,
                side=side,
                page_size=page_size,
                timeout=timeout,
                min_window=min_window,
                verbose=verbose,
            )
            for row in side_rows:
                row["queried_token_id"] = token_id
            all_rows.extend(side_rows)
            _log(
                verbose,
                f"[polycluster]   ({iteration}/{n_iterations}) "
                f"token {token_idx + 1}/{len(token_ids)} side={side}: "
                f"{len(side_rows)} rows in {time.time() - t0:.1f}s",
            )

    out = _sort_rows(_dedupe_rows(all_rows))
    _log(
        verbose,
        f"[polycluster] get_orderfilled_events: done, {len(out)} unique rows "
        f"in {time.time() - t_total:.1f}s",
    )
    return out


def _fetch_orderfilled_side_checkpointed(
    token_ids: Sequence[str],
    start_ts: int,
    end_ts: int,
    side: Side = "maker",
    page_size: int = 1000,
    timeout: float = 30.0,
    min_window: int = 0,
    initial_chunk_seconds: int = 1800,
    verbose: bool = False,
    log_prefix: str = "",
) -> dict:
    """Like :func:`_fetch_orderfilled_side` but chunked and resumable.

    Walks the time window in fixed-size chunks
    (``initial_chunk_seconds``), splitting saturated chunks recursively.
    When a chunk hits ``min_window`` and is still saturated, falls back
    to cursor pagination via ``id_gt``. On a server-timeout that can't
    be split further, returns the partial results plus enough state to
    resume.
    """
    if isinstance(token_ids, str):
        token_ids = [token_ids]

    start_ts = int(start_ts)
    end_ts = int(end_ts)

    def paginate_exact_window(s: int, e: int, first_rows: list[dict]) -> list[dict]:
        all_rows = list(first_rows)
        if not first_rows:
            return all_rows

        last_id = first_rows[-1]["id"]
        while True:
            next_page = _query_orderfilled_window(
                token_ids=token_ids,
                start_ts=s,
                end_ts=e,
                side=side,
                page_size=page_size,
                timeout=timeout,
                after_id=last_id,
            )
            if not next_page:
                break
            all_rows.extend(next_page)
            last_id = next_page[-1]["id"]
            if len(next_page) < page_size:
                break
        return all_rows

    def helper(s: int, e: int) -> list[dict]:
        rows = _query_orderfilled_window(
            token_ids=token_ids,
            start_ts=s,
            end_ts=e,
            side=side,
            page_size=page_size,
            timeout=timeout,
        )

        if len(rows) < page_size:
            return rows

        if e - s <= min_window:
            return paginate_exact_window(s, e, rows)

        mid = (s + e) // 2
        return helper(s, mid) + helper(mid + 1, e)

    all_rows: list[dict] = []
    chunk_start = start_ts
    total_chunks = max(1, (end_ts - start_ts + initial_chunk_seconds) // initial_chunk_seconds)
    chunk_idx = 0

    while chunk_start <= end_ts:
        chunk_end = min(chunk_start + initial_chunk_seconds - 1, end_ts)
        chunk_idx += 1
        t0 = time.time()
        try:
            chunk_rows = helper(chunk_start, chunk_end)
            all_rows.extend(chunk_rows)
            _log(
                verbose,
                f"{log_prefix}chunk {chunk_idx}/{total_chunks} "
                f"[{chunk_start}, {chunk_end}]: {len(chunk_rows)} rows "
                f"(running total {len(all_rows)}) in {time.time() - t0:.1f}s",
            )
        except RuntimeError as err:
            _log(
                verbose,
                f"{log_prefix}chunk {chunk_idx}/{total_chunks} "
                f"[{chunk_start}, {chunk_end}]: FAILED after "
                f"{time.time() - t0:.1f}s ({err}); checkpointing for resume",
            )
            return {
                "rows": _sort_rows(_dedupe_rows(all_rows)),
                "complete": False,
                "resume_side": side,
                "resume_start_ts": chunk_start,
                "failed_chunk_end_ts": chunk_end,
                "error": str(err),
            }
        chunk_start = chunk_end + 1

    return {
        "rows": _sort_rows(_dedupe_rows(all_rows)),
        "complete": True,
        "resume_side": None,
        "resume_start_ts": None,
        "failed_chunk_end_ts": None,
        "error": None,
    }


def get_orderfilled_events_checkpointed(
    token_ids: list[str] | str,
    start_ts: int,
    end_ts: int,
    page_size: int = 1000,
    timeout: float = 30.0,
    min_window: int = 0,
    initial_chunk_seconds: int = 1800,
    sides: Sequence[Side] = ("maker", "taker"),
    resume_token_idx: int = 0,
    resume_side_idx: int = 0,
    resume_start_ts: int | None = None,
    verbose: bool = False,
) -> dict:
    """Resumable variant of :func:`get_orderfilled_events` for long markets.

    Iterates over (token_id, side) pairs and fetches each via
    :func:`_fetch_orderfilled_side_checkpointed`. If a chunk fails - for
    example, persistent server timeouts - the function returns the
    partial results so far plus a checkpoint
    (``resume_token_idx``, ``resume_side_idx``, ``resume_start_ts``)
    that can be passed back in to continue where it left off.

    Typical usage::

        result = get_orderfilled_events_checkpointed(token_ids, start, end)
        rows = result["rows"]
        while not result["complete"]:
            result = get_orderfilled_events_checkpointed(
                token_ids, start, end,
                resume_token_idx=result["resume_token_idx"],
                resume_side_idx=result["resume_side_idx"],
                resume_start_ts=result["resume_start_ts"],
            )
            rows = _dedupe_rows(rows + result["rows"])

    Args:
        token_ids: A single CLOB token ID, a list of them, or a
            JSON-encoded list.
        start_ts: Inclusive lower bound on event timestamp (unix
            seconds).
        end_ts: Inclusive upper bound on event timestamp (unix seconds).
        page_size: Subgraph page size.
        timeout: HTTP timeout per request, in seconds.
        min_window: Smallest sub-window before falling back to cursor
            pagination.
        initial_chunk_seconds: Size of each time-chunk processed in one
            helper call.
        sides: Order sides to query (default: both maker and taker).
        resume_token_idx: Index in ``token_ids`` at which to resume.
        resume_side_idx: Index in ``sides`` at which to resume.
        resume_start_ts: Timestamp at which to resume the current chunk.
        verbose: If True, print per-chunk progress and per-(token, side)
            summaries while fetching. Useful for tracking long-running
            checkpointed fetches in notebooks.

    Returns:
        A dict with keys ``rows``, ``complete``, ``resume_token_idx``,
        ``resume_side_idx``, ``resume_token_id``, ``resume_side``,
        ``resume_start_ts``, ``failed_chunk_end_ts`` and ``error``. When
        ``complete`` is ``True`` the resume fields are ``None``.
    """
    token_ids = _normalize_token_ids(token_ids)
    all_rows: list[dict] = []

    n_pairs = (len(token_ids) - resume_token_idx) * len(sides) - resume_side_idx
    pair_idx = 0
    t_total = time.time()
    _log(
        verbose,
        f"[polycluster] get_orderfilled_events_checkpointed: "
        f"{len(token_ids)} token(s), sides={list(sides)}, "
        f"window=[{start_ts}, {end_ts}] ({end_ts - start_ts}s), "
        f"chunk={initial_chunk_seconds}s",
    )

    for token_idx in range(resume_token_idx, len(token_ids)):
        token_id = token_ids[token_idx]
        side_start_idx = resume_side_idx if token_idx == resume_token_idx else 0

        for side_idx in range(side_start_idx, len(sides)):
            side = sides[side_idx]
            side_start_ts = (
                resume_start_ts
                if (
                    token_idx == resume_token_idx
                    and side_idx == resume_side_idx
                    and resume_start_ts is not None
                )
                else start_ts
            )

            pair_idx += 1
            t_pair = time.time()
            _log(
                verbose,
                f"[polycluster] ({pair_idx}/{n_pairs}) "
                f"token {token_idx + 1}/{len(token_ids)} side={side}: "
                f"starting from {side_start_ts}",
            )

            result = _fetch_orderfilled_side_checkpointed(
                token_ids=[token_id],
                start_ts=side_start_ts,
                end_ts=end_ts,
                side=side,
                page_size=page_size,
                timeout=timeout,
                min_window=min_window,
                initial_chunk_seconds=initial_chunk_seconds,
                verbose=verbose,
                log_prefix=f"[polycluster]   token {token_idx + 1} side={side} ",
            )

            for row in result["rows"]:
                row = dict(row)
                row["queried_token_id"] = token_id
                row["queried_side"] = side
                all_rows.append(row)

            _log(
                verbose,
                f"[polycluster] ({pair_idx}/{n_pairs}) "
                f"token {token_idx + 1}/{len(token_ids)} side={side}: "
                f"{len(result['rows'])} rows in {time.time() - t_pair:.1f}s "
                f"(complete={result['complete']})",
            )

            if not result["complete"]:
                _log(
                    verbose,
                    f"[polycluster] checkpointing at token_idx={token_idx} "
                    f"side_idx={side_idx} resume_start_ts={result['resume_start_ts']}",
                )
                return {
                    "rows": _dedupe_rows(all_rows),
                    "complete": False,
                    "resume_token_idx": token_idx,
                    "resume_side_idx": side_idx,
                    "resume_token_id": token_id,
                    "resume_side": side,
                    "resume_start_ts": result["resume_start_ts"],
                    "failed_chunk_end_ts": result["failed_chunk_end_ts"],
                    "error": result["error"],
                }

    out_rows = _dedupe_rows(all_rows)
    _log(
        verbose,
        f"[polycluster] get_orderfilled_events_checkpointed: done, "
        f"{len(out_rows)} unique rows in {time.time() - t_total:.1f}s",
    )
    return {
        "rows": out_rows,
        "complete": True,
        "resume_token_idx": None,
        "resume_side_idx": None,
        "resume_token_id": None,
        "resume_side": None,
        "resume_start_ts": None,
        "failed_chunk_end_ts": None,
        "error": None,
    }


def _data_api_trade_to_orderfilled_event(t: dict) -> dict:
    """Map one Polymarket Data API ``/trades`` record to the subgraph schema.

    The Data API exposes only the user's side of a fill (no counterparty),
    so the synthesized event puts the user on the taker leg and stamps the
    maker as :data:`DATA_API_PLACEHOLDER_WALLET`. Token + cash amounts use
    the 6-decimal scaling Polymarket applies on-chain (matching what the
    subgraph emits).
    """
    proxy = str(t["proxyWallet"]).lower()
    side = str(t["side"]).upper()
    asset = str(t["asset"])
    size = float(t["size"])
    price = float(t["price"])
    ts = int(t["timestamp"])
    tx_hash = str(t["transactionHash"])

    token_amt_raw = round(size * 1_000_000)
    cash_amt_raw = round(size * price * 1_000_000)

    if side == "BUY":
        maker_asset, taker_asset = asset, "0"
        maker_amount, taker_amount = token_amt_raw, cash_amt_raw
    else:
        maker_asset, taker_asset = "0", asset
        maker_amount, taker_amount = cash_amt_raw, token_amt_raw

    return {
        "id": f"{tx_hash}_{asset[:16]}_{ts}",
        "timestamp": str(ts),
        "transactionHash": tx_hash,
        "orderHash": "",
        "maker": DATA_API_PLACEHOLDER_WALLET,
        "taker": proxy,
        "makerAssetId": maker_asset,
        "takerAssetId": taker_asset,
        "makerAmountFilled": str(maker_amount),
        "takerAmountFilled": str(taker_amount),
        "fee": "0",
        "query_side": "taker",
        "queried_token_id": asset,
    }


def get_market_trades_via_data_api(
    market: Market | str,
    *,
    page_size: int = 500,
    timeout: float = 30.0,
    verbose: bool = False,
) -> list[dict]:
    """Fetch a market's trades from Polymarket's Data API.

    Returns events in the same dict shape as
    :func:`get_orderfilled_events`, so
    :func:`polycluster.parsing.parse_orderfilled_events` and
    :func:`polycluster.parsing.build_history_df_from_orderfilled` can
    consume the result unchanged. Intended as a fallback when the Goldsky
    subgraph is stale.

    Two limitations vs. the subgraph fetcher:

    * The Data API caps pagination at offset
      :data:`DATA_API_MAX_OFFSET` (3000). Markets with more than that many
      trades are silently truncated to the most recent 3000.
    * Each fill yields one record (the user's view), so counterparty
      wallets and ``role="maker"`` trade records are not recovered.

    Args:
        market: A :class:`~polycluster.markets.Market` or slug string.
        page_size: Per-request page size. The endpoint accepts up to 1000.
        timeout: HTTP timeout per request, in seconds.
        verbose: If True, print pagination progress.

    Returns:
        List of subgraph-shaped OrderFilled event dicts. Empty if the
        market has no trades or the conditionId is not resolvable.
    """
    if isinstance(market, str):
        market = get_market_by_slug(market, timeout=timeout)

    condition_id = market.raw.get("conditionId")
    if not condition_id:
        _log(verbose, "[data-api] market.raw has no conditionId; returning []")
        return []

    page_size = max(1, min(page_size, 1000))
    raw_trades: list[dict] = []
    offset = 0
    while offset < DATA_API_MAX_OFFSET:
        remaining = DATA_API_MAX_OFFSET - offset
        this_limit = min(page_size, remaining)
        resp = requests.get(
            f"{DATA_API_BASE}/trades",
            params={
                "market": condition_id,
                "limit": this_limit,
                "offset": offset,
            },
            timeout=timeout,
        )
        resp.raise_for_status()
        page = resp.json()
        if isinstance(page, dict) and "error" in page:
            _log(verbose, f"[data-api] stopped at offset={offset}: {page['error']}")
            break
        if not page:
            break
        raw_trades.extend(page)
        _log(
            verbose,
            f"[data-api] fetched {len(raw_trades)} trades (offset={offset})",
        )
        offset += len(page)
        if len(page) < this_limit:
            break

    return [_data_api_trade_to_orderfilled_event(t) for t in raw_trades]


def get_market_orderfilled_events(
    market: Market | str,
    start_ts: int | None = None,
    end_ts: int | None = None,
    page_size: int = 1000,
    timeout: float = 30.0,
    min_window: int = 60,
    checkpointed: bool = False,
    initial_chunk_seconds: int = 1800,
    source: Source = "auto",
    verbose: bool = False,
) -> list[dict] | dict:
    """Fetch OrderFilled events for a Polymarket market.

    Convenience wrapper that resolves the market (if a slug is given),
    then dispatches to either :func:`get_orderfilled_events` (the
    default, one-shot) or :func:`get_orderfilled_events_checkpointed`
    when ``checkpointed=True``.

    By default the full market lifetime
    ``[market.start_ts, market.end_ts]`` is fetched. ``start_ts`` and
    ``end_ts`` can be overridden to query a sub-window - for example,
    only the last 24 hours before resolution.

    Args:
        market: Either a :class:`~polycluster.markets.Market` already
            fetched via
            :func:`~polycluster.markets.get_market_by_slug`, or a slug
            string (the function will fetch the market for you).
        start_ts: Inclusive lower bound on event timestamp (unix
            seconds). Defaults to ``market.start_ts``.
        end_ts: Inclusive upper bound on event timestamp (unix
            seconds). Defaults to ``market.end_ts``.
        page_size: Subgraph page size.
        timeout: HTTP timeout per request, in seconds.
        min_window: Smallest sub-window for the recursive split (only
            used when ``checkpointed=False``).
        checkpointed: If ``True``, use the resumable checkpointed
            fetcher and return its full result dict (with ``rows``,
            ``complete``, etc.). If ``False``, return just the row list.
        initial_chunk_seconds: Chunk size for the checkpointed path.
        verbose: If True, print progress lines while fetching - market
            resolution, per-(token, side) starts/ends with row counts
            and elapsed time, and (in the checkpointed path) per-chunk
            progress. Helpful in notebooks to confirm a long-running
            cell is making progress.

    Returns:
        A list of event dicts (when ``checkpointed=False``) or a
        checkpointed-result dict (when ``checkpointed=True``).
    """
    if isinstance(market, str):
        _log(verbose, f"[polycluster] resolving market slug={market!r}...")
        t0 = time.time()
        market = get_market_by_slug(market, timeout=timeout)
        _log(
            verbose,
            f"[polycluster] resolved slug={market.slug!r}: "
            f"{len(market.token_ids)} token(s), closed={market.closed}, "
            f"window=[{market.start_ts}, {market.end_ts}] in {time.time() - t0:.1f}s",
        )

    if start_ts is None:
        start_ts = market.start_ts
    if end_ts is None:
        end_ts = market.end_ts

    if source == "data_api":
        return get_market_trades_via_data_api(
            market, page_size=500, timeout=timeout, verbose=verbose,
        )

    if checkpointed:
        return get_orderfilled_events_checkpointed(
            market.token_ids,
            start_ts,
            end_ts,
            page_size=page_size,
            timeout=timeout,
            initial_chunk_seconds=initial_chunk_seconds,
            verbose=verbose,
        )

    try:
        events = get_orderfilled_events(
            market.token_ids,
            start_ts,
            end_ts,
            page_size=page_size,
            timeout=timeout,
            min_window=min_window,
            verbose=verbose,
        )
    except Exception as exc:
        if source != "auto":
            raise
        _log(
            verbose,
            f"[polycluster] subgraph raised {type(exc).__name__}: {exc!s:.180}; "
            "falling back to Polymarket Data API",
        )
        events = []

    if not events and source == "auto":
        if events == []:
            _log(
                verbose,
                "[polycluster] subgraph returned 0 events; "
                "falling back to Polymarket Data API",
            )
        events = get_market_trades_via_data_api(
            market, page_size=500, timeout=timeout, verbose=verbose,
        )

    return events
