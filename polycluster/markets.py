"""Market metadata fetched from the Polymarket Gamma API.

This module wraps the public ``gamma-api.polymarket.com`` endpoint so the
rest of the pipeline can work with a single :class:`Market` object instead
of repeatedly poking at the raw JSON response.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

import requests

GAMMA_API_BASE = "https://gamma-api.polymarket.com"
DATA_API_BASE = "https://data-api.polymarket.com"
DATA_API_MAX_OFFSET = 3000


def _normalize_token_ids(token_ids: list | str) -> list[str]:
    """Coerce a gamma-api ``clobTokenIds`` payload into a clean ``list[str]``.

    The gamma API sometimes returns the field as a JSON-encoded string
    (e.g. ``'["123", "456"]'``) rather than a real list, so this helper
    normalizes both forms.

    Args:
        token_ids: Either a list of token-id values or a JSON-encoded
            string representing such a list.

    Returns:
        A list of token IDs as strings.

    Raises:
        TypeError: If ``token_ids`` is neither a list nor a string.
    """
    if isinstance(token_ids, list):
        return [str(x) for x in token_ids]
    if isinstance(token_ids, str):
        token_ids = token_ids.strip()
        if token_ids.startswith("[") and token_ids.endswith("]"):
            return [str(x) for x in json.loads(token_ids)]
        return [token_ids]
    raise TypeError(f"Unsupported token_ids type: {type(token_ids)}")


def _normalize_outcomes(outcomes: list | str) -> list[str]:
    """Coerce a gamma-api ``outcomes`` payload into a clean ``list[str]``.

    Args:
        outcomes: Either a list of outcome labels or a JSON-encoded string
            representing such a list.

    Returns:
        A list of outcome labels as strings (e.g. ``["Yes", "No"]``).

    Raises:
        TypeError: If ``outcomes`` is neither a list nor a string.
    """
    if isinstance(outcomes, list):
        return [str(x) for x in outcomes]
    if isinstance(outcomes, str):
        outcomes = outcomes.strip()
        if outcomes.startswith("[") and outcomes.endswith("]"):
            return [str(x) for x in json.loads(outcomes)]
        return [outcomes]
    raise TypeError(f"Unsupported outcomes type: {type(outcomes)}")


def _time_str_to_uint256(time_str: str) -> int:
    """Convert an ISO-8601 timestamp string to an integer UTC unix timestamp.

    Accepts the trailing ``Z`` shorthand for UTC.

    Args:
        time_str: A timestamp like ``"2024-01-01T00:00:00Z"``.

    Returns:
        The corresponding unix timestamp in seconds.
    """
    time_str = time_str.replace("Z", "+00:00")
    dt = datetime.fromisoformat(time_str)
    return int(dt.timestamp())


@dataclass
class Market:
    """A Polymarket market with the fields needed to drive a pipeline run.

    Attributes:
        slug: The human-readable market slug used in URLs.
        outcomes: Ordered list of outcome labels, e.g. ``["Yes", "No"]``.
        token_ids: CLOB token IDs in the same order as ``outcomes``.
        yes_token_id: Convenience pointer to ``token_ids[0]`` for binary
            markets. For non-binary markets this is still the first entry.
        no_token_id: Convenience pointer to ``token_ids[1]`` for binary
            markets; ``None`` if the market has fewer than two outcomes.
        start_ts: Unix timestamp at which the market began accepting orders
            (gamma-api ``acceptingOrdersTimestamp``).
        end_ts: Unix timestamp at which the market closed, or the current
            UTC time if the market is still open.
        closed: Whether the market has resolved/closed.
        outcome_prices: Final outcome prices reported by the gamma API,
            in the same order as ``outcomes``.
        raw: The full untouched gamma-api response, for downstream
            consumers that need fields not promoted onto the dataclass.
    """

    slug: str
    outcomes: list[str]
    token_ids: list[str]
    yes_token_id: str
    no_token_id: str | None
    start_ts: int
    end_ts: int
    closed: bool
    outcome_prices: list[float]
    raw: dict[str, Any] = field(repr=False)


def get_market_by_slug(slug: str, *, timeout: float = 30.0) -> Market:
    """Fetch a market's metadata from the Polymarket Gamma API by slug.

    The returned :class:`Market` carries everything needed to fetch
    orderfilled events for the market: the CLOB token IDs and a time
    window ``[start_ts, end_ts]``. For closed markets ``end_ts`` is the
    on-chain close time; for open markets it is the current UTC timestamp.

    Args:
        slug: The market slug (the trailing path of a Polymarket market URL).
        timeout: HTTP timeout in seconds for the gamma API request.

    Returns:
        A populated :class:`Market`.

    Raises:
        requests.HTTPError: If the gamma API returns a non-2xx response.
        KeyError: If the gamma API response is missing required fields.
    """
    url = f"{GAMMA_API_BASE}/markets/slug/{slug}"
    resp = requests.get(url, timeout=timeout)
    resp.raise_for_status()
    payload = resp.json()

    outcomes = _normalize_outcomes(payload["outcomes"])
    token_ids = _normalize_token_ids(payload["clobTokenIds"])
    start_ts = _time_str_to_uint256(payload["acceptingOrdersTimestamp"])

    closed = str(payload.get("closed", "")).lower() == "true"
    if closed:
        end_ts = _time_str_to_uint256(payload["closedTime"])
    else:
        end_ts = int(datetime.now(timezone.utc).timestamp())

    raw_prices = payload.get("outcomePrices", "[]")
    if isinstance(raw_prices, str):
        outcome_prices = [float(x) for x in json.loads(raw_prices)]
    else:
        outcome_prices = [float(x) for x in raw_prices]

    return Market(
        slug=slug,
        outcomes=outcomes,
        token_ids=token_ids,
        yes_token_id=token_ids[0],
        no_token_id=token_ids[1] if len(token_ids) > 1 else None,
        start_ts=start_ts,
        end_ts=end_ts,
        closed=closed,
        outcome_prices=outcome_prices,
        raw=payload,
    )


def get_markets_traded_by_wallet(
    wallet_address: str,
    *,
    page_size: int = 500,
    timeout: float = 30.0,
    verbose: bool = False,
) -> list[str]:
    """Return the deduped list of market slugs a wallet has traded in.

    Queries Polymarket's Data API ``/trades`` endpoint filtered by
    ``user=<wallet_address>`` and collects the ``slug`` field of every
    trade record. Slugs are returned in the order they are first seen
    (the Data API returns trades newest-first, so a wallet's most
    recently traded market appears first).

    The Data API caps pagination at offset :data:`DATA_API_MAX_OFFSET`
    (3000). Wallets with more than that many trades will silently miss
    the oldest end of their trading history.

    Args:
        wallet_address: A 0x-prefixed wallet address. Case-insensitive;
            normalized to lowercase before being sent to the API.
        page_size: Per-request page size. The endpoint accepts up to 1000.
        timeout: HTTP timeout per request, in seconds.
        verbose: If True, print pagination progress.

    Returns:
        A deduped list of market slugs the wallet has traded in. Empty
        if the wallet has no trades or is unknown to the Data API.

    Raises:
        requests.HTTPError: If the Data API returns a non-2xx response.
    """
    wallet_address = wallet_address.lower()
    page_size = max(1, min(page_size, 1000))

    seen: set[str] = set()
    slugs: list[str] = []
    offset = 0

    while offset < DATA_API_MAX_OFFSET:
        remaining = DATA_API_MAX_OFFSET - offset
        this_limit = min(page_size, remaining)
        resp = requests.get(
            f"{DATA_API_BASE}/trades",
            params={
                "user": wallet_address,
                "limit": this_limit,
                "offset": offset,
            },
            timeout=timeout,
        )
        resp.raise_for_status()
        page = resp.json()

        if isinstance(page, dict) and "error" in page:
            if verbose:
                print(
                    f"[data-api] stopped at offset={offset}: {page['error']}",
                    flush=True,
                )
            break
        if not page:
            break

        for trade in page:
            slug = trade.get("slug")
            if slug and slug not in seen:
                seen.add(slug)
                slugs.append(slug)

        if verbose:
            print(
                f"[data-api] offset={offset}: {len(page)} trades, "
                f"{len(slugs)} unique slugs so far",
                flush=True,
            )

        offset += len(page)
        if len(page) < this_limit:
            break

    return slugs
