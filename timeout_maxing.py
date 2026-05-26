"""Stress-test the Goldsky subgraph to see when timeouts trigger.

Two tests:
  1. RAW: call _query_orderfilled_window directly on the full market
     window with page_size=1000 — no recursive splitting. Either we get
     up to 1000 rows back, or the server kills the query and we see the
     "statement timeout" error.
  2. FULL: call get_market_orderfilled_events with page_size=1000 and
     verbose=True so the recursive splitter's [INFO] timeout lines
     surface every time a window had to be halved.

Edit MARKET_SLUG below to point at the market you want to hammer.
High-volume / long-lived markets are most likely to expose timeouts.
"""

from __future__ import annotations

import time
import traceback

from polycluster.events import (
    _query_orderfilled_window,
    get_market_orderfilled_events,
)
from polycluster.markets import get_market_by_slug


# Candidates from the project's known-insider list — pick one you expect
# to be high-volume:
#   "nothing-ever-happens-microstrategy"
#   "will-microstrategy-announce-holding-650k-btc-by-november-30"
#   "openai-browser-in-2025"
MARKET_SLUG = "nothing-ever-happens-microstrategy"


def stress_raw_query(market) -> None:
    print("\n=== RAW STRESS TEST (single _query_orderfilled_window) ===")
    print(f"market: {market.slug}")
    print(
        f"window: [{market.start_ts}, {market.end_ts}] "
        f"({market.end_ts - market.start_ts}s ≈ "
        f"{(market.end_ts - market.start_ts) / 86400:.1f}d)"
    )
    print(f"tokens: {len(market.token_ids)} — using first one, maker side")
    print("page_size=1000, no client timeout (server timeout is the binding limit)")

    t0 = time.time()
    try:
        rows = _query_orderfilled_window(
            token_ids=[market.token_ids[0]],
            start_ts=market.start_ts,
            end_ts=market.end_ts,
            side="maker",
            page_size=1000,
        )
        elapsed = time.time() - t0
        print(f"OK: {len(rows)} rows in {elapsed:.2f}s")
        if len(rows) == 1000:
            print("Page saturated — there are likely more rows beyond this batch.")
    except Exception as exc:
        elapsed = time.time() - t0
        print(f"FAILED after {elapsed:.2f}s: {type(exc).__name__}")
        print(f"  message: {str(exc)[:500]}")
        traceback.print_exc()


def stress_full_fetch(market) -> None:
    print("\n=== FULL FETCH (get_market_orderfilled_events, verbose) ===")
    print(f"market: {market.slug}")
    print("page_size=1000 — watch for '[INFO] timeout on [...]' lines from the splitter")

    t0 = time.time()
    try:
        events = get_market_orderfilled_events(
            market,
            page_size=1000,
            verbose=True,
        )
        elapsed = time.time() - t0
        print(f"\nOK: {len(events)} events in {elapsed:.2f}s")
    except Exception as exc:
        elapsed = time.time() - t0
        print(f"\nFAILED after {elapsed:.2f}s: {type(exc).__name__}")
        print(f"  message: {str(exc)[:500]}")
        traceback.print_exc()


def main() -> None:
    print(f"Resolving market slug={MARKET_SLUG!r}...")
    t0 = time.time()
    market = get_market_by_slug(MARKET_SLUG)
    print(
        f"Resolved in {time.time() - t0:.2f}s: "
        f"{len(market.token_ids)} token(s), closed={market.closed}"
    )

    stress_raw_query(market)
    stress_full_fetch(market)


if __name__ == "__main__":
    main()
