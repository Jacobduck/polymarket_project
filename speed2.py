"""Speed test B: page_size=100, min_window=60.

Smaller pages + more aggressive recursive splitting (halves the window
all the way down to 60s). More queries to Goldsky, but each one does
less server-side work — much safer against statement timeouts.

Fetches the full market lifetime, both tokens, both sides (maker+taker).
No local caching: polycluster.events always hits Goldsky live.
"""

from __future__ import annotations

import time

from polycluster.events import get_market_orderfilled_events
from polycluster.markets import get_market_by_slug


MARKET_SLUG = "will-kamala-harris-win-the-2024-us-presidential-election"

PAGE_SIZE = 100
MIN_WINDOW = 60


def main() -> None:
    print(f"[speed2] params: page_size={PAGE_SIZE}, min_window={MIN_WINDOW}")
    print(f"[speed2] resolving slug={MARKET_SLUG!r}...")
    t0 = time.time()
    market = get_market_by_slug(MARKET_SLUG)
    print(
        f"[speed2] resolved in {time.time() - t0:.2f}s: "
        f"{len(market.token_ids)} token(s), closed={market.closed}, "
        f"window=[{market.start_ts}, {market.end_ts}] "
        f"({(market.end_ts - market.start_ts) / 86400:.1f}d)"
    )

    print("[speed2] starting fetch (full lifetime, both tokens, both sides)...")
    t_start = time.time()
    events = get_market_orderfilled_events(
        market,
        page_size=PAGE_SIZE,
        min_window=MIN_WINDOW,
        verbose=True,
    )
    elapsed = time.time() - t_start

    print()
    print("=" * 60)
    print(f"[speed2] RESULT: {len(events)} events in {elapsed:.2f}s")
    print(f"[speed2] params: page_size={PAGE_SIZE}, min_window={MIN_WINDOW}")
    print("=" * 60)


if __name__ == "__main__":
    main()
