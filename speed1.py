from __future__ import annotations

"""Speed test A: page_size=1000, min_window=600.

Bigger pages + less aggressive recursive splitting (stops halving the
window once it's <= 600s). Fewer queries to Goldsky, but each one does
more server-side work — higher risk of statement timeouts on dense
windows.

Fetches the full market lifetime, both tokens, both sides (maker+taker).
No local caching: polycluster.events always hits Goldsky live.
"""


""""caffeinate -i .venv/bin/python speed1.py 2>&1 | tee speed1.log && \
caffeinate -i .venv/bin/python speed2.py 2>&1 | tee speed2.log"""

"""speed1 with 1000 and 600 takes 170s"""



import time

from polycluster.events import get_market_orderfilled_events
from polycluster.markets import get_market_by_slug


MARKET_SLUG = "will-zelenskyy-wear-a-suit-before-july"

PAGE_SIZE = 1000
MIN_WINDOW = 60


def main() -> None:
    print(f"[speed1] params: page_size={PAGE_SIZE}, min_window={MIN_WINDOW}")
    print(f"[speed1] resolving slug={MARKET_SLUG!r}...")
    t0 = time.time()
    market = get_market_by_slug(MARKET_SLUG)
    print(
        f"[speed1] resolved in {time.time() - t0:.2f}s: "
        f"{len(market.token_ids)} token(s), closed={market.closed}, "
        f"window=[{market.start_ts}, {market.end_ts}] "
        f"({(market.end_ts - market.start_ts) / 86400:.1f}d)"
    )

    print("[speed1] starting fetch (full lifetime, both tokens, both sides)...")
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
    print(f"[speed1] RESULT: {len(events)} events in {elapsed:.2f}s")
    print(f"[speed1] params: page_size={PAGE_SIZE}, min_window={MIN_WINDOW}")
    print("=" * 60)


if __name__ == "__main__":
    main()
