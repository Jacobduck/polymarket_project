"""Plot a wallet's trades over a market's YES-price history."""

from __future__ import annotations

from typing import Any, Literal, Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from polycluster.events import get_market_orderfilled_events
from polycluster.markets import Market, get_market_by_slug
from polycluster.parsing import (
    build_history_df_from_orderfilled,
    map_trades_to_history,
    normalize_trades_to_yes_df,
    parse_orderfilled_events,
)

VLineSpec = float | int | Sequence[float | int]


def _get_marker_sizes(
    df: pd.DataFrame,
    size_by_amount: bool = False,
    base_size: int = 36,
    scale: int = 18,
    max_size: int = 160,
) -> np.ndarray:
    """Compute matplotlib scatter marker sizes for a slice of trades.

    If ``size_by_amount`` is False, every marker gets ``base_size``.
    Otherwise marker area scales with ``sqrt(token_amount)``, capped at
    ``max_size``.
    """
    if df.empty:
        return np.array([])

    if not size_by_amount:
        return np.full(len(df), base_size)

    sizes = base_size + scale * np.sqrt(df["token_amount"].clip(lower=0))
    return np.minimum(sizes, max_size)


def _filter_plot_window(
    history_df: pd.DataFrame,
    mapped_trades_df: pd.DataFrame | None = None,
    start_time: int | None = None,
    end_time: int | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame | None]:
    """Restrict history (and optional trades) to ``[start_time, end_time]``.

    Both bounds are inclusive. ``None`` on either bound means unbounded
    on that side. Returns the filtered frames sorted by timestamp.
    """
    hist = history_df.copy()

    if start_time is not None:
        hist = hist[hist["timestamp"] >= int(start_time)]
    if end_time is not None:
        hist = hist[hist["timestamp"] <= int(end_time)]
    hist = hist.sort_values("timestamp").reset_index(drop=True)

    if mapped_trades_df is None:
        return hist, None

    trades = mapped_trades_df.copy()
    if start_time is not None:
        trades = trades[trades["timestamp"] >= int(start_time)]
    if end_time is not None:
        trades = trades[trades["timestamp"] <= int(end_time)]
    trades = trades.sort_values("timestamp").reset_index(drop=True)

    return hist, trades


def _compute_auto_ylim(
    history_df: pd.DataFrame,
    mapped_trades_df: pd.DataFrame | None = None,
    use_trade_price: bool = False,
    pad_frac: float = 0.08,
) -> tuple[float | None, float | None]:
    """Compute padded y-limits from the data actually visible in the window.

    Looks at ``history_df["yes_price"]`` plus, if available, the
    matching column from ``mapped_trades_df``
    (``"yes_trade_price"`` when ``use_trade_price=True``, else
    ``"history_yes_price"``).
    """
    y_vals: list[float] = []

    if history_df is not None and not history_df.empty:
        y_vals.extend(history_df["yes_price"].dropna().tolist())

    if mapped_trades_df is not None and not mapped_trades_df.empty:
        col = "yes_trade_price" if use_trade_price else "history_yes_price"
        if col in mapped_trades_df.columns:
            y_vals.extend(mapped_trades_df[col].dropna().tolist())

    if not y_vals:
        return None, None

    y_min = min(y_vals)
    y_max = max(y_vals)

    if y_min == y_max:
        pad = max(0.01, abs(y_min) * pad_frac, 0.01)
        return y_min - pad, y_max + pad

    span = y_max - y_min
    pad = span * pad_frac
    return y_min - pad, y_max + pad


def _displayed_x_to_real(
    display_start: float,
    display_end: float,
    axis_offset: int = 1764000000,
) -> tuple[float, float]:
    """Translate matplotlib display-axis coords back to unix timestamps.

    Matplotlib renders x-axis ticks with an offset for large numbers
    (e.g. ``+1.764e9``). This helper undoes that for an interactively
    selected ``[display_start, display_end]`` range.
    """
    return axis_offset + display_start, axis_offset + display_end


def plot_user_on_history(
    mapped_trades_df: pd.DataFrame,
    history_df: pd.DataFrame,
    use_trade_price: bool = False,
    figsize: tuple[float, float] = (14, 6),
    size_by_amount: bool = False,
    base_marker_size: int = 36,
    marker_scale: int = 18,
    max_marker_size: int = 160,
    buy_marker: str = "^",
    sell_marker: str = "v",
    alpha: float = 0.85,
    y_min: float | None = None,
    y_max: float | None = None,
    history_color: str = "skyblue",
    buy_color: str = "limegreen",
    sell_color: str = "tomato",
    start_time: int | None = None,
    end_time: int | None = None,
    auto_ylim: bool = True,
    ylim_pad_frac: float = 0.05,
    vlines: VLineSpec | None = None,
    vline_color: str | Sequence[str] = "orange",
    vline_style: str | Sequence[str] = "--",
    vline_width: float | Sequence[float] = 1.5,
    vline_labels: str | Sequence[str | None] | None = None,
    show: bool = True,
) -> tuple[plt.Figure, plt.Axes]:
    """Plot a wallet's BUY/SELL_YES trades over the market's YES-price history.

    The history is drawn as a step line, BUY_YES trades as up-triangles
    and SELL_YES trades as down-triangles. By default a marker is
    placed at the *history* price for the trade's timestamp; pass
    ``use_trade_price=True`` to place it at the wallet's actual
    fill price instead (useful for spotting off-market fills).

    Args:
        mapped_trades_df: Output of
            :func:`polycluster.parsing.map_trades_to_history`. Must
            contain ``timestamp``, ``yes_action``, ``yes_trade_price``
            and ``history_yes_price`` columns.
        history_df: Output of
            :func:`polycluster.parsing.build_history_df_from_orderfilled`.
            Must contain ``timestamp`` and ``yes_price``.
        use_trade_price: If True, plot trades at their actual fill
            price; otherwise plot them at the history price for that
            timestamp.
        figsize: Figure size in inches passed to ``plt.subplots``.
        size_by_amount: If True, scale marker area with the trade's
            token amount.
        base_marker_size: Marker size when ``size_by_amount=False`` and
            the floor when scaling.
        marker_scale: Multiplier on ``sqrt(token_amount)`` for scaling.
        max_marker_size: Cap on the scaled marker size.
        buy_marker: Matplotlib marker for BUY_YES trades.
        sell_marker: Matplotlib marker for SELL_YES trades.
        alpha: Marker alpha.
        y_min: Lower y-axis bound. ``None`` defers to ``auto_ylim``.
        y_max: Upper y-axis bound. ``None`` defers to ``auto_ylim``.
        history_color: Color of the history step line.
        buy_color: Color of BUY_YES markers.
        sell_color: Color of SELL_YES markers.
        start_time: Inclusive lower bound on the visible time window
            (unix seconds). ``None`` means no lower bound.
        end_time: Inclusive upper bound on the visible time window
            (unix seconds). ``None`` means no upper bound.
        auto_ylim: If True and neither ``y_min`` nor ``y_max`` is
            provided, derive y-limits from the visible data with
            ``ylim_pad_frac`` padding.
        ylim_pad_frac: Fraction of the data span to pad on each side
            when computing auto y-limits.
        vlines: One or more timestamps at which to draw vertical
            reference lines (e.g. resolution time, dispute time).
        vline_color: Single color or list of colors parallel to
            ``vlines``.
        vline_style: Single linestyle or list of linestyles parallel
            to ``vlines``.
        vline_width: Single linewidth or list of linewidths parallel
            to ``vlines``.
        vline_labels: Optional legend labels for vlines. Pass ``None``
            for an entry to suppress its legend entry.
        show: If True, call ``plt.show()`` at the end. Set to False
            when embedding the figure (e.g. saving to disk, returning
            from a notebook cell, or composing into a larger figure).

    Returns:
        The created ``(figure, axes)`` pair, so the caller can further
        customize, save, or embed the plot.
    """
    history_df, mapped_trades_df = _filter_plot_window(
        history_df=history_df,
        mapped_trades_df=mapped_trades_df,
        start_time=start_time,
        end_time=end_time,
    )

    fig, ax = plt.subplots(figsize=figsize)

    ax.step(
        history_df["timestamp"],
        history_df["yes_price"],
        where="post",
        linewidth=1.5,
        alpha=0.95,
        color=history_color,
        label="YES price history",
        zorder=1,
    )

    if mapped_trades_df is not None and not mapped_trades_df.empty:
        plot_df = mapped_trades_df.dropna(subset=["history_yes_price"]).copy()

        buys = plot_df[plot_df["yes_action"] == "BUY_YES"].copy()
        sells = plot_df[plot_df["yes_action"] == "SELL_YES"].copy()

        buy_y = buys["yes_trade_price"] if use_trade_price else buys["history_yes_price"]
        sell_y = (
            sells["yes_trade_price"] if use_trade_price else sells["history_yes_price"]
        )

        buy_sizes = _get_marker_sizes(
            buys,
            size_by_amount=size_by_amount,
            base_size=base_marker_size,
            scale=marker_scale,
            max_size=max_marker_size,
        )
        sell_sizes = _get_marker_sizes(
            sells,
            size_by_amount=size_by_amount,
            base_size=base_marker_size,
            scale=marker_scale,
            max_size=max_marker_size,
        )

        ax.scatter(
            buys["timestamp"], buy_y,
            s=buy_sizes, marker=buy_marker, alpha=alpha,
            color=buy_color, edgecolors="black", linewidths=0.5,
            label="Buy YES (normalized)", zorder=3,
        )
        ax.scatter(
            sells["timestamp"], sell_y,
            s=sell_sizes, marker=sell_marker, alpha=alpha,
            color=sell_color, edgecolors="black", linewidths=0.5,
            label="Sell YES (normalized)", zorder=3,
        )

    if vlines is not None:
        vlines_list = list(vlines) if isinstance(vlines, (list, tuple)) else [vlines]
        n = len(vlines_list)
        colors_list = (
            list(vline_color) if isinstance(vline_color, (list, tuple))
            else [vline_color] * n
        )
        styles_list = (
            list(vline_style) if isinstance(vline_style, (list, tuple))
            else [vline_style] * n
        )
        widths_list = (
            list(vline_width) if isinstance(vline_width, (list, tuple))
            else [vline_width] * n
        )
        labels_list = (
            list(vline_labels) if isinstance(vline_labels, (list, tuple))
            else [vline_labels] * n
        )

        for t, c, ls, lw, lbl in zip(
            vlines_list, colors_list, styles_list, widths_list, labels_list
        ):
            ax.axvline(
                x=t, color=c, linestyle=ls, linewidth=lw, label=lbl, zorder=2,
            )

    ax.set_xlabel("Timestamp")
    ax.set_ylabel("YES price")
    ax.set_title("User trades over YES-token price history")
    ax.grid(True, alpha=0.3)
    ax.legend()

    if y_min is not None or y_max is not None:
        cur_ymin, cur_ymax = ax.get_ylim()
        ax.set_ylim(
            cur_ymin if y_min is None else y_min,
            cur_ymax if y_max is None else y_max,
        )
    elif auto_ylim:
        auto_ymin, auto_ymax = _compute_auto_ylim(
            history_df,
            mapped_trades_df,
            use_trade_price=use_trade_price,
            pad_frac=ylim_pad_frac,
        )
        if auto_ymin is not None and auto_ymax is not None:
            ax.set_ylim(auto_ymin, auto_ymax)

    plt.tight_layout()

    if show:
        plt.show()

    return fig, ax


def plot_user_on_market(
    wallet: str,
    market: Market | str,
    *,
    timeout: float = 30.0,
    page_size: int = 100,
    min_window: int = 60,
    checkpointed: bool = False,
    initial_chunk_seconds: int = 1800,
    history_aggregate: Literal["last", "first", "mean"] | None = "last",
    map_tolerance: int | None = None,
    verbose: bool = False,
    **plot_kwargs: Any,
) -> tuple[plt.Figure, plt.Axes]:
    """Fetch a wallet's trades for a market and plot them on the price history.

    End-to-end convenience wrapper that chains the full pipeline:

    1. :func:`~polycluster.markets.get_market_by_slug` (if a slug is
       passed) to resolve the market's token IDs and outcomes.
    2. :func:`~polycluster.events.get_market_orderfilled_events` to
       fetch every OrderFilled event for the market.
    3. :func:`~polycluster.parsing.build_history_df_from_orderfilled`
       to build the YES-price history.
    4. :func:`~polycluster.parsing.parse_orderfilled_events` +
       :func:`~polycluster.parsing.normalize_trades_to_yes_df` to
       extract the wallet's trades and put them on a YES axis.
    5. :func:`~polycluster.parsing.map_trades_to_history` to align each
       trade to the nearest history price.
    6. :func:`plot_user_on_history` to draw the chart.

    Wallet matching is case-insensitive.

    Args:
        wallet: Wallet address to plot.
        market: Either a :class:`~polycluster.markets.Market` (if
            already fetched) or a slug string.
        timeout: HTTP timeout in seconds for the underlying gamma-api
            and subgraph requests.
        page_size: Subgraph page size.
        min_window: Smallest sub-window for the recursive fetch (only
            used when ``checkpointed=False``).
        checkpointed: If True, use the resumable checkpointed fetcher.
        initial_chunk_seconds: Chunk size for the checkpointed path.
        history_aggregate: How to collapse fills sharing a timestamp
            when building the history dataframe. See
            :func:`~polycluster.parsing.build_history_df_from_orderfilled`.
        map_tolerance: Maximum seconds between a trade and its matched
            history row. ``None`` means no tolerance limit. See
            :func:`~polycluster.parsing.map_trades_to_history`.
        verbose: If True, print fetch progress to stdout.
        **plot_kwargs: Forwarded to :func:`plot_user_on_history`. Use
            this to control styling (``figsize``, ``vlines``,
            ``start_time``/``end_time``, ``use_trade_price``,
            ``show``, etc.). See that function for the full list.

    Returns:
        The ``(figure, axes)`` pair from :func:`plot_user_on_history`.
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

    history_df = build_history_df_from_orderfilled(
        events,
        yes_token=market.yes_token_id,
        no_token=market.no_token_id,
        aggregate_same_timestamp=history_aggregate,
    )

    parsed = parse_orderfilled_events(
        events,
        market.token_ids,
        market.outcomes,
    )

    user_trades = parsed["wallet_trades"].get(wallet.lower(), [])
    trades_df = normalize_trades_to_yes_df(
        user_trades,
        yes_token=market.yes_token_id,
        no_token=market.no_token_id,
    )

    mapped_df = map_trades_to_history(
        trades_df, history_df, tolerance=map_tolerance
    )

    return plot_user_on_history(mapped_df, history_df, **plot_kwargs)
