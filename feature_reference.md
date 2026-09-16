# Feature Reference — Insider-Detection Models

This document describes the most influential features consumed by the insider-detection
models. There are two families:

- **Behavioral features** — computed from the wallet's on-market order fills
  (`OrderFilled` events → `cache/training_data.parquet`) by `polycluster/features.py`.
  They describe *how* a wallet trades one specific market: entry timing, markout
  (does price move the trader's way after they act), realized/unrealized edge, and
  accumulation patterns.
- **Metadata features** — computed from the data-api activity feed
  (`cache/metadata_features.parquet`) by `polycluster/metadata.py`. They describe the
  wallet's *broader footprint*: age, sizing, market concentration, and funder graph.

**How "top" was ranked.** Features are ordered by the aggregate `Σ|SHAP contribution|`
across all 35 models in the latest d4vd audit
(`audit_out/audit_willd4vdbethe1search_c1259ddd_20260715_100334_shap.csv`). This measures
how much each feature moved model scores *on that pair*, so it reflects real influence but
is anchored to one audited case; a different wallet could reorder the tail.

**Anti-look-ahead.** Behavioral features may reference the market's eventual outcome
(`yes_wins`, settle price) because scoring happens *after* resolution — they are
descriptive, not predictive-in-time. Metadata *outcome* features (`winrate`, `pnl_sharpe`)
are instead pinned to the wallet's entry moment (`as_of_ts` = first trade on the market)
to avoid label leakage; outcome-neutral sizing features span the wallet's full lifetime.

---

## Naming conventions (behavioral markout features)

Markout features encode their parameters in the tag suffix:

- `h0p030` → **horizon** = 0.030 × market-lifetime (how far *forward* the post-trade
  price move is measured).
- `lb0p200` → **lookback** = 0.200 × market-lifetime (how far *back* the pre-trade trend
  is measured, used for the momentum penalty / contrarian reward).
- `grosswt` → **token-weighted** and normalized by the wallet's gross lifetime tokens
  (a size-aware average over qualifying trades). `mean`/`median` are the unweighted
  per-trade central tendencies instead.
- `buy_` / `sell_` → restricted to buys that *build* (or sells that *reduce*) inventory in
  the trade's own direction.
- `adj` → "adjusted": the raw post-trade move is penalized for pre-trade momentum and
  rewarded for contrarian entries; `latespike_adj` additionally subtracts gains earned
  only from a detected terminal price spike.

Positive markout = price moved in the trader's favor after they traded (skill/edge signal).

---

## Top 20 Behavioral Features

### 1. `time_to_major_move_pct_of_market`
**What:** Signed gap, as a fraction of market lifetime, between the wallet's first entry and
the market's first "major move." The single strongest driver across models — an insider who
buys *just before* a large price jump scores extreme here.
**How:** `detect_major_move` scans the YES-price history for the earliest timestamp where the
price moves ≥ 0.15 over the next 3600 s (`|future − current| ≥ threshold`). The feature is the
relative gap from `first_entry_time` to that `major_move_time`, normalized by market duration
(negative = entered before the move).

### 2. `unrealized_edge_per_open_token_sqrt`
**What:** Paper edge on positions still open at the end, i.e. how well-priced the wallet's
un-exited bets were versus the settlement outcome — sqrt-weighted so a few giant lots don't
dominate.
**How:** For each final open lot, `sign(qty)·(settle_price − entry_price)/price_range`,
weighted by `sqrt(|qty|)`, summed and divided by `sqrt(gross_lifetime_tokens)`
(`_compute_unrealized_edge_over_gross_lifetime_tokens`). `settle_price` is the resolved
outcome (1/0); `price_range` = max−min YES price in history.

### 3. `buy_markout_adj_grosswt_h0p030_lb0p200`
**What:** Token-weighted adjusted markout on inventory-building buys, measured 3% of lifetime
forward with a 20%-lifetime lookback trend. High = the wallet's buys were quickly followed by
favorable price moves (short-horizon foresight).
**How:** `compute_markout_features`, buy branch. Per buy: `price_weight·post_move
− 0.10·pos_pretrend + 0.30·neg_pretrend`, where `post_move = future − current`,
`price_weight = min(1/current_price^1.5, 8)` (rewards moves from extreme-cheap entries).
Weighted by `token_amount/gross_tokens` and summed. Only buys with `inventory_before ≥ 0`.

### 4. `major_move_time_remaining_pct`
**What:** Where the market's major move sits in its lifetime, measured as fraction of time
*remaining* after the move. Locates the regime shift the insider is trading around.
**How:** `_safe_time_remaining_pct(major_move_time, market_start, market_end)` — 1 minus the
move's position along the lifetime.

### 5. `realized_pnl_per_closing_trade`
**What:** Average realized profit per position-closing trade — how much the wallet actually
banked each time it took money off the table.
**How:** FIFO matching of BUY/SELL lots (`compute_realized_pnl`-style loop): each closing
match contributes `(exit − entry)·matched_tokens`; summed into `realized_pnl_total` and
divided by the count of trades that realized anything (`n_closing`).

### 6. `buy_markout_adj_grosswt_h0p030_lb0p010`
**What:** Same as #3 but with a very short 1%-lifetime lookback — near-instantaneous
foresight, less momentum context.
**How:** `compute_markout_features`, buy branch, tag `h0.030_lb0.010`.

### 7. `buy_markout_adj_grosswt_h0p070_lb0p150`
**What:** Buy markout at a medium 7%-forward horizon with 15% lookback — captures edge that
plays out slightly later than the 3% horizon.
**How:** `compute_markout_features`, buy branch, tag `h0.070_lb0.150`.

### 8. `n_low_buy_building_trades`
**What:** Count of trades that *buy cheap while adding to* a long position — i.e. accumulating
a low-priced YES side rather than flattening. A hallmark of conviction accumulation.
**How:** `compute_contrarian_price_features`: `low_buy_mask = BUY & norm_price ≤ low_threshold`,
intersected with `inventory_before ≥ 0`; `n_low_buy_building_trades = mask.sum()`.

### 9. `adaptive_markout_raw_grosswt_h0p070`
**What:** Lot-level markout at 7% horizon, measured from each lot's entry but capped by the
trader's *own* derisking times (doesn't credit gains after they'd already sold down).
Token-weighted, raw (before the late-spike penalty).
**How:** `compute_adaptive_markout_features`: FIFO lots; per open lot,
`future_time = entry + 0.07·lifetime`, clamped between its derisk-25% and derisk-50% times;
`scaled_post_move = (future_price − entry_price)/price_range` (sign-adjusted by side),
weighted by `open_tokens/gross_tokens`, summed.

### 10. `adaptive_plus_realized_alpha0p5_h0p070`
**What:** Blend of the late-spike-adjusted 7% adaptive markout **plus** 0.5× realized edge —
rewards wallets that are strong on *both* paper and banked edge.
**How:** `adjusted_markout_grosswt + 0.5·realized_edge_grosswt`, where `adjusted =
raw_markout − late_spike_edge` (see #17's family).

### 11. `adaptive_plus_realized_alpha0p25_h0p030`
**What:** Same blend at 3% horizon with a lighter 0.25× realized-edge weight.
**How:** `adjusted_markout_grosswt(h0.030) + 0.25·realized_edge_grosswt`.

### 12. `realized_pnl_total`
**What:** Total realized (banked) profit across all closing trades on the market.
**How:** Sum of all FIFO close-match PnL pieces `(exit − entry)·matched_tokens` (see #5).

### 13. `unrealized_edge_per_open_token`
**What:** Linear-weighted twin of #2 — paper edge on open lots, weighted by raw lot size
(so big lots count fully).
**How:** `Σ qty·(settle − entry)/price_range` over final open lots, divided by
`gross_lifetime_tokens`.

### 14. `realized_edge_grosswt`
**What:** Realized edge per matched token, normalized to price range and token-weighted —
"quality" of banked profit independent of raw dollar size.
**How:** `compute_adaptive_markout_features`: over FIFO realized matches,
`Σ (raw_pnl_per_token/price_range)·(matched_tokens/gross_tokens)`.

### 15. `buy_markout_adj_mean_h0p200_lb0p200`
**What:** *Unweighted mean* adjusted buy markout at a long 20% horizon / 20% lookback —
average per-trade edge over a slow horizon, ignoring size.
**How:** `compute_markout_features`, `np.mean(buy_scores)` for tag `h0.200_lb0.200`.

### 16. `buy_markout_adj_mean_h0p200_lb0p030`
**What:** Same 20% horizon but short 3% lookback — long-horizon edge with little momentum
context.
**How:** `compute_markout_features`, `np.mean(buy_scores)` for tag `h0.200_lb0.030`.

### 17. `derisk_0p75_time_position_mean`
**What:** Average point in market lifetime (0–1) at which lots are 75% closed — *when* the
wallet takes most of its risk off. Late derisking near resolution can indicate outcome
knowledge.
**How:** `compute_adaptive_markout_features`: for each lot, `_find_lot_derisk_time(lot, 0.75)`
= first time cumulative closes reach 75% of original tokens; expressed as
`(t − market_start)/lifetime` and averaged across lots.

### 18. `adaptive_markout_raw_grosswt_h0p150`
**What:** Lot-level raw adaptive markout at a longer 15% horizon (see #9 for mechanics).
**How:** `compute_adaptive_markout_features`, tag `h0.150`.

### 19. `adaptive_markout_raw_grosswt_h0p030`
**What:** Lot-level raw adaptive markout at a short 3% horizon.
**How:** `compute_adaptive_markout_features`, tag `h0.030`.

### 20. `low_buy_building_cash_frac`
**What:** Fraction of total buy cash spent on cheap, inventory-building buys — the *dollar*
intensity of low-price accumulation (companion to the count in #8).
**How:** `compute_contrarian_price_features`:
`cash(low_buy_building_mask) / total_buy_cash`.

---

## Top 10 Metadata Features

> Sizing features (`avg/median/total_bet_size_usdc`, `lifetime_volume_share_on_market`) are
> computed over the wallet's **full lifetime** TRADE history (`all_trades`). Concentration,
> ages, and outcome features are pinned to the entry moment `as_of_ts`
> (`pre_trades = activity[timestamp ≤ as_of_ts]`). This split is the retrain fix: sizes carry
> no label information, so widening their window is not look-ahead.

### 1. `median_bet_size_usdc`
**What:** Median USDC size of the wallet's trades across its lifetime. Small median → looks
like retail; typically pushes scores *down* even for insiders who place many small fills.
**How:** `all_trades["usdcSize"].median()`.

### 2. `herfindahl_index_markets`
**What:** Concentration of the wallet's trading across markets *at entry* (HHI ∈ (0,1]).
`1.0` = the wallet has only ever touched this one market — a fresh single-market wallet is
suspicious.
**How:** `pre_trades` volume per slug → shares → `Σ share²`.

### 3. `wallet_to_market_age_ratio`
**What:** Wallet age relative to how long the market has been open at entry. A tiny ratio =
a brand-new wallet trading a long-lived market (just-in-time account).
**How:** `wallet_age_seconds / max(market_age_seconds, 1.0)`.

### 4. `time_since_first_deposit_to_first_trade_seconds`
**What:** Seconds from the wallet's first-ever USDC deposit to its first trade on this market.
Very small = funded and traded almost immediately (single-purpose wallet). Equals
`wallet_age_seconds` here.
**How:** `as_of_ts − first_deposit_ts`, where `first_deposit_ts` comes from
`get_wallet_first_usdc_deposit` (Polygonscan/cached).

### 5. `avg_bet_size_usdc`
**What:** Mean lifetime trade size. Large mean with small median flags a "many small fills +
a few whales" pattern.
**How:** `all_trades["usdcSize"].mean()`.

### 6. `wallet_minus_market_age_seconds`
**What:** Absolute (not ratio) difference between wallet age and market age at entry.
Strongly negative = wallet much younger than the market.
**How:** `wallet_age_seconds − market_age_seconds`.

### 7. `wallet_age_seconds`
**What:** Age of the wallet (first deposit → entry) in seconds. Fresh wallets (seconds/minutes
old) are a core insider signal.
**How:** `as_of_ts − first_deposit_ts`.

### 8. `winrate`
**What:** Fraction of the wallet's *prior* markets that were net-profitable, measured **as of
entry** (not after resolution). Fresh insider wallets have `0` here because they haven't
resolved any prior positions yet — deliberately leak-free.
**How:** `_per_market_pnl(pre)` = `Σ SELL + Σ REDEEM − Σ BUY` per slug over
`timestamp ≤ as_of_ts`; `winrate = (pnls > 0).mean()`.

### 9. `total_bet_size_usdc`
**What:** Total USDC the wallet has ever staked across trades. The retrain fix's headline
feature — recovers whales that enter with a small probe then load up (d4vd: $42,371 total vs
$24 median).
**How:** `all_trades["usdcSize"].sum()`.

### 10. `n_markets_traded`
**What:** Number of distinct markets the wallet had traded as of entry. `1` reinforces the
single-purpose-wallet signal.
**How:** `len(unique slugs in pre_trades)`.
