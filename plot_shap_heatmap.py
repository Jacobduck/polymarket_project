#!/usr/bin/env python3
"""Model x 30-feature SHAP heatmap for one (wallet, market) pair.

Reuses audit_pair.py's pipeline to score every discovered model on the pair,
captures the FULL per-feature SHAP contribution (top_k=0 = all features), then
plots the top-N models (ranked by insider score) against the 30 documented
features (20 behavioral + 10 metadata from feature_reference.md).

Cells are signed SHAP contributions toward the insider class in each model's
own margin space (red = pushes ↑insider, blue = ↓insider). Blank/hatched =
the model does not use that feature.
"""
from __future__ import annotations

import argparse
import logging

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm

import audit_pair as ap
from polycluster.explain import explain_prediction

# The 30 documented features, in feature_reference.md order.
TOP20_BEHAV = [
    "time_to_major_move_pct_of_market",
    "unrealized_edge_per_open_token_sqrt",
    "buy_markout_adj_grosswt_h0p030_lb0p200",
    "major_move_time_remaining_pct",
    "realized_pnl_per_closing_trade",
    "buy_markout_adj_grosswt_h0p030_lb0p010",
    "buy_markout_adj_grosswt_h0p070_lb0p150",
    "n_low_buy_building_trades",
    "adaptive_markout_raw_grosswt_h0p070",
    "adaptive_plus_realized_alpha0p5_h0p070",
    "adaptive_plus_realized_alpha0p25_h0p030",
    "realized_pnl_total",
    "unrealized_edge_per_open_token",
    "realized_edge_grosswt",
    "buy_markout_adj_mean_h0p200_lb0p200",
    "buy_markout_adj_mean_h0p200_lb0p030",
    "derisk_0p75_time_position_mean",
    "adaptive_markout_raw_grosswt_h0p150",
    "adaptive_markout_raw_grosswt_h0p030",
    "low_buy_building_cash_frac",
]
TOP10_META = [
    "median_bet_size_usdc",
    "herfindahl_index_markets",
    "wallet_to_market_age_ratio",
    "time_since_first_deposit_to_first_trade_seconds",
    "avg_bet_size_usdc",
    "wallet_minus_market_age_seconds",
    "wallet_age_seconds",
    "winrate",
    "total_bet_size_usdc",
    "n_markets_traded",
]
FEATURES30 = TOP20_BEHAV + TOP10_META


def collect(wallet: str, slug: str, api_key: str | None):
    """Score every model on the pair; return list of result dicts with full SHAP."""
    tags = [t for t in ap.discover_model_tags()]
    metas = {}
    for t in tags:
        try:
            metas[t] = ap.load_meta(t)
        except Exception:
            pass
    tags = [t for t in tags if t in metas]

    meta_cols = set(ap.METADATA_FEATURE_COLUMNS)
    all_feats: set[str] = set()
    for m in metas.values():
        all_feats.update(m.get("features", []))
    behavioral_needed = {f for f in all_feats if f not in meta_cols}
    metadata_needed = {f for f in all_feats if f in meta_cols}

    market, events = ap.load_market_and_events(slug)
    history_df = ap.build_history_df_from_orderfilled(
        events, yes_token=market.yes_token_id, no_token=market.no_token_id)
    parsed = ap.parse_orderfilled_events(events, market.token_ids, market.outcomes)
    wallet_trades = parsed["wallet_trades"].get(wallet, [])
    flagged = ap.load_flagged_wallets()

    row, ctx = ap.build_feature_row(
        wallet, slug, market, history_df, wallet_trades,
        behavioral_needed, metadata_needed, api_key, flagged)

    results = []
    for tag in tags:
        try:
            model, _ = ap.load_model(tag)
        except Exception:
            continue
        cal = ap.load_calibrator(tag)
        meta = metas[tag]
        feats = list(meta["features"])
        X = pd.DataFrame([{c: row.get(c, np.nan) for c in feats}])[feats].astype(float)
        if hasattr(model, "predict_proba"):
            raw = float(model.predict_proba(X)[:, 1][0])
            prob = float(ap.apply_calibrator(cal, [raw])[0]) if cal is not None else raw
            thr = meta.get("threshold") or 0.5
            score, kind, flag = prob, "prob", bool(prob >= thr)
        elif hasattr(model, "decision_function"):
            raw = float(model.decision_function(X)[0])
            score, kind, flag = raw, "anomaly", bool(raw < 0)
        else:
            continue
        expl = explain_prediction(model, X, feats, top_k=0,
                                  logger=logging.getLogger())
        contrib = {a["feature"]: a["contribution"] for a in expl["attributions"]}
        results.append({
            "tag": tag, "score": score, "kind": kind, "flag": flag,
            "label": ap.LABELS.get(tag, tag), "contrib": contrib,
        })
    return results, ctx, row


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--wallet", default=ap.DEFAULT_WALLET)
    p.add_argument("--market", default=ap.DEFAULT_MARKET)
    p.add_argument("--api-key", default=None)
    p.add_argument("--top-n", type=int, default=10)
    p.add_argument("--only", choices=["flagged", "notflagged", "all"],
                   default="flagged",
                   help="restrict to models that flagged / did not flag the insider")
    p.add_argument("--out", default="audit_out/shap_heatmap_top10.png")
    args = p.parse_args()

    logging.basicConfig(level=logging.WARNING)
    wallet = args.wallet.lower()
    api_key = ap.resolve_api_key(args.api_key)

    results, ctx, row = collect(wallet, args.market, api_key)
    # Optionally restrict to (non-)flaggers of this insider.
    if args.only == "flagged":
        results = [r for r in results if r["flag"]]
    elif args.only == "notflagged":
        results = [r for r in results if not r["flag"]]
    # Rank by insider score (probability models first via kind, then value).
    results.sort(key=lambda r: (r["kind"] == "prob", r["score"]), reverse=True)
    top = results[: args.top_n]

    # Build the matrix: rows = top models, cols = 30 features.
    M = np.full((len(top), len(FEATURES30)), np.nan)
    for i, r in enumerate(top):
        for j, f in enumerate(FEATURES30):
            if f in r["contrib"]:
                M[i, j] = r["contrib"][f]

    finite = M[np.isfinite(M)]
    vmax = np.nanpercentile(np.abs(finite), 98) if finite.size else 1.0
    vmax = max(vmax, 1e-6)
    norm = TwoSlopeNorm(vmin=-vmax, vcenter=0.0, vmax=vmax)

    fig, ax = plt.subplots(figsize=(20, 9))
    cmap = plt.cm.RdBu_r.copy()
    cmap.set_bad("#e8e8e8")  # missing feature
    im = ax.imshow(M, aspect="auto", cmap=cmap, norm=norm)

    ax.set_xticks(range(len(FEATURES30)))
    ax.set_xticklabels(
        [f if len(f) <= 34 else f[:33] + "…" for f in FEATURES30],
        rotation=90, fontsize=8)
    ylabels = [f"{r['tag']}\n{r['kind']}={r['score']:.3f} {'FLAG' if r['flag'] else '·'}"
               for r in top]
    ax.set_yticks(range(len(top)))
    ax.set_yticklabels(ylabels, fontsize=8)

    # Separator between the 20 behavioral and 10 metadata columns.
    ax.axvline(len(TOP20_BEHAV) - 0.5, color="k", lw=2)
    ax.text(len(TOP20_BEHAV) / 2 - 0.5, -1.2, "BEHAVIORAL (20)",
            ha="center", va="bottom", fontsize=11, fontweight="bold")
    ax.text(len(TOP20_BEHAV) + len(TOP10_META) / 2 - 0.5, -1.2, "METADATA (10)",
            ha="center", va="bottom", fontsize=11, fontweight="bold")

    # Annotate each finite cell with its value.
    for i in range(M.shape[0]):
        for j in range(M.shape[1]):
            v = M[i, j]
            if np.isfinite(v):
                ax.text(j, i, f"{v:+.2f}", ha="center", va="center",
                        fontsize=6.2,
                        color="white" if abs(v) > 0.55 * vmax else "black")

    cbar = fig.colorbar(im, ax=ax, fraction=0.02, pad=0.01)
    cbar.set_label("SHAP contribution toward insider (model margin space)\n"
                   "red = ↑insider   ·   blue = ↓insider", fontsize=9)
    mode_txt = {"flagged": "that FLAGGED", "notflagged": "that did NOT flag",
                "all": "scored on"}[args.only]
    ax.set_title(
        f"Top {len(top)} models {mode_txt} the d4vd insider × 30 features "
        f"({wallet[:10]}… · {args.market})\n"
        f"gray = feature not used by that model · magnitudes are per-model "
        f"(compare patterns across a row, not raw sizes across rows)",
        fontsize=12)
    fig.tight_layout()
    fig.savefig(args.out, dpi=140, bbox_inches="tight")
    print("wrote", args.out)
    print("\ntop models (rank by insider score):")
    for r in top:
        print(f"  {r['kind']}={r['score']:.4f} {'FLAG' if r['flag'] else '   '}  {r['tag']}")


if __name__ == "__main__":
    main()
