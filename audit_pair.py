"""audit_pair.py — score ONE (wallet, market) pair against every insider model.

You found a wallet you believe is an insider on a given market. This script
asks the rest of the zoo: *would the other models flag it too, and why?* It
loads the curated, de-duplicated set of trained models (xgb / catboost / rf /
lightgbm / logreg / isoforest — the same set ``compare_models.py`` ranks, minus
exact duplicates and the un-recomputable 319-feature model), computes the exact
feature vector this pair needs, and for each model logs:

  * the model's probability / anomaly score for the pair,
  * whether it flags the pair as an insider (at the model's own saved
    threshold), and
  * the per-prediction SHAP attribution — which features drove *this* score,
    signed and ranked — via :func:`polycluster.explain.explain_prediction`.

Feature vector (built once, shared by all models):
  * behavioral features  -> ``compute_market_user_features`` (needs the market's
    OrderFilled events; reused from ``cache/events/<slug>.pkl`` when present).
  * metadata features    -> ``compute_metadata_features`` (Polymarket data-api +
    Polygonscan/Etherscan). Needs internet + ``POLYGONSCAN_API_KEY`` for the
    funder-graph signals; without a key those are NaN (models tolerate NaN).
Each model then selects its own feature subset (in its trained order); anything
a model wants that we didn't compute is passed as NaN.

Runs fine on Insomnia. The single pair is cheap (no process pool needed); the
only slow part is the metadata network calls, which are cached under
``cache/metadata`` after the first run.

Example:
  export POLYGONSCAN_API_KEY=...
  python audit_pair.py \
    --wallet 0xc1259ddd92f58aded52d029e37e3fbbebafaa4c1 \
    --market will-d4vd-be-the-1-searched-person-on-google-this-year

Outputs (in --out-dir, default "."):
  audit_<slug8>_<wallet8>_<ts>.log          human-readable, one block per model
  audit_<slug8>_<wallet8>_<ts>_models.csv   one row per model (prob/flag/method)
  audit_<slug8>_<wallet8>_<ts>_shap.csv      long form: model x feature x contrib
"""

from __future__ import annotations

import argparse
import glob
import json
import logging
import os
import pickle
import sys
import time
from pathlib import Path

os.environ.setdefault("MPLBACKEND", "Agg")

import joblib
import numpy as np
import pandas as pd
import xgboost as xgb

from polycluster.events import get_market_orderfilled_events
from polycluster.explain import HAS_SHAP, explain_prediction, format_attributions
from polycluster.features import compute_market_user_features
from polycluster.markets import get_market_by_slug
from polycluster.metadata import METADATA_FEATURE_COLUMNS, compute_metadata_features
from polycluster.parsing import (
    build_history_df_from_orderfilled,
    parse_orderfilled_events,
)

CACHE = Path("cache")
EVENTS_DIR = CACHE / "events"
MODEL_DIR = CACHE / "models"
KNOWN_INSIDER_JSON = CACHE / "known_insider_pairs.json"

DEFAULT_WALLET = "0xc1259ddd92f58aded52d029e37e3fbbebafaa4c1"
DEFAULT_MARKET = "will-d4vd-be-the-1-searched-person-on-google-this-year"

# Pretty labels, mirrored from compare_models.py (fallback = tag).
LABELS = {
    "xgb_insider_meta_v7": "xgb 20b+10m v7",
    "xgb_insider_meta_v5": "xgb 10b+17m v5",
    "xgb_insider_meta10_v6": "xgb meta-only v6",
    "rf_insider_meta_v5": "rf 10b+10m v5 (tuned)",
    "rf_insider_meta_v4": "rf 10b+17m v4 (axiom)",
    "rf_insider_meta_v3": "rf 10b+17m v3",
    "xgb_insider_20feat_v2": "xgb 20feat v2",
    "xgb_insider_20feat_v2_mstr": "xgb 20feat v2 (mstr)",
    "xgb_insider_20feat_v2_stable": "xgb 20feat v2 (stable)",
    "xgb_insider_20feat_v2_axiom": "xgb 20feat v2 (axiom)",
    "xgb_insider_10feat_v2": "xgb 10feat v2",
    "xgb_insider_30feat_v2": "xgb 30feat v2",
    "xgb_insider_14feat": "xgb 14feat",
    "xgb_insider_50feat": "xgb 50feat",
    "cb_insider_30feat_v2": "cb 30feat v2",
    "cb_insider_10feat_v2": "cb 10feat v2",
    "cb_insider_10feat_v2_d2": "cb 10feat v2 (d2)",
    "rf_insider_10feat_v2": "rf 10feat v2",
    "lgbm_insider_30feat": "lgbm 30feat",
    "lr_insider_30feat": "logreg 30feat",
    "iso_insider_30feat": "isoforest 30feat",
    "stack_insider_30feat": "stack 30feat",
    "rf_insider_25ir": "rf 25ir-hidden (20f)",
    "rf_insider_0ir": "rf 0ir-hidden (20f)",
    "xgb_insider_25ir": "xgb 25ir-hidden (30f)",
    "xgb_insider_0ir": "xgb 0ir-hidden (30f)",
}

# Exact-duplicate / superseded artifacts (identical metrics to a kept one),
# plus the 319-feature "all" model whose feature set we can't recompute for a
# single pair. Mirrors compare_models.SKIP_TAGS.
SKIP_TAGS = {
    "xgb_insider_20feat_v2_nocal",
    "cb_insider_30feat",
    "rf_insider_30feat",
    "xgb_insider_30feat",
    "xgb_insider",  # 319-feature model: needs features we don't recompute here
}

# Iran-experiment models: no "_latest" meta copies -> load newest timestamped.
NEW_TAGS = ["rf_insider_25ir", "rf_insider_0ir", "xgb_insider_25ir", "xgb_insider_0ir"]

DEFAULT_FLAG_THRESHOLD = 0.5  # only used if a proba model's meta omits one
SHAP_TOP_K = 12


# --------------------------------------------------------------------------- #
# Logging
# --------------------------------------------------------------------------- #
def configure_logging(out_dir: Path, stem: str) -> tuple[str, str]:
    ts = time.strftime("%Y%m%d_%H%M%S")
    log_path = str(out_dir / f"audit_{stem}_{ts}.log")
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    for h in list(logger.handlers):
        logger.removeHandler(h)
    fmt = logging.Formatter("%(asctime)s [audit] %(message)s", datefmt="%H:%M:%S")
    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(fmt)
    logger.addHandler(sh)
    fh = logging.FileHandler(log_path)
    fh.setFormatter(fmt)
    logger.addHandler(fh)
    return log_path, ts


# --------------------------------------------------------------------------- #
# Model discovery + loading (all families)
# --------------------------------------------------------------------------- #
def algo_of(tag: str) -> str:
    base = tag[len("retrained_"):] if tag.startswith("retrained_") else tag
    for k in ("xgb", "cb", "rf", "lgbm", "lr", "iso", "stack"):
        if base.startswith(k):
            return k
    return "other"


def discover_model_tags() -> list[str]:
    """Curated, de-duplicated set of model tags to score against."""
    tags: list[str] = []
    for p in sorted(glob.glob(str(MODEL_DIR / "*_latest.meta.json"))):
        tag = os.path.basename(p).replace("_latest.meta.json", "")
        if tag in SKIP_TAGS:
            continue
        tags.append(tag)
    for tag in NEW_TAGS:
        if tag not in tags and tag not in SKIP_TAGS:
            tags.append(tag)
    return tags


def _newest(pattern: str, *, exclude_substr: tuple[str, ...] = ()) -> str | None:
    cands = [
        q for q in glob.glob(pattern)
        if not any(s in os.path.basename(q) for s in exclude_substr)
    ]
    return max(cands, key=os.path.getmtime) if cands else None


def load_meta(tag: str) -> dict:
    p = MODEL_DIR / f"{tag}_latest.meta.json"
    if not p.exists():
        newest = _newest(str(MODEL_DIR / f"{tag}_*.meta.json"))
        if newest is None:
            raise FileNotFoundError(f"no meta.json for {tag}")
        p = Path(newest)
    return json.load(open(p))


def load_model(tag: str):
    """Load a model artifact by tag, handling both xgb-json and joblib families."""
    algo = algo_of(tag)
    if algo == "xgb":
        path = MODEL_DIR / f"{tag}_latest.json"
        if not path.exists():
            # exclude meta.json (also matches *.json)
            newest = _newest(str(MODEL_DIR / f"{tag}_*.json"),
                             exclude_substr=(".meta.json",))
            if newest is None:
                raise FileNotFoundError(f"no xgb json for {tag}")
            path = Path(newest)
        model = xgb.XGBClassifier()
        model.load_model(str(path))
        return model, path.name
    # joblib families: cb / rf / lgbm / lr / iso / stack
    path = MODEL_DIR / f"{tag}_latest.joblib"
    if not path.exists():
        newest = _newest(str(MODEL_DIR / f"{tag}_*.joblib"),
                         exclude_substr=(".calibrator.joblib",))
        if newest is None:
            raise FileNotFoundError(f"no joblib for {tag}")
        path = Path(newest)
    return joblib.load(path), path.name


def load_calibrator(tag: str):
    p = MODEL_DIR / f"{tag}_latest.calibrator.joblib"
    if p.exists():
        return joblib.load(p)
    newest = _newest(str(MODEL_DIR / f"{tag}_*.calibrator.joblib"))
    return joblib.load(newest) if newest else None


def apply_calibrator(calibrator, raw) -> np.ndarray:
    """Apply a Platt (LogisticRegression) calibrator directly from its linear
    params — version-proof vs. sklearn's ``predict_proba`` internals (matches
    supercompute.apply_calibrator)."""
    raw_arr = np.asarray(raw, dtype=float).ravel()
    try:
        coef = np.asarray(calibrator.coef_, dtype=float).ravel()[0]
        intercept = float(np.asarray(calibrator.intercept_).ravel()[0])
        z = raw_arr * coef + intercept
        p1 = 1.0 / (1.0 + np.exp(-z))
        classes = getattr(calibrator, "classes_", np.array([0, 1]))
        if len(classes) == 2 and classes[1] == 0:
            p1 = 1.0 - p1
        return p1
    except AttributeError:
        return calibrator.predict_proba(raw_arr.reshape(-1, 1))[:, 1]


# --------------------------------------------------------------------------- #
# Market + events + feature building (shared across models)
# --------------------------------------------------------------------------- #
def load_market_and_events(slug: str):
    EVENTS_DIR.mkdir(parents=True, exist_ok=True)
    pkl = EVENTS_DIR / f"{slug}.pkl"
    if pkl.exists():
        logging.info(f"events: loading market + events from cache {pkl}")
        with open(pkl, "rb") as f:
            market, events = pickle.load(f)
        logging.info(f"events: {len(events)} loaded from cache (offline-ok)")
        return market, events
    logging.info("events: cache miss -> resolving market + fetching (needs internet)...")
    market = get_market_by_slug(slug)
    events = get_market_orderfilled_events(market, verbose=True)
    with open(pkl, "wb") as f:
        pickle.dump((market, events), f)
    logging.info(f"events: fetched + cached {len(events)} to {pkl}")
    return market, events


def load_flagged_wallets() -> set[str]:
    if not KNOWN_INSIDER_JSON.exists():
        return set()
    with open(KNOWN_INSIDER_JSON) as f:
        pairs = json.load(f)
    return {p["wallet"].lower() for p in pairs}


def resolve_api_key(cli_key: str | None) -> str | None:
    return (cli_key or os.environ.get("POLYGONSCAN_API_KEY")
            or os.environ.get("ETHERSCAN_API_KEY"))


def build_feature_row(
    wallet: str,
    slug: str,
    market,
    history_df,
    wallet_trades: list,
    behavioral_needed: set[str],
    metadata_needed: set[str],
    api_key: str | None,
    flagged_wallets: set[str],
) -> tuple[dict, dict]:
    """Return ``(row, context)``: the full feature dict for the pair plus a few
    human-readable context fields (as_of, volume, winrate, ...)."""
    if market.closed and market.outcome_prices:
        final_yes = 1 if float(market.outcome_prices[0]) >= 0.5 else 0
    else:
        final_yes = None

    trade_ts = [int(t["timestamp"]) for t in wallet_trades
                if t.get("timestamp") is not None]
    as_of = min(trade_ts) if trade_ts else None
    volume = sum(float(t["cash_amount"]) for t in wallet_trades)

    logging.info(f"features: computing {len(behavioral_needed)} behavioral features")
    behav = compute_market_user_features(
        parsed_trades=wallet_trades,
        history_df=history_df,
        market_start_time=market.start_ts,
        market_end_time=market.end_ts,
        final_outcome_yes=final_yes,
        feature_filter=set(behavioral_needed),
    )

    meta_vals: dict = {c: np.nan for c in metadata_needed}
    if metadata_needed:
        logging.info(f"features: computing {len(metadata_needed)} metadata features "
                     f"(api_key={'set' if api_key else 'MISSING -> funder feats NaN'})")
        try:
            mf = compute_metadata_features(
                wallet.lower(), slug, int(market.start_ts),
                as_of_ts=as_of, api_key=api_key, flagged_wallets=flagged_wallets,
            )
            for c in metadata_needed:
                meta_vals[c] = mf.get(c, np.nan)
        except Exception as exc:  # noqa: BLE001
            logging.warning(f"metadata failed: {type(exc).__name__}: {exc}; leaving NaN")

    row = {}
    for c in behavioral_needed:
        row[c] = behav.get(c, np.nan)
    row.update(meta_vals)

    context = {
        "as_of_ts": as_of,
        "volume": volume,
        "n_trades": len(wallet_trades),
        "final_outcome_yes": final_yes,
        "winrate": meta_vals.get("winrate", np.nan),
        "n_markets_traded": meta_vals.get("n_markets_traded", np.nan),
        "herfindahl_index_markets": meta_vals.get("herfindahl_index_markets", np.nan),
        "funder_flagged_wallet_count": meta_vals.get("funder_flagged_wallet_count", np.nan),
    }
    return row, context


# --------------------------------------------------------------------------- #
# Scoring one model
# --------------------------------------------------------------------------- #
def score_model(tag: str, meta: dict, model, calibrator, row: dict) -> dict | None:
    """Score the pair with one model. Returns a result dict, or None if the
    artifact can't produce a score (e.g. the broken stack MinMaxScaler)."""
    features = list(meta["features"])
    threshold = meta.get("threshold")
    X = pd.DataFrame([{c: row.get(c, np.nan) for c in features}])[features].astype(float)

    has_proba = hasattr(model, "predict_proba")
    has_decision = hasattr(model, "decision_function")

    if has_proba:
        raw = float(model.predict_proba(X)[:, 1][0])
        prob = float(apply_calibrator(calibrator, [raw])[0]) if calibrator is not None else raw
        thr = threshold if threshold is not None else DEFAULT_FLAG_THRESHOLD
        flagged = bool(prob >= thr)
        score_kind = "prob"
        display = prob
    elif has_decision:  # IsolationForest-style anomaly model (no predict_proba)
        raw = float(model.decision_function(X)[0])
        prob = None
        pred = model.predict(X)[0] if hasattr(model, "predict") else None
        flagged = bool(pred == -1) if pred is not None else bool(raw < 0)
        thr = "anomaly<0"
        score_kind = "anomaly_score"
        display = raw
    else:
        logging.warning(f"model {tag}: no predict_proba/decision_function "
                        f"({type(model).__name__}); skipping")
        return None

    expl = explain_prediction(model, X, features, top_k=SHAP_TOP_K,
                              logger=logging.getLogger())

    return {
        "tag": tag,
        "label": LABELS.get(tag, tag),
        "algo": algo_of(tag),
        "n_features": len(features),
        "calibrated": calibrator is not None,
        "score_kind": score_kind,
        "score": display,
        "raw": raw,
        "prob": prob,
        "threshold": thr,
        "flagged": flagged,
        "explain": expl,
        "features": features,
    }


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Score one (wallet, market) pair against every insider model with SHAP.")
    p.add_argument("--wallet", default=DEFAULT_WALLET, help="0x wallet address")
    p.add_argument("--market", default=DEFAULT_MARKET, help="Polymarket market slug")
    p.add_argument("--api-key", default=None,
                   help="Polygonscan/Etherscan key (else uses env var)")
    p.add_argument("--top-k", type=int, default=SHAP_TOP_K,
                   help="SHAP features to log per model")
    p.add_argument("--out-dir", default=".", help="where to write log + CSVs")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    wallet = args.wallet.lower()
    slug = args.market
    slug8 = slug.replace("-", "")[:20] or "market"
    wallet8 = wallet.replace("0x", "")[:8]
    stem = f"{slug8}_{wallet8}"

    log_path, ts = configure_logging(out_dir, stem)
    logging.info("=" * 72)
    logging.info(f"audit_pair starting, log={log_path}")
    logging.info(f"wallet={wallet}")
    logging.info(f"market={slug}")
    logging.info(f"shap available: {HAS_SHAP}")

    # --- discover models + collect the feature union they need ------------- #
    tags = discover_model_tags()
    metas: dict[str, dict] = {}
    for tag in tags:
        try:
            metas[tag] = load_meta(tag)
        except Exception as exc:  # noqa: BLE001
            logging.warning(f"skip {tag}: cannot load meta ({exc})")
    tags = [t for t in tags if t in metas]
    logging.info(f"models to score: {len(tags)}")

    meta_col_set = set(METADATA_FEATURE_COLUMNS)
    all_feats: set[str] = set()
    for m in metas.values():
        all_feats.update(m.get("features", []))
    behavioral_needed = {f for f in all_feats if f not in meta_col_set}
    metadata_needed = {f for f in all_feats if f in meta_col_set}
    logging.info(f"feature union: {len(all_feats)} "
                 f"({len(behavioral_needed)} behavioral + {len(metadata_needed)} metadata)")

    # --- resolve market + events, parse this wallet's trades --------------- #
    market, events = load_market_and_events(slug)
    history_df = build_history_df_from_orderfilled(
        events, yes_token=market.yes_token_id, no_token=market.no_token_id)
    parsed = parse_orderfilled_events(events, market.token_ids, market.outcomes)
    wallet_trades = parsed["wallet_trades"].get(wallet, [])
    if not wallet_trades:
        logging.error(f"wallet {wallet} has NO trades on market {slug}; nothing to score")
        return
    logging.info(f"wallet trades on this market: {len(wallet_trades)}")

    api_key = resolve_api_key(args.api_key)
    flagged_wallets = load_flagged_wallets()
    row, ctx = build_feature_row(
        wallet, slug, market, history_df, wallet_trades,
        behavioral_needed, metadata_needed, api_key, flagged_wallets)

    ft = ctx["as_of_ts"]
    ft_str = (time.strftime("%Y-%m-%d %H:%M UTC", time.gmtime(int(ft)))
              if ft is not None else "n/a")
    logging.info("-" * 72)
    logging.info("PAIR CONTEXT")
    logging.info(f"  first trade on market : {ft_str}")
    logging.info(f"  volume / n_trades     : ${ctx['volume']:,.0f} / {ctx['n_trades']}")
    logging.info(f"  market closed / YES-won: {market.closed} / {ctx['final_outcome_yes']}")

    def _fmt(v):
        return f"{v:.4f}" if isinstance(v, (int, float)) and v == v else "NaN"

    logging.info(f"  winrate / n_markets    : {_fmt(ctx['winrate'])} / "
                 f"{_fmt(ctx['n_markets_traded'])}")
    logging.info(f"  herfindahl / funder-flagged siblings: "
                 f"{_fmt(ctx['herfindahl_index_markets'])} / "
                 f"{_fmt(ctx['funder_flagged_wallet_count'])}")

    # --- score every model ------------------------------------------------- #
    results: list[dict] = []
    for tag in tags:
        try:
            model, artifact = load_model(tag)
        except Exception as exc:  # noqa: BLE001
            logging.warning(f"skip {tag}: cannot load model ({exc})")
            continue
        cal = load_calibrator(tag)
        try:
            res = score_model(tag, metas[tag], model, cal, row)
        except Exception as exc:  # noqa: BLE001
            logging.warning(f"skip {tag}: scoring failed ({type(exc).__name__}: {exc})")
            continue
        if res is None:
            continue
        res["artifact"] = artifact
        results.append(res)

    if not results:
        logging.error("no models produced a score; exiting")
        return

    # --- per-model detail blocks ------------------------------------------ #
    # Order: flagged first, then by score descending within proba models.
    results.sort(key=lambda r: (not r["flagged"],
                                -(r["prob"] if r["prob"] is not None else -1)))
    logging.info("=" * 72)
    logging.info("PER-MODEL RESULTS (flagged first)")
    for r in results:
        logging.info("-" * 72)
        cal = "calibrated" if r["calibrated"] else "raw"
        verdict = "==> FLAGGED insider" if r["flagged"] else "    not flagged"
        logging.info(f"MODEL  {r['label']}   "
                     f"[tag={r['tag']}, algo={r['algo']}, nf={r['n_features']}, {cal}]")
        if r["score_kind"] == "prob":
            logging.info(f"  prob={r['prob']:.4f} (raw={r['raw']:.4f})  "
                         f"threshold={r['threshold']}   {verdict}")
        else:
            logging.info(f"  anomaly_score={r['raw']:.4f} "
                         f"(lower = more anomalous; {r['threshold']})   {verdict}")
        logging.info("  features driving THIS prediction (SHAP, signed toward insider):")
        for line in format_attributions(r["explain"]).splitlines():
            logging.info(line)

    # --- summary table ----------------------------------------------------- #
    n_flag = sum(1 for r in results if r["flagged"])
    logging.info("=" * 72)
    logging.info(f"SUMMARY: {n_flag}/{len(results)} models flag this pair as an insider")
    logging.info(f"{'model':<26}{'algo':>5}{'nf':>4}{'score':>10}{'thr':>10}  flagged")
    logging.info("-" * 72)
    for r in results:
        score_s = f"{r['score']:.4f}"
        thr_s = f"{r['threshold']}" if not isinstance(r["threshold"], float) else f"{r['threshold']:.3f}"
        flag_s = "YES" if r["flagged"] else "no"
        logging.info(f"{r['label']:<26}{r['algo']:>5}{r['n_features']:>4}"
                     f"{score_s:>10}{thr_s:>10}  {flag_s}")
    logging.info("=" * 72)

    # --- CSVs -------------------------------------------------------------- #
    models_csv = out_dir / f"audit_{stem}_{ts}_models.csv"
    shap_csv = out_dir / f"audit_{stem}_{ts}_shap.csv"
    mrows = []
    srows = []
    for r in results:
        top = r["explain"]["attributions"]
        mrows.append({
            "tag": r["tag"], "label": r["label"], "algo": r["algo"],
            "n_features": r["n_features"], "calibrated": r["calibrated"],
            "score_kind": r["score_kind"], "score": r["score"], "raw": r["raw"],
            "prob": r["prob"], "threshold": r["threshold"], "flagged": r["flagged"],
            "explain_method": r["explain"]["method"],
            "top_features": "; ".join(
                f"{a['feature']}({a['contribution']:+.3f})" for a in top[:6]),
        })
        for rank, a in enumerate(top, 1):
            srows.append({
                "tag": r["tag"], "label": r["label"], "rank": rank,
                "feature": a["feature"], "value": a["value"],
                "contribution": a["contribution"],
                "explain_method": r["explain"]["method"],
            })
    pd.DataFrame(mrows).to_csv(models_csv, index=False)
    pd.DataFrame(srows).to_csv(shap_csv, index=False)
    logging.info(f"wrote {models_csv}")
    logging.info(f"wrote {shap_csv}")
    logging.info(f"log saved to {log_path}")


if __name__ == "__main__":
    main()
