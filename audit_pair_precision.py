"""audit_pair_precision.py — re-audit one (wallet, market) pair, but flag each
model at a threshold tuned for high precision instead of its saved default.

The plain :mod:`audit_pair` flags the pair at each model's *saved* threshold,
and how many models fire is very sensitive to those (differently-chosen)
cutoffs. This script instead asks a fairer question:

    "If I set every model's threshold so it runs at ~80% precision, how many
     of them still flag this suspicious pair?"

For each model in the same curated set as ``audit_pair`` it:

  1. Rebuilds the model's own held-out TEST split from the labelled training
     data (``cache/training_data_with_metadata.parquet``) using the exact
     ``test_markets`` recorded in the model's meta.json (GroupShuffleSplit by
     market, seed 42). Precision is measured only on markets the model never
     trained on.
  2. Scores those held-out rows on the same scale ``audit_pair`` uses — the
     *calibrated* probability for proba models; ``-decision_function`` (higher
     = more anomalous) for the isolation forest.
  3. Sweeps the threshold and picks the operating point with the highest recall
     whose precision is >= 80%. If no threshold reaches 80% precision, it falls
     back to the model's *max achievable* precision point (and the model is
     still counted).
  4. Re-scores the suspicious pair and flags it at that tuned threshold.

It logs, per model: the saved-vs-tuned threshold, the precision/recall the tuned
threshold buys on the held-out split (and how many test insiders it was
measured on, so you can judge noise), the pair's score, and the flag under both
thresholds — then a summary of how many models flag under each.

Reuses all model-loading / feature-building from :mod:`audit_pair`.

Example:
  export POLYGONSCAN_API_KEY=...
  python audit_pair_precision.py \
    --wallet 0xc1259ddd92f58aded52d029e37e3fbbebafaa4c1 \
    --market will-d4vd-be-the-1-searched-person-on-google-this-year
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

import audit_pair as ap
from polycluster.metadata import METADATA_FEATURE_COLUMNS

TRAINING_DATA = ap.CACHE / "training_data_with_metadata.parquet"
LABEL_COL = "is_insider"
SLUG_COL = "market_slug"
TARGET_PRECISION = 0.80


# --------------------------------------------------------------------------- #
# Scoring on the common "insider score" scale (higher = more suspicious)
# --------------------------------------------------------------------------- #
def insider_scores(model, calibrator, X: pd.DataFrame):
    """Return (scores, kind, raw). ``scores`` is on a scale where higher = more
    likely insider (what we flag on); ``raw`` is the model's own output before
    calibration, kept so the log can show both:
      * proba models -> scores = calibrated prob, raw = uncalibrated predict_proba,
      * isolation forest -> scores = ``-decision_function``, raw = decision_function.
    """
    if hasattr(model, "predict_proba"):
        raw = np.asarray(model.predict_proba(X)[:, 1], dtype=float).ravel()
        prob = ap.apply_calibrator(calibrator, raw) if calibrator is not None else raw
        return np.asarray(prob, dtype=float).ravel(), "prob", raw
    if hasattr(model, "decision_function"):
        dec = np.asarray(model.decision_function(X), dtype=float).ravel()
        return -dec, "neg_anomaly", dec
    return None, None, None


def tune_threshold(scores: np.ndarray, labels: np.ndarray,
                   target: float = TARGET_PRECISION,
                   objective: str = "precision") -> dict:
    """Pick a threshold on ``scores`` (higher = more insider). Return a dict with
    the chosen threshold plus the precision/recall it achieves.

    Two objectives:
      * ``precision`` — among all cut points, take the one with the highest
        recall whose precision >= ``target``; if none reaches ``target``, fall
        back to the maximum-precision cut (tie-break: higher recall).
      * ``recall`` — among all cut points, take the one with the highest
        precision whose recall >= ``target``; if none reaches ``target``, fall
        back to the maximum-recall cut (tie-break: higher precision).

    ``reached`` records whether the target floor was met (vs the fallback).
    """
    scores = np.asarray(scores, dtype=float)
    labels = np.asarray(labels, dtype=int)
    n_pos = int(labels.sum())
    if len(scores) == 0 or n_pos == 0:
        return {"ok": False, "reason": "no positives in test split",
                "n_test": int(len(scores)), "n_pos": n_pos}

    order = np.argsort(-scores)  # highest score first
    s_sorted = scores[order]
    y_sorted = labels[order]

    # Walk down the threshold: after including row i, predictions are
    # {score >= s_sorted[i]}. Only evaluate at distinct score boundaries.
    tp = fp = 0
    cand = []  # (thr, precision, recall, n_pred_pos)
    for i in range(len(s_sorted)):
        if y_sorted[i] == 1:
            tp += 1
        else:
            fp += 1
        if i + 1 < len(s_sorted) and s_sorted[i + 1] == s_sorted[i]:
            continue  # wait until the full tie block is included
        prec = tp / (tp + fp)
        rec = tp / n_pos
        cand.append((float(s_sorted[i]), prec, rec, tp + fp))

    if objective == "recall":
        # highest precision among cut points clearing the recall floor
        meeting = [c for c in cand if c[2] >= target]
        if meeting:
            thr, prec, rec, npos = max(meeting, key=lambda c: (c[1], c[2], c[0]))
            reached = True
        else:
            thr, prec, rec, npos = max(cand, key=lambda c: (c[2], c[1]))
            reached = False
    else:
        # highest recall among cut points clearing the precision floor
        meeting = [c for c in cand if c[1] >= target]
        if meeting:
            thr, prec, rec, npos = max(meeting, key=lambda c: (c[2], c[0]))
            reached = True
        else:
            thr, prec, rec, npos = max(cand, key=lambda c: (c[1], c[2]))
            reached = False

    return {"ok": True, "threshold": thr, "precision": prec, "recall": rec,
            "n_pred_pos": npos, "n_test": int(len(scores)), "n_pos": n_pos,
            "reached": reached}


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def configure_logging(out_dir: Path, stem: str) -> tuple[str, str]:
    ts = time.strftime("%Y%m%d_%H%M%S")
    log_path = str(out_dir / f"auditprec_{stem}_{ts}.log")
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    for h in list(logger.handlers):
        logger.removeHandler(h)
    fmt = logging.Formatter("%(asctime)s [auditprec] %(message)s", datefmt="%H:%M:%S")
    for h in (logging.StreamHandler(sys.stdout), logging.FileHandler(log_path)):
        h.setFormatter(fmt)
        logger.addHandler(h)
    return log_path, ts


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Re-audit a pair with per-model thresholds tuned to >=80% precision.")
    p.add_argument("--wallet", default=ap.DEFAULT_WALLET, help="0x wallet address")
    p.add_argument("--market", default=ap.DEFAULT_MARKET, help="Polymarket market slug")
    p.add_argument("--api-key", default=None,
                   help="Polygonscan/Etherscan key (else uses env var)")
    p.add_argument("--objective", choices=["precision", "recall"], default="precision",
                   help="tune each threshold for a precision floor (max recall, default) "
                        "or a recall floor (max precision).")
    p.add_argument("--target-precision", type=float, default=TARGET_PRECISION,
                   help="precision floor for --objective precision (default 0.80)")
    p.add_argument("--target-recall", type=float, default=0.80,
                   help="recall floor for --objective recall (default 0.80)")
    p.add_argument("--out-dir", default=".", help="where to write log + CSV")
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

    objective = args.objective
    target = args.target_recall if objective == "recall" else args.target_precision
    floor_word = "recall" if objective == "recall" else "precision"
    maxed_word = "precision" if objective == "recall" else "recall"

    log_path, ts = configure_logging(out_dir, stem)
    logging.info("=" * 72)
    logging.info(f"audit_pair_precision starting, log={log_path}")
    logging.info(f"wallet={wallet}")
    logging.info(f"market={slug}")
    logging.info(f"objective: max {maxed_word} s.t. {floor_word} >= {target:.2f} "
                 f"per model")

    if not TRAINING_DATA.exists():
        logging.error(f"labelled dataset not found: {TRAINING_DATA}; cannot tune thresholds")
        return
    data = pd.read_parquet(TRAINING_DATA)
    logging.info(f"labelled data: {len(data)} rows, {int(data[LABEL_COL].sum())} insiders, "
                 f"{data[SLUG_COL].nunique()} markets")

    # --- discover models + feature union ----------------------------------- #
    tags = ap.discover_model_tags()
    metas: dict[str, dict] = {}
    for tag in tags:
        try:
            metas[tag] = ap.load_meta(tag)
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

    # --- resolve market + events, build the pair's feature row ------------- #
    market, events = ap.load_market_and_events(slug)
    history_df = ap.build_history_df_from_orderfilled(
        events, yes_token=market.yes_token_id, no_token=market.no_token_id)
    parsed = ap.parse_orderfilled_events(events, market.token_ids, market.outcomes)
    wallet_trades = parsed["wallet_trades"].get(wallet, [])
    if not wallet_trades:
        logging.error(f"wallet {wallet} has NO trades on market {slug}; nothing to score")
        return

    api_key = ap.resolve_api_key(args.api_key)
    flagged_wallets = ap.load_flagged_wallets()
    row, ctx = ap.build_feature_row(
        wallet, slug, market, history_df, wallet_trades,
        behavioral_needed, metadata_needed, api_key, flagged_wallets)
    logging.info(f"pair context: vol=${ctx['volume']:,.0f}  n_trades={ctx['n_trades']}  "
                 f"winrate={ctx['winrate']}  herfindahl={ctx['herfindahl_index_markets']}")

    # --- per model: tune threshold on held-out test, then score the pair --- #
    results: list[dict] = []
    for tag in tags:
        meta = metas[tag]
        features = list(meta["features"])
        # Held-out markets: most metas store a ``test_markets`` list; the
        # pin-one-market models (e.g. rf_insider_meta_v5) store a single
        # ``test_market`` string instead. Older random-split artifacts store
        # neither -> no honest per-market held-out set, left un-tuned.
        test_markets = meta.get("test_markets")
        if not test_markets and meta.get("test_market"):
            test_markets = [meta["test_market"]]
        default_thr = meta.get("threshold")

        try:
            model, _artifact = ap.load_model(tag)
        except Exception as exc:  # noqa: BLE001
            logging.warning(f"skip {tag}: cannot load model ({exc})")
            continue
        cal = ap.load_calibrator(tag)

        # Pair score on the common insider scale.
        X_pair = pd.DataFrame([{c: row.get(c, np.nan) for c in features}])[features].astype(float)
        pair_scores, kind, pair_raws = insider_scores(model, cal, X_pair)
        if pair_scores is None:
            logging.warning(f"skip {tag}: no predict_proba/decision_function")
            continue
        pair_score = float(pair_scores[0])
        pair_raw = float(pair_raws[0])

        # Held-out test split from this model's recorded test_markets.
        tune = {"ok": False, "reason": "no test_markets in meta"}
        if test_markets:
            mask = data[SLUG_COL].isin(set(test_markets))
            sub = data.loc[mask]
            if len(sub):
                X_test = sub.reindex(columns=features).astype(float)
                y_test = sub[LABEL_COL].to_numpy()
                test_scores, _, _ = insider_scores(model, cal, X_test)
                tune = tune_threshold(test_scores, y_test, target, objective)

        default_flag = (pair_score >= default_thr) if (kind == "prob" and default_thr is not None) \
            else (bool(pair_score >= -0.0) if kind == "neg_anomaly" else None)

        rec = {
            "tag": tag, "label": ap.LABELS.get(tag, tag), "algo": ap.algo_of(tag),
            "nf": len(features), "kind": kind, "pair_score": pair_score,
            "pair_raw": pair_raw,
            "default_threshold": default_thr, "default_flag": default_flag,
            "tune": tune,
        }
        if tune.get("ok"):
            rec["tuned_threshold"] = tune["threshold"]
            rec["tuned_flag"] = bool(pair_score >= tune["threshold"])
        else:
            rec["tuned_threshold"] = None
            rec["tuned_flag"] = None
        results.append(rec)

    if not results:
        logging.error("no models produced a score; exiting")
        return

    # Order: tuned-flagged first, then by pair score descending.
    results.sort(key=lambda r: (r["tuned_flag"] is not True, -r["pair_score"]))

    logging.info("=" * 72)
    logging.info(f"PER-MODEL (threshold tuned for {floor_word} >= {target:.2f} "
                 f"on held-out test split; maximizing {maxed_word})")
    for r in results:
        logging.info("-" * 72)
        t = r["tune"]
        logging.info(f"MODEL  {r['label']}   [tag={r['tag']}, algo={r['algo']}, nf={r['nf']}]")
        if r["kind"] == "prob":
            logging.info(f"  pair model score   : prob={r['pair_score']:.4f} "
                         f"(raw={r['pair_raw']:.4f})")
        else:
            logging.info(f"  pair model score   : anomaly={r['pair_raw']:.4f} "
                         f"(insider-score={r['pair_score']:.4f}; higher = more anomalous)")
        if t.get("ok"):
            note = "" if t["reached"] else f"  (could NOT reach target; using max-{maxed_word} point)"
            logging.info(f"  tuned threshold    : {r['tuned_threshold']:.4f}{note}")
            logging.info(f"  held-out precision : {t['precision']:.3f} @ recall {t['recall']:.3f} "
                         f"(flagged {t['n_pred_pos']} of {t['n_test']} test rows; "
                         f"{t['n_pos']} true insiders in split)")
            logging.info(f"  FLAG (tuned)       : "
                         f"{'==> FLAGGED insider' if r['tuned_flag'] else '    not flagged'}")
        else:
            logging.info(f"  tuned threshold    : N/A ({t.get('reason')})")
        df_thr = r["default_threshold"]
        df_thr_s = f"{df_thr:.3f}" if isinstance(df_thr, float) else str(df_thr)
        logging.info(f"  (saved threshold={df_thr_s} -> "
                     f"default flag={'YES' if r['default_flag'] else 'no'})")

    # --- summary ----------------------------------------------------------- #
    tuned_flags = [r for r in results if r["tuned_flag"] is True]
    default_flags = [r for r in results if r["default_flag"] is True]
    unreach = [r for r in results if r["tune"].get("ok") and not r["tune"]["reached"]]
    no_test = [r for r in results if not r["tune"].get("ok")]

    logging.info("=" * 72)
    logging.info(f"SUMMARY over {len(results)} models")
    logging.info(f"  flagged at TUNED thresholds (>= {target:.0%} {floor_word}): "
                 f"{len(tuned_flags)}/{len(results)}")
    logging.info(f"  flagged at SAVED thresholds (for reference)                : "
                 f"{len(default_flags)}/{len(results)}")
    if unreach:
        logging.info(f"  models that could NOT reach {target:.0%} {floor_word} "
                     f"(used max-{maxed_word} point): {len(unreach)}")
    if no_test:
        logging.info(f"  models with no usable test split (threshold left un-tuned): {len(no_test)}")
    logging.info("-" * 81)
    logging.info(f"{'model':<26}{'nf':>4}{'score':>9}{'raw':>9}{'tuned_thr':>11}"
                 f"{'prec':>7}{'rec':>7}  tuned_flag")
    logging.info("-" * 81)
    for r in results:
        t = r["tune"]
        tt = f"{r['tuned_threshold']:.3f}" if r["tuned_threshold"] is not None else "N/A"
        pr = f"{t['precision']:.2f}" if t.get("ok") else "-"
        rc = f"{t['recall']:.2f}" if t.get("ok") else "-"
        fl = "YES" if r["tuned_flag"] else ("no" if r["tuned_flag"] is False else "N/A")
        logging.info(f"{r['label']:<26}{r['nf']:>4}{r['pair_score']:>9.4f}"
                     f"{r['pair_raw']:>9.4f}{tt:>11}"
                     f"{pr:>7}{rc:>7}  {fl}")
    logging.info("=" * 81)

    # --- CSV --------------------------------------------------------------- #
    csv_path = out_dir / f"auditprec_{stem}_{ts}_models.csv"
    rows = []
    for r in results:
        t = r["tune"]
        rows.append({
            "tag": r["tag"], "label": r["label"], "algo": r["algo"], "nf": r["nf"],
            "kind": r["kind"], "pair_score": r["pair_score"], "pair_raw": r["pair_raw"],
            "default_threshold": r["default_threshold"], "default_flag": r["default_flag"],
            "tuned_threshold": r["tuned_threshold"], "tuned_flag": r["tuned_flag"],
            "test_precision": t.get("precision"), "test_recall": t.get("recall"),
            "n_test": t.get("n_test"), "n_test_insiders": t.get("n_pos"),
            "reached_target": t.get("reached"),
        })
    pd.DataFrame(rows).to_csv(csv_path, index=False)
    logging.info(f"wrote {csv_path}")
    logging.info(f"log saved to {log_path}")


if __name__ == "__main__":
    main()
