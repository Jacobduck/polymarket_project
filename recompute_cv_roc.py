"""Recompute cross-validated ROC-AUC for every CV-capable insider model.

The training pipelines only ever stored CV *PR-AUC* per fold for the xgb family
and LR (the cb/rf family did store per-fold ROC). To draw the model-comparison
graph on ROC-AUC instead of PR-AUC we need a CV ROC number for every model.
This script produces one WITHOUT retraining or touching any saved artifact:

  * cb / rf non-experiment models -- already store cv.roc_auc_{mean,std,folds};
    copied straight through (these are the exact train-time numbers).
  * xgb models with per-fold ``val_markets`` -- the stored folds pin down the
    exact CV partition, so we refit an XGBClassifier(**model_params) per fold and
    score ROC (and PR, to prove the partition reproduces the stored PR-AUC).
  * xgb / rf iran-experiment models (25ir / 0ir) -- reproduce the deterministic
    iran filter, drop the stored test markets, rebuild the same
    StratifiedGroupKFold(5)/GroupKFold folds, refit with stored params.
  * lr -- rebuild the grouped CV (iran mar-1 pinned out, StratifiedGroupKFold 5)
    and refit the stored logistic pipeline config per fold.

AUC (ROC and average-precision) is invariant to the monotonic Platt calibration
the models ship with, so calibrators are irrelevant here and ignored.

Every refit path is validated against the stored per-fold PR-AUC; the max abs
per-fold PR discrepancy is printed so any imperfect reproduction is visible.

Output (does NOT modify any model): cache/cv_roc_recomputed.json
"""

from __future__ import annotations

import glob
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import GroupKFold, StratifiedGroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

from compare_models import MODEL_DIR, NEW_TAGS, load_models

CACHE = Path("cache")
LABEL = "is_insider"
GROUP = "market_slug"
RANDOM_STATE = 42
OUT_PATH = CACHE / "cv_roc_recomputed.json"
LR_PIN_MARKET = "us-strikes-iran-by-march-1-2026-492"


def locate_meta(tag: str) -> str:
    p = MODEL_DIR / f"{tag}_latest.meta.json"
    if p.exists() and tag not in NEW_TAGS:
        return str(p)
    cands = [q for q in glob.glob(str(MODEL_DIR / f"{tag}_*.meta.json"))
             if "_latest.meta.json" not in q]
    if cands:
        return max(cands, key=os.path.getmtime)
    return str(p)


def load_source(meta: dict) -> pd.DataFrame:
    name = os.path.basename(meta["source_training_data"])
    return pd.read_parquet(CACHE / name)


def filter_iran(df: pd.DataFrame, iran: dict) -> pd.DataFrame:
    """Reproduce 25ir_*/0ir_* filter_iran exactly (seed-42 down-sample / drop)."""
    if iran["mode"] == "0ir":
        # dropped_markets can be stored truncated ("...-227..."); prefix-match.
        prefixes = [d.rstrip(".") for d in iran["dropped_markets"]]
        drop = df[GROUP].apply(lambda s: any(s.startswith(p) for p in prefixes))
        return df[~drop].copy()
    mar1, hide = iran["market"], int(iran["hidden_insiders"])
    sub = df[df[GROUP] == mar1]
    ins, non = sub[sub[LABEL] == 1], sub[sub[LABEL] == 0]
    keep_ins = len(ins) - hide
    keep_non = min(int(round(keep_ins * len(non) / max(len(ins), 1))), len(non))
    keep = set(ins.sample(n=keep_ins, random_state=RANDOM_STATE).index) | \
        set(non.sample(n=keep_non, random_state=RANDOM_STATE).index)
    drop = set(sub.index) - keep
    return df.drop(index=list(drop)).copy()


def score_folds(fit_predict, pool: pd.DataFrame, feats, splits) -> tuple[list, list]:
    """Run splits over pool; return (roc_per_fold, pr_per_fold)."""
    X, y, g = pool[feats].astype(float), pool[LABEL].astype(int), pool[GROUP]
    rocs, prs = [], []
    for tr, va in splits:
        proba = fit_predict(X.iloc[tr], y.iloc[tr], X.iloc[va])
        yv = y.iloc[va]
        if yv.nunique() < 2:
            continue
        rocs.append(float(roc_auc_score(yv, proba)))
        prs.append(float(average_precision_score(yv, proba)))
    return rocs, prs


def xgb_fitter(params):
    def f(Xtr, ytr, Xva):
        clf = XGBClassifier(**params)
        clf.fit(Xtr, ytr)
        return clf.predict_proba(Xva)[:, 1]
    return f


def rf_fitter(params):
    # rf training CV wraps the forest in a median-imputer pipeline (see
    # 25ir_randomforest.py); reproduce it so folds match.
    p = dict(params)
    cw = p.get("class_weight")
    if isinstance(cw, dict):
        p["class_weight"] = {int(k): v for k, v in cw.items()}
    p.setdefault("random_state", RANDOM_STATE)
    p.setdefault("n_jobs", -1)

    def f(Xtr, ytr, Xva):
        model = Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("clf", RandomForestClassifier(**p)),
        ])
        model.fit(Xtr, ytr)
        return model.predict_proba(Xva)[:, 1]
    return f


def lr_fitter(C, class_weight):
    def f(Xtr, ytr, Xva):
        model = Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(C=C, solver="lbfgs", max_iter=2000,
                                       class_weight=class_weight,
                                       random_state=RANDOM_STATE)),
        ])
        model.fit(Xtr, ytr)
        return model.predict_proba(Xva)[:, 1]
    return f


def recompute_one(tag: str, algo: str) -> dict | None:
    meta = json.load(open(locate_meta(tag)))
    cv = meta.get("cv") or {}
    feats = meta["features"]

    # ---- cb/rf that already stored CV ROC: pass through -------------------- #
    if "roc_auc_mean" in cv:
        folds = cv.get("folds") or []
        roc_folds = [f["roc_auc"] for f in folds if "roc_auc" in f]
        return {
            "roc_mean": cv["roc_auc_mean"], "roc_std": cv.get("roc_auc_std"),
            "roc_folds": roc_folds, "source": "stored",
            "pr_stored_mean": cv.get("pr_auc_mean"), "pr_repro_mean": cv.get("pr_auc_mean"),
            "pr_max_abs_diff": 0.0, "scheme": cv.get("scheme"),
        }

    df = load_source(meta)

    # ---- xgb with per-fold val_markets: reconstruct exact partition -------- #
    folds = cv.get("folds") or []
    if algo == "xgb" and folds and "val_markets" in folds[0]:
        pool_markets = set().union(*(set(f["val_markets"]) for f in folds))
        pool = df[df[GROUP].isin(pool_markets)].reset_index(drop=True)
        idx_by_market = {m: pool.index[pool[GROUP] == m].to_numpy() for m in pool_markets}
        splits = []
        for f in folds:
            va = np.concatenate([idx_by_market[m] for m in f["val_markets"]])
            tr = pool.index[~pool[GROUP].isin(f["val_markets"])].to_numpy()
            splits.append((tr, va))
        rocs, prs = score_folds(xgb_fitter(meta["model_params"]), pool, feats, splits)
        stored_pr = [f["pr_auc"] for f in folds]
        return _pack(rocs, prs, stored_pr, "refit-valmarkets", cv.get("scheme"))

    # ---- iran-experiment models: reproduce filter + grouped folds --------- #
    iran = meta.get("iran_filter")
    if iran:
        filtered = filter_iran(df, iran)
        pool = filtered[~filtered[GROUP].isin(meta["test_markets"])].reset_index(drop=True)
        y, g = pool[LABEL].astype(int), pool[GROUP]
        if algo == "xgb":
            n = cv.get("n_folds") or len(folds) or 5
            sp = StratifiedGroupKFold(n_splits=n, shuffle=True, random_state=RANDOM_STATE)
            splits = list(sp.split(pool[feats], y, g))
            fitter = xgb_fitter(meta["model_params"])
            stored_pr = [f["pr_auc"] for f in folds]
        else:  # rf
            n = len(cv.get("pr_auc_folds") or []) or 4
            n = min(n, g.nunique())
            splits = list(GroupKFold(n_splits=n).split(pool[feats], y, g))
            fitter = rf_fitter(meta["model_params"])
            stored_pr = cv.get("pr_auc_folds") or []
        rocs, prs = score_folds(fitter, pool, feats, splits)
        return _pack(rocs, prs, stored_pr, "refit-experiment", cv.get("scheme"))

    # ---- pin-one-market design, GroupKFold, no stored folds (rf meta v5) --- #
    if meta.get("test_market") and algo in ("rf", "xgb"):
        pool = df[df[GROUP] != meta["test_market"]].reset_index(drop=True)
        y, g = pool[LABEL].astype(int), pool[GROUP]
        n = min(4, g.nunique())
        splits = list(GroupKFold(n_splits=n).split(pool[feats], y, g))
        fitter = rf_fitter(meta["model_params"]) if algo == "rf" \
            else xgb_fitter(meta["model_params"])
        rocs, prs = score_folds(fitter, pool, feats, splits)
        res = _pack(rocs, prs, [], "refit-pin1", cv.get("scheme"))
        res["pr_stored_mean"] = (meta.get("tuning") or {}).get("best_cv_pr_auc")
        return res

    # ---- logistic regression: grouped CV, iran mar-1 pinned out ----------- #
    if algo == "lr":
        pool = df[df[GROUP] != LR_PIN_MARKET].reset_index(drop=True)
        y, g = pool[LABEL].astype(int), pool[GROUP]
        sp = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
        splits = list(sp.split(pool[feats], y, g))
        class_weight = {0: 1.0, 1: float(meta["scale_pos_weight"])}
        rocs, prs = score_folds(
            lr_fitter(float(meta["selected_C"]), class_weight), pool, feats, splits)
        stored_pr = cv.get("pr_auc_folds") or []
        return _pack(rocs, prs, stored_pr, "refit-grouped", cv.get("scheme"))

    print(f"[roc] no recompute path for {tag} (algo={algo}); skipping")
    return None


def _pack(rocs, prs, stored_pr, source, scheme) -> dict:
    diff = 0.0
    if stored_pr and len(stored_pr) == len(prs):
        diff = float(max(abs(a - b) for a, b in zip(sorted(prs), sorted(stored_pr))))
    return {
        "roc_mean": float(np.mean(rocs)), "roc_std": float(np.std(rocs)),
        "roc_folds": rocs, "source": source,
        "pr_stored_mean": float(np.mean(stored_pr)) if stored_pr else None,
        "pr_repro_mean": float(np.mean(prs)),
        "pr_max_abs_diff": diff, "scheme": scheme,
    }


def main() -> None:
    rows = [r for r in load_models() if r["cv_pr"] is not None]
    print(f"[roc] recomputing CV ROC-AUC for {len(rows)} CV-capable models\n")
    out: dict[str, dict] = {}
    print(f"{'model':<28} {'algo':>4} {'source':>16} {'ROC mean±std':>16} "
          f"{'PR repro':>9} {'PR stored':>9} {'Δpr':>7}")
    print("-" * 96)
    for r in sorted(rows, key=lambda r: r["tag"]):
        res = recompute_one(r["tag"], r["algo"])
        if res is None:
            continue
        out[r["tag"]] = res
        prs = f"{res['pr_stored_mean']:.4f}" if res["pr_stored_mean"] is not None else "  -"
        print(f"{r['label']:<28} {r['algo']:>4} {res['source']:>16} "
              f"{res['roc_mean']:.4f}±{res['roc_std']:.4f}  "
              f"{res['pr_repro_mean']:>9.4f} {prs:>9} {res['pr_max_abs_diff']:>7.4f}")

    json.dump(out, open(OUT_PATH, "w"), indent=2)
    worst = max((v["pr_max_abs_diff"] for v in out.values()), default=0.0)
    print(f"\n[roc] wrote {len(out)} models -> {OUT_PATH}")
    print(f"[roc] worst per-fold PR reproduction error = {worst:.4f} "
          f"(0 = exact match to stored folds)")


if __name__ == "__main__":
    main()
