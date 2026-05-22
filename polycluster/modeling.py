"""Modeling step of the polycluster pipeline.

This module turns the per-(wallet, market) feature dicts produced by
:mod:`polycluster.features` into a labeled, model-ready DataFrame and
trains / evaluates logistic-regression classifiers on it.

The typical flow is:

1. :func:`build_feature_dataframe` — drive
   :func:`polycluster.features.get_user_features` over many
   (wallet, market) pairs and stack the per-row dicts into a single
   DataFrame.
2. :func:`impute_feature_only_df` — add ``has_*`` availability flags
   inferred from the feature columns themselves, then impute the
   feature columns to ``0`` so an estimator can consume them.
3. Concatenate insiders (``is_insider=1``) and non-insiders
   (``is_insider=0``) into one labeled frame.
4. Pick a CV strategy:

   * :func:`nested_kfold_loocv_logistic` — outer ``StratifiedKFold``
     for unbiased evaluation, inner LOOCV to pick ``C``.
   * :func:`tune_C_loocv_logistic` — single-loop LOOCV across a
     ``C``-grid, picks best ``C``, optionally plots AUC vs ``C``.
   * :func:`run_loocv_logistic` / :func:`run_loocv_elasticnet_logistic`
     — LOOCV at a fixed ``C`` (and ``l1_ratio`` for the elastic-net
     variant); building blocks that the tuners call.

5. Use :func:`fit_final_logistic` to refit on all data, then
   :func:`score_new_rows` for inference and
   :func:`explain_single_prediction` /
   :func:`explain_misclassification_direction` /
   :func:`get_misclassified_rows` for diagnostics.

The default sklearn pipeline is
``SimpleImputer(median) → StandardScaler → LogisticRegression``,
matching the choices in the original analysis notebook.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.model_selection import LeaveOneOut, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from polycluster.features import get_user_features
from polycluster.markets import Market

# ---------------------------------------------------------------------------
# Default feature shortlist
# ---------------------------------------------------------------------------

DEFAULT_BEHAVIORAL_FEATURES: list[str] = [
    "unrealized_edge_per_open_token",
    "adaptive_markout_raw_grosswt_h0p030",
    "adaptive_markout_raw_grosswt_h0p070",
    "adaptive_markout_raw_grosswt_h0p150",
    "adaptive_late_spike_edge_grosswt_h0p030",
    "adaptive_markout_latespike_adj_grosswt_h0p070",
    "adaptive_plus_realized_alpha0p5_h0p070",
    "adaptive_plus_realized_alpha1p0_h0p030",
    "adaptive_plus_realized_alpha1p0_h0p070",
    "adaptive_minus_realized_alpha0p5_h0p070",
    "adaptive_minus_realized_alpha1p0_h0p030",
    "adaptive_minus_realized_alpha1p0_h0p070",
    "low_buy_high_sell_pctl",
    "realized_pnl_per_matched_token",
]
"""Curated subset of :func:`compute_market_user_features` outputs that the
original insider-trading analysis notebook used as the logistic-regression
inputs. Useful as a starting point; pass a custom list to override.
"""


# ---------------------------------------------------------------------------
# Feature-matrix assembly
# ---------------------------------------------------------------------------

def build_feature_dataframe(
    pairs: Sequence[tuple[str, Market | str]] | Mapping[Market | str, Sequence[str]],
    *,
    is_insider: int | None = None,
    extra_columns: Mapping[str, Any] | None = None,
    **get_user_features_kwargs: Any,
) -> pd.DataFrame:
    """Run :func:`get_user_features` over many (wallet, market) pairs.

    Groups inputs by market so each market's events are fetched and
    parsed exactly once even when many wallets share a market. Returns
    one row per (wallet, market) pair, with the feature dict expanded
    into columns plus ``wallet`` and ``market_slug`` identifier columns.

    Args:
        pairs: Either a sequence of ``(wallet, market)`` tuples, or a
            mapping from market (slug or :class:`~polycluster.markets.Market`)
            to a sequence of wallet addresses. Duplicate (wallet, market)
            pairs are de-duplicated within a market.
        is_insider: If provided, an ``is_insider`` column is added with
            this value on every row. Convenient for assembling the
            insider / normal frames before concatenating.
        extra_columns: Optional mapping of constant column-name → value
            pairs to attach to every row (e.g. ``{"cohort": "v1"}``).
        **get_user_features_kwargs: Forwarded to :func:`get_user_features`.
            Use this to control fetching (``timeout``, ``checkpointed``,
            ``verbose``) and feature-tuning kwargs (``final_outcome_yes``,
            ``major_move_kwargs``, etc.).

    Returns:
        A DataFrame with one row per (wallet, market) pair. Column order
        matches the feature dict, with ``wallet``, ``market_slug``, and
        (optionally) ``is_insider`` and any ``extra_columns`` appended.

    Examples:
        Pairs form (mixed markets)::

            df = build_feature_dataframe(
                [("0xabc...", "slug-a"), ("0xdef...", "slug-b")],
                is_insider=1,
                verbose=True,
            )

        Mapping form (one market, many wallets)::

            df = build_feature_dataframe(
                {"slug-a": ["0xabc...", "0xdef..."]},
                is_insider=0,
            )
    """
    market_buckets: dict[str, tuple[Market | str, list[str]]] = {}

    def _market_key(m: Market | str) -> str:
        return m.slug if isinstance(m, Market) else m

    if isinstance(pairs, Mapping):
        for market, wallets in pairs.items():
            key = _market_key(market)
            bucket = market_buckets.setdefault(key, (market, []))
            bucket[1].extend(list(wallets))
    else:
        for wallet, market in pairs:
            key = _market_key(market)
            bucket = market_buckets.setdefault(key, (market, []))
            bucket[1].append(wallet)

    rows: list[dict[str, Any]] = []
    for key, (market, wallets) in market_buckets.items():
        unique_wallets = list(dict.fromkeys(wallets))
        if not unique_wallets:
            continue
        feats_by_wallet = get_user_features(
            unique_wallets, market, **get_user_features_kwargs
        )
        for addr in unique_wallets:
            feats = feats_by_wallet.get(addr, {})
            row: dict[str, Any] = dict(feats)
            row["wallet"] = addr
            row["market_slug"] = key
            if is_insider is not None:
                row["is_insider"] = is_insider
            if extra_columns:
                row.update(dict(extra_columns))
            rows.append(row)

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Imputation + availability indicators
# ---------------------------------------------------------------------------

def impute_feature_only_df(
    df: pd.DataFrame,
    cols_exact: Sequence[str],
) -> pd.DataFrame:
    """Add ``has_*`` availability flags and fill the listed features with 0.

    Many feature columns are NaN for legitimate reasons (a wallet had no
    sells, never closed a trade, has no open inventory, etc.). Naively
    filling those with ``0`` would conflate "absent" with "zero." This
    function preserves the missingness signal as binary indicator
    columns *before* imputing.

    Indicator columns produced (where the underlying features exist):

    * ``has_buy``                 — any ``buy_markout_*`` / ``low_buy_*``
      column non-null.
    * ``has_sell``                — any ``sell_markout_*`` / ``high_sell_*``
      column non-null.
    * ``has_adaptive_features``   — any ``adaptive_*`` / ``realized_edge``
      / ``realized_raw_pnl`` / ``derisk_*`` / ``terminal_*`` non-null.
    * ``has_realized_match``      — ``realized_pnl_per_matched_token`` non-null.
    * ``has_open_inventory``      — ``unrealized_edge_per_open_token`` non-null.
    * ``has_position_path``       — ``path_alignment`` non-null.
    * ``has_any_trade``           — ``active_day_frac`` non-null.

    Args:
        df: Input DataFrame, typically from
            :func:`build_feature_dataframe`.
        cols_exact: List of feature column names to impute. Columns not
            present in ``df`` are added as NaN before imputation, so
            single-row inference DataFrames missing a feature key get
            the same indicator-flag-then-fill-zero treatment as rows
            in a multi-row training set. Indicator columns are always
            added (set to ``0`` if their group is absent).

    Returns:
        A copy of ``df`` with indicator columns added and the listed
        feature columns NaN-filled with ``0.0``. Non-feature columns
        (``wallet``, ``market_slug``, labels, etc.) are left untouched.
    """
    out = df.copy()

    for col in cols_exact:
        if col not in out.columns:
            out[col] = np.nan

    feature_cols = list(cols_exact)

    buy_cols = [
        c for c in feature_cols
        if c.startswith("buy_markout") or c.startswith("low_buy")
    ]
    sell_cols = [
        c for c in feature_cols
        if c.startswith("sell_markout") or c.startswith("high_sell")
    ]
    adaptive_cols = [
        c for c in feature_cols
        if c.startswith("adaptive_")
        or c.startswith("realized_edge")
        or c.startswith("realized_raw_pnl")
        or c.startswith("derisk_")
        or c.startswith("terminal_")
    ]

    out["has_adaptive_features"] = (
        out[adaptive_cols].notna().any(axis=1).astype(int) if adaptive_cols else 0
    )
    out["has_buy"] = (
        out[buy_cols].notna().any(axis=1).astype(int) if buy_cols else 0
    )
    out["has_sell"] = (
        out[sell_cols].notna().any(axis=1).astype(int) if sell_cols else 0
    )
    out["has_realized_match"] = (
        out["realized_pnl_per_matched_token"].notna().astype(int)
        if "realized_pnl_per_matched_token" in out.columns
        else 0
    )
    out["has_open_inventory"] = (
        out["unrealized_edge_per_open_token"].notna().astype(int)
        if "unrealized_edge_per_open_token" in out.columns
        else 0
    )
    out["has_position_path"] = (
        out["path_alignment"].notna().astype(int)
        if "path_alignment" in out.columns
        else 0
    )
    out["has_any_trade"] = (
        out["active_day_frac"].notna().astype(int)
        if "active_day_frac" in out.columns
        else 0
    )

    if feature_cols:
        out[feature_cols] = out[feature_cols].fillna(0.0)

    return out


# ---------------------------------------------------------------------------
# Pipeline helpers (private)
# ---------------------------------------------------------------------------

def _build_logistic_pipeline(
    C: float,
    penalty: str = "l1",
    solver: str = "liblinear",
    class_weight: str | dict | None = "balanced",
    impute_strategy: str = "median",
) -> Pipeline:
    """Build the standard ``impute → scale → logistic`` pipeline."""
    return Pipeline([
        ("imputer", SimpleImputer(strategy=impute_strategy)),
        ("scaler", StandardScaler()),
        ("model", LogisticRegression(
            C=C,
            penalty=penalty,
            solver=solver,
            class_weight=class_weight,
            max_iter=2000,
        )),
    ])


def _inner_loocv_select_C(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    C_grid: Sequence[float],
    penalty: str = "l1",
    solver: str = "liblinear",
    class_weight: str | dict | None = "balanced",
    impute_strategy: str = "median",
) -> tuple[float, pd.DataFrame]:
    """Pick the best ``C`` by LOOCV on a single (outer-)training split."""
    loo = LeaveOneOut()
    records: list[dict[str, Any]] = []

    for C in C_grid:
        y_true_inner: list[int] = []
        y_prob_inner: list[float] = []

        for inner_tr_idx, inner_val_idx in loo.split(X_train, y_train):
            X_tr = X_train.iloc[inner_tr_idx]
            y_tr = y_train.iloc[inner_tr_idx]
            X_val = X_train.iloc[inner_val_idx]
            y_val = y_train.iloc[inner_val_idx]

            pipe = _build_logistic_pipeline(
                C=C,
                penalty=penalty,
                solver=solver,
                class_weight=class_weight,
                impute_strategy=impute_strategy,
            )
            pipe.fit(X_tr, y_tr)
            prob = pipe.predict_proba(X_val)[:, 1][0]

            y_true_inner.append(int(y_val.iloc[0]))
            y_prob_inner.append(float(prob))

        y_true_arr = np.asarray(y_true_inner)
        y_prob_arr = np.asarray(y_prob_inner)
        y_pred_arr = (y_prob_arr >= 0.5).astype(int)

        if len(np.unique(y_true_arr)) < 2:
            auc = np.nan
        else:
            auc = roc_auc_score(y_true_arr, y_prob_arr)
        acc = accuracy_score(y_true_arr, y_pred_arr)

        records.append({
            "C": C,
            "inner_loocv_auc": auc,
            "inner_loocv_accuracy_at_0.5": acc,
        })

    inner_summary = pd.DataFrame(records)
    best_idx = inner_summary.sort_values(
        ["inner_loocv_auc", "inner_loocv_accuracy_at_0.5", "C"],
        ascending=[False, False, True],
    ).index[0]
    best_C = float(inner_summary.loc[best_idx, "C"])

    return best_C, inner_summary


# ---------------------------------------------------------------------------
# Cross-validation drivers
# ---------------------------------------------------------------------------

def nested_kfold_loocv_logistic(
    df: pd.DataFrame,
    feature_cols: Sequence[str],
    label_col: str = "is_insider",
    outer_n_splits: int = 5,
    C_grid: Sequence[float] | None = None,
    penalty: str = "l1",
    solver: str = "liblinear",
    class_weight: str | dict | None = "balanced",
    impute_strategy: str = "median",
    random_state: int = 42,
    shuffle: bool = True,
    fit_final_model: bool = True,
) -> dict[str, Any]:
    """Nested CV — outer ``StratifiedKFold``, inner LOOCV ``C``-tuning.

    Each outer fold's training split picks its own ``C`` via LOOCV, then
    a fresh pipeline is trained at that ``C`` and scored on the held-out
    outer-test rows. Aggregating across folds gives an unbiased estimate
    of the *full tuning procedure*'s out-of-sample performance.

    Args:
        df: Labeled feature matrix.
        feature_cols: Columns of ``df`` to use as inputs.
        label_col: Binary label column. Rows where it is NaN are dropped.
        outer_n_splits: Number of outer folds.
        C_grid: Inverse-regularization values to consider. Defaults to
            ``[0.01, 0.03, 0.05, 0.1, 0.2, 0.4, 0.7, 1.0, 2.0, 5.0]``.
        penalty, solver, class_weight, impute_strategy: Forwarded to
            the inner pipeline.
        random_state, shuffle: Forwarded to ``StratifiedKFold``.
        fit_final_model: If True, also runs LOOCV-``C``-selection on the
            full dataset and fits a final model at the chosen ``C``.

    Returns:
        Dict with keys:

        * ``metrics`` — overall nested-CV AUC, accuracy, row/feature/class counts.
        * ``outer_fold_summary`` — one row per outer fold (selected ``C``,
          fold AUC, fold accuracy).
        * ``outer_predictions`` — one row per held-out sample
          (``y_true``, ``y_prob``, ``y_pred_0.5``, ``selected_C``,
          ``fold``, ``row_index``).
        * ``inner_summaries_by_fold`` — per-fold inner ``C``-grid summaries.
        * ``final_best_C`` — ``C`` chosen on full data (None if
          ``fit_final_model=False``).
        * ``final_inner_summary`` — full-data inner ``C``-grid summary.
        * ``final_model`` — pipeline fitted on all data at ``final_best_C``.
        * ``feature_cols`` — echoed for downstream convenience.
    """
    if C_grid is None:
        C_grid = [0.01, 0.03, 0.05, 0.1, 0.2, 0.4, 0.7, 1.0, 2.0, 5.0]

    work = df.dropna(subset=[label_col]).copy()
    X = work[list(feature_cols)].copy()
    y = work[label_col].astype(int).copy()

    outer_cv = StratifiedKFold(
        n_splits=outer_n_splits,
        shuffle=shuffle,
        random_state=random_state,
    )

    outer_rows: list[dict[str, Any]] = []
    outer_pred_rows: list[dict[str, Any]] = []
    inner_summaries: dict[int, pd.DataFrame] = {}

    for fold_id, (outer_tr_idx, outer_te_idx) in enumerate(outer_cv.split(X, y), start=1):
        X_outer_train = X.iloc[outer_tr_idx].reset_index(drop=True)
        y_outer_train = y.iloc[outer_tr_idx].reset_index(drop=True)
        X_outer_test = X.iloc[outer_te_idx].reset_index(drop=True)
        y_outer_test = y.iloc[outer_te_idx].reset_index(drop=True)

        best_C, inner_summary = _inner_loocv_select_C(
            X_train=X_outer_train,
            y_train=y_outer_train,
            C_grid=C_grid,
            penalty=penalty,
            solver=solver,
            class_weight=class_weight,
            impute_strategy=impute_strategy,
        )
        inner_summaries[fold_id] = inner_summary

        final_pipe = _build_logistic_pipeline(
            C=best_C,
            penalty=penalty,
            solver=solver,
            class_weight=class_weight,
            impute_strategy=impute_strategy,
        )
        final_pipe.fit(X_outer_train, y_outer_train)

        outer_prob = final_pipe.predict_proba(X_outer_test)[:, 1]
        outer_pred = (outer_prob >= 0.5).astype(int)

        for j, idx in enumerate(outer_te_idx):
            outer_pred_rows.append({
                "row_index": work.index[idx],
                "fold": fold_id,
                "y_true": int(y_outer_test.iloc[j]),
                "y_prob": float(outer_prob[j]),
                "y_pred_0.5": int(outer_pred[j]),
                "selected_C": best_C,
            })

        if len(np.unique(y_outer_test)) < 2:
            fold_auc = np.nan
        else:
            fold_auc = roc_auc_score(y_outer_test, outer_prob)
        fold_acc = accuracy_score(y_outer_test, outer_pred)

        outer_rows.append({
            "fold": fold_id,
            "n_train": len(outer_tr_idx),
            "n_test": len(outer_te_idx),
            "selected_C": best_C,
            "outer_test_auc": fold_auc,
            "outer_test_accuracy_at_0.5": fold_acc,
        })

    outer_fold_summary = pd.DataFrame(outer_rows)
    outer_predictions = (
        pd.DataFrame(outer_pred_rows).sort_values("row_index").reset_index(drop=True)
    )

    y_true_all = outer_predictions["y_true"].to_numpy()
    y_prob_all = outer_predictions["y_prob"].to_numpy()
    y_pred_all = outer_predictions["y_pred_0.5"].to_numpy()

    overall_auc = (
        roc_auc_score(y_true_all, y_prob_all)
        if len(np.unique(y_true_all)) >= 2
        else np.nan
    )
    overall_acc = accuracy_score(y_true_all, y_pred_all)

    metrics = {
        "nested_cv_auc": overall_auc,
        "nested_cv_accuracy_at_0.5": overall_acc,
        "n_rows": len(work),
        "n_features": len(feature_cols),
        "n_positive": int((y == 1).sum()),
        "n_negative": int((y == 0).sum()),
    }

    final_model = None
    final_best_C: float | None = None
    final_inner_summary: pd.DataFrame | None = None

    if fit_final_model:
        final_best_C, final_inner_summary = _inner_loocv_select_C(
            X_train=X.reset_index(drop=True),
            y_train=y.reset_index(drop=True),
            C_grid=C_grid,
            penalty=penalty,
            solver=solver,
            class_weight=class_weight,
            impute_strategy=impute_strategy,
        )
        final_model = _build_logistic_pipeline(
            C=final_best_C,
            penalty=penalty,
            solver=solver,
            class_weight=class_weight,
            impute_strategy=impute_strategy,
        )
        final_model.fit(X, y)

    return {
        "metrics": metrics,
        "outer_fold_summary": outer_fold_summary,
        "outer_predictions": outer_predictions,
        "inner_summaries_by_fold": inner_summaries,
        "final_best_C": final_best_C,
        "final_inner_summary": final_inner_summary,
        "final_model": final_model,
        "feature_cols": list(feature_cols),
    }


def run_loocv_logistic(
    df: pd.DataFrame,
    feature_cols: Sequence[str],
    label_col: str = "is_insider",
    C: float = 0.4,
    penalty: str = "l1",
    solver: str = "liblinear",
    class_weight: str | dict | None = "balanced",
    zero_tol: float = 1e-10,
) -> dict[str, Any]:
    """Leave-one-out CV for logistic regression at a fixed ``C``.

    Returns out-of-fold predictions, per-fold coefficient traces, a
    coefficient stability summary, and overall LOOCV metrics. Used as
    the building block for :func:`tune_C_loocv_logistic`.

    Args:
        df: Labeled feature matrix.
        feature_cols: Input columns.
        label_col: Binary label column.
        C: Inverse regularization strength.
        penalty, solver, class_weight: Forwarded to ``LogisticRegression``.
        zero_tol: Magnitude below which a coefficient counts as zero
            (used for ``nonzero_frac`` etc. in ``coef_summary``).

    Returns:
        Dict with ``oof_df``, ``coef_df``, ``coef_summary``, ``metrics``.
    """
    data = df.copy()
    X = data[list(feature_cols)].copy()
    y = data[label_col].astype(int).values

    loo = LeaveOneOut()

    oof_pred = np.zeros(len(data), dtype=float)
    oof_label = np.zeros(len(data), dtype=int)
    coef_rows: list[pd.DataFrame] = []

    for fold_idx, (train_idx, test_idx) in enumerate(loo.split(X, y), start=1):
        X_train = X.iloc[train_idx]
        X_test = X.iloc[test_idx]
        y_train = y[train_idx]
        y_test = y[test_idx]

        pipe = Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(
                penalty=penalty,
                solver=solver,
                C=C,
                class_weight=class_weight,
                max_iter=5000,
            )),
        ])
        pipe.fit(X_train, y_train)

        prob = pipe.predict_proba(X_test)[:, 1][0]
        oof_pred[test_idx[0]] = prob
        oof_label[test_idx[0]] = y_test[0]

        coef = pipe.named_steps["clf"].coef_[0]
        coef_rows.append(pd.DataFrame({
            "fold": fold_idx,
            "feature": list(feature_cols),
            "coef": coef,
        }))

    coef_df = pd.concat(coef_rows, axis=0, ignore_index=True)

    coef_summary = (
        coef_df.groupby("feature")["coef"]
        .agg(
            mean_coef="mean",
            median_coef="median",
            std_coef="std",
            mean_abs_coef=lambda s: np.mean(np.abs(s)),
            median_abs_coef=lambda s: np.median(np.abs(s)),
            nonzero_frac=lambda s: np.mean(np.abs(s) > zero_tol),
            positive_frac=lambda s: np.mean(s > zero_tol),
            negative_frac=lambda s: np.mean(s < -zero_tol),
        )
        .sort_values(["nonzero_frac", "mean_abs_coef"], ascending=False)
        .reset_index()
    )

    auc = roc_auc_score(oof_label, oof_pred)
    pred_class = (oof_pred >= 0.5).astype(int)
    acc = accuracy_score(oof_label, pred_class)

    metrics = {
        "loocv_auc": auc,
        "loocv_accuracy_at_0.5": acc,
        "n_rows": len(data),
        "n_features": len(feature_cols),
        "n_positive": int(np.sum(oof_label)),
        "n_negative": int(np.sum(oof_label == 0)),
        "C": C,
    }

    oof_df = data.copy()
    oof_df["y_true"] = oof_label
    oof_df["pred_prob_insider"] = oof_pred
    oof_df["pred_class_0.5"] = pred_class

    return {
        "oof_df": oof_df,
        "coef_df": coef_df,
        "coef_summary": coef_summary,
        "metrics": metrics,
    }


def run_loocv_elasticnet_logistic(
    df: pd.DataFrame,
    feature_cols: Sequence[str],
    label_col: str = "is_insider",
    C: float = 1.0,
    l1_ratio: float = 0.5,
    class_weight: str | dict | None = "balanced",
    zero_tol: float = 1e-10,
) -> dict[str, Any]:
    """LOOCV for binary logistic regression with an elastic-net penalty.

    Uses ``solver="saga"`` and ``penalty="elasticnet"`` so a feature
    with a coefficient driven exactly to zero is fully excluded.

    Args:
        df, feature_cols, label_col: As in :func:`run_loocv_logistic`.
        C: Inverse regularization strength.
        l1_ratio: Mixing parameter (``0`` = ridge, ``1`` = lasso).
        class_weight: Forwarded to ``LogisticRegression``.
        zero_tol: As in :func:`run_loocv_logistic`.

    Returns:
        Dict with ``oof_df``, ``coef_df``, ``coef_summary``, ``metrics``,
        and ``fitted_pipelines`` (the per-fold pipelines).
    """
    data = df.copy()
    X = data[list(feature_cols)].copy()
    y = data[label_col].astype(int).values

    loo = LeaveOneOut()

    oof_pred = np.zeros(len(data), dtype=float)
    oof_label = np.zeros(len(data), dtype=int)
    coef_rows: list[pd.DataFrame] = []
    fitted_pipelines: list[Pipeline] = []

    for fold_idx, (train_idx, test_idx) in enumerate(loo.split(X, y), start=1):
        X_train = X.iloc[train_idx]
        X_test = X.iloc[test_idx]
        y_train = y[train_idx]
        y_test = y[test_idx]

        pipe = Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(
                penalty="elasticnet",
                solver="saga",
                C=C,
                l1_ratio=l1_ratio,
                class_weight=class_weight,
                max_iter=10000,
                random_state=42,
            )),
        ])
        pipe.fit(X_train, y_train)
        fitted_pipelines.append(pipe)

        prob = pipe.predict_proba(X_test)[:, 1][0]
        oof_pred[test_idx[0]] = prob
        oof_label[test_idx[0]] = y_test[0]

        coef = pipe.named_steps["clf"].coef_[0]
        coef_rows.append(pd.DataFrame({
            "fold": fold_idx,
            "feature": list(feature_cols),
            "coef": coef,
        }))

    coef_df = pd.concat(coef_rows, axis=0, ignore_index=True)

    coef_summary = (
        coef_df.groupby("feature")["coef"]
        .agg(
            mean_coef="mean",
            median_coef="median",
            std_coef="std",
            mean_abs_coef=lambda s: np.mean(np.abs(s)),
            median_abs_coef=lambda s: np.median(np.abs(s)),
            nonzero_frac=lambda s: np.mean(np.abs(s) > zero_tol),
            positive_frac=lambda s: np.mean(s > zero_tol),
            negative_frac=lambda s: np.mean(s < -zero_tol),
        )
        .sort_values(["nonzero_frac", "mean_abs_coef"], ascending=False)
        .reset_index()
    )

    auc = roc_auc_score(oof_label, oof_pred)
    pred_class = (oof_pred >= 0.5).astype(int)
    acc = accuracy_score(oof_label, pred_class)

    metrics = {
        "loocv_auc": auc,
        "loocv_accuracy_at_0.5": acc,
        "n_rows": len(data),
        "n_features": len(feature_cols),
        "n_positive": int(np.sum(oof_label)),
        "n_negative": int(np.sum(oof_label == 0)),
        "C": C,
        "l1_ratio": l1_ratio,
    }

    oof_df = data.copy()
    oof_df["y_true"] = oof_label
    oof_df["pred_prob_insider"] = oof_pred
    oof_df["pred_class_0.5"] = pred_class

    return {
        "oof_df": oof_df,
        "coef_df": coef_df,
        "coef_summary": coef_summary,
        "metrics": metrics,
        "fitted_pipelines": fitted_pipelines,
    }


def fit_final_logistic(
    df: pd.DataFrame,
    feature_cols: Sequence[str],
    label_col: str = "is_insider",
    C: float = 0.4,
    penalty: str = "l1",
    solver: str = "liblinear",
    class_weight: str | dict | None = "balanced",
) -> Pipeline:
    """Fit the production pipeline on all rows at the given ``C``.

    Returns the fitted ``imputer → scaler → logistic`` pipeline. Use
    after a ``C`` has been chosen by :func:`tune_C_loocv_logistic` or
    :func:`nested_kfold_loocv_logistic`.
    """
    data = df.copy()
    X = data[list(feature_cols)].copy()
    y = data[label_col].astype(int).values

    pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(
            penalty=penalty,
            solver=solver,
            C=C,
            class_weight=class_weight,
            max_iter=5000,
        )),
    ])
    pipe.fit(X, y)
    return pipe


def tune_C_loocv_logistic(
    df: pd.DataFrame,
    feature_cols: Sequence[str],
    label_col: str = "is_insider",
    C_grid: Sequence[float] | None = None,
    penalty: str = "l1",
    solver: str = "liblinear",
    class_weight: str | dict | None = "balanced",
    plot: bool = True,
) -> dict[str, Any]:
    """Try a grid of ``C`` values, pick the best by LOOCV AUC, fit final model.

    For each ``C`` in ``C_grid``, runs :func:`run_loocv_logistic`, then
    selects the best by AUC (tie-breaks by accuracy then by smaller
    ``C``). Optionally plots LOOCV AUC vs ``C`` on a log axis.

    Args:
        df, feature_cols, label_col, penalty, solver, class_weight:
            Forwarded to :func:`run_loocv_logistic` and
            :func:`fit_final_logistic`.
        C_grid: Defaults to ``[0.01, 0.03, 0.05, 0.1, 0.2, 0.4, 0.7,
            1.0, 2.0, 5.0]``.
        plot: If True, draw the LOOCV AUC vs ``C`` chart.

    Returns:
        Dict with:

        * ``summary_df`` — one row per ``C`` in the grid with metrics.
        * ``best_C`` — chosen value.
        * ``best_result`` — the full :func:`run_loocv_logistic` payload
          at ``best_C``.
        * ``best_model`` — pipeline fitted on all data at ``best_C``.
        * ``result_by_C`` — every ``run_loocv_logistic`` payload, keyed
          by ``C``.
    """
    if C_grid is None:
        C_grid = [0.01, 0.03, 0.05, 0.1, 0.2, 0.4, 0.7, 1.0, 2.0, 5.0]

    all_results: list[dict[str, Any]] = []
    result_by_C: dict[float, dict[str, Any]] = {}

    for C in C_grid:
        res = run_loocv_logistic(
            df=df,
            feature_cols=feature_cols,
            label_col=label_col,
            C=C,
            penalty=penalty,
            solver=solver,
            class_weight=class_weight,
        )
        metrics = res["metrics"].copy()
        all_results.append(metrics)
        result_by_C[float(C)] = res

    summary_df = pd.DataFrame(all_results).sort_values("C").reset_index(drop=True)

    best_idx = summary_df.sort_values(
        ["loocv_auc", "loocv_accuracy_at_0.5", "C"],
        ascending=[False, False, True],
    ).index[0]
    best_row = summary_df.loc[best_idx]
    best_C = float(best_row["C"])
    best_result = result_by_C[best_C]

    best_model = fit_final_logistic(
        df=df,
        feature_cols=feature_cols,
        label_col=label_col,
        C=best_C,
        penalty=penalty,
        solver=solver,
        class_weight=class_weight,
    )

    if plot:
        plt.figure(figsize=(7, 4))
        plt.plot(summary_df["C"], summary_df["loocv_auc"], marker="o", label="LOOCV AUC")
        plt.axvline(best_C, linestyle="--", label=f"Best C = {best_C:g}")
        plt.xscale("log")
        plt.xlabel("C (log scale)")
        plt.ylabel("LOOCV AUC")
        plt.title("LOOCV AUC vs C")
        plt.legend()
        plt.tight_layout()
        plt.show()

    return {
        "summary_df": summary_df,
        "best_C": best_C,
        "best_result": best_result,
        "best_model": best_model,
        "result_by_C": result_by_C,
    }


# ---------------------------------------------------------------------------
# Inference + diagnostics
# ---------------------------------------------------------------------------

def score_new_rows(
    model: Any,
    df_new: pd.DataFrame,
    feature_cols: Sequence[str],
) -> pd.DataFrame:
    """Apply a fitted classifier to new rows.

    Args:
        model: Anything with ``predict_proba`` and ``decision_function``
            (a fitted sklearn pipeline).
        df_new: New rows (must contain ``feature_cols``).
        feature_cols: Columns to feed to the model.

    Returns:
        A copy of ``df_new`` with two extra columns:
        ``pred_prob_insider`` (probability of class 1) and
        ``decision_score`` (raw decision-function output).
    """
    X_new = df_new[list(feature_cols)].copy()
    pred_prob = model.predict_proba(X_new)[:, 1]
    decision = model.decision_function(X_new)

    out = df_new.copy()
    out["pred_prob_insider"] = pred_prob
    out["decision_score"] = decision
    return out


def explain_single_prediction(
    model: Any,
    X: pd.DataFrame,
    row_idx: Any,
) -> dict[str, Any]:
    """Per-feature contribution table for one row's prediction.

    Computes ``contribution = value * coef`` for every feature, plus the
    intercept, the resulting logit and probability. Useful for sanity-
    checking individual scores. Assumes a *bare* logistic-regression
    model (i.e. inputs have already been scaled the same way the model
    was trained), not a pipeline — call ``model.named_steps["clf"]`` if
    you have a pipeline.

    Args:
        model: A fitted ``LogisticRegression`` (binary).
        X: Feature matrix used at training time.
        row_idx: The label of the row in ``X`` to explain.

    Returns:
        Dict with ``row_index``, ``intercept``, ``logit``,
        ``prob_class_1``, and a ``contrib_table`` DataFrame sorted by
        absolute contribution.
    """
    row = X.loc[row_idx]
    coefs = model.coef_.ravel()
    intercept = model.intercept_[0]

    contrib = pd.DataFrame({
        "feature": X.columns,
        "value": row.values,
        "coef": coefs,
        "contribution": row.values * coefs,
    })
    contrib = contrib.sort_values("contribution", key=np.abs, ascending=False)

    logit = intercept + contrib["contribution"].sum()
    prob = 1 / (1 + np.exp(-logit))

    return {
        "row_index": row_idx,
        "intercept": intercept,
        "logit": logit,
        "prob_class_1": prob,
        "contrib_table": contrib,
    }


def explain_misclassification_direction(
    model: Any,
    X: pd.DataFrame,
    y: pd.Series,
    row_idx: Any,
) -> dict[str, Any]:
    """Find which features pushed a misclassified row in the wrong direction.

    For a false positive (true=0, pred=1), drivers are the features
    with the largest positive contributions. For a false negative
    (true=1, pred=0), drivers are the largest negative contributions.

    Args:
        model: A fitted ``LogisticRegression`` (binary).
        X: Feature matrix used at training time.
        y: Aligned true labels.
        row_idx: Label of the row to explain.

    Returns:
        If correctly classified: ``{"message": ...}``. Otherwise a dict
        with ``true_label``, ``pred_label``, ``pred_prob`` and a
        ``drivers`` DataFrame sorted to put the most blame-worthy
        features at the top.
    """
    row = X.loc[row_idx]
    true_label = y.loc[row_idx] if hasattr(y, "loc") else y[row_idx]

    coefs = model.coef_.ravel()
    intercept = model.intercept_[0]
    contributions = row.values * coefs

    df = pd.DataFrame({
        "feature": X.columns,
        "value": row.values,
        "coef": coefs,
        "contribution": contributions,
    })

    logit = intercept + contributions.sum()
    prob = 1 / (1 + np.exp(-logit))
    pred = int(prob >= 0.5)

    if pred == true_label:
        return {"message": "This row is not misclassified."}

    if true_label == 0 and pred == 1:
        driver_df = df.sort_values("contribution", ascending=False)
    else:
        driver_df = df.sort_values("contribution", ascending=True)

    return {
        "true_label": true_label,
        "pred_label": pred,
        "pred_prob": prob,
        "drivers": driver_df,
    }


def get_misclassified_rows(
    model: Any,
    X: pd.DataFrame,
    y: pd.Series | np.ndarray,
    threshold: float = 0.5,
) -> pd.DataFrame:
    """Return the rows where the model's prediction disagrees with truth.

    Args:
        model: Fitted classifier with ``predict_proba``.
        X: Feature matrix.
        y: Aligned true labels.
        threshold: Probability cutoff for the positive class.

    Returns:
        A copy of ``X`` filtered to misclassified rows, with extra
        columns ``true_label``, ``pred_proba``, ``pred_label`` and
        ``misclassified``, sorted by ``pred_proba`` ascending.
    """
    proba = model.predict_proba(X)[:, 1]
    pred = (proba >= threshold).astype(int)

    result = X.copy()
    result["true_label"] = y.values if hasattr(y, "values") else y
    result["pred_proba"] = proba
    result["pred_label"] = pred
    result["misclassified"] = result["true_label"] != result["pred_label"]
    return result[result["misclassified"]].sort_values("pred_proba")
