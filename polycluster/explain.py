"""Per-prediction SHAP explanations for the insider models.

Where :mod:`polycluster.modeling` fits models and :mod:`polycluster.features`
builds the inputs, this module answers a single question for one scored row:
*which features drove this particular prediction, and in which direction?*

The one public entry point, :func:`explain_prediction`, takes any fitted model
we train — a raw estimator or an sklearn ``Pipeline`` — plus a single feature
row, and returns the signed per-feature contributions toward the positive
("insider") class, ranked by magnitude. It handles every family in the zoo:

  * XGBoost / CatBoost / LightGBM  -> ``shap.TreeExplainer`` on the estimator.
  * RandomForest / IsolationForest -> ``shap.TreeExplainer`` on the pipeline's
    final estimator, with the row first pushed through the pipeline's imputer
    (and any scaler) so SHAP sees exactly what the tree saw.
  * LogisticRegression(CV)         -> the exact linear contribution
    ``coef * scaled_x`` (SHAP for a linear model is closed-form), read straight
    off the fitted pipeline.

SHAP is an optional dependency. If it is not installed, or an explainer raises,
the helper degrades to the model's global ``feature_importances_`` / ``coef_``
so callers always get *something* back rather than a crash — the ``method``
field of the result says which path was taken.

For tree ensembles the contributions are in the model's margin space (log-odds
for xgb/lgbm/gbm, the CatBoost raw score, the averaged tree output for RF, the
anomaly score for IsolationForest); for the linear model they are in log-odds.
They are meant for *ranking* what mattered for this row, not as calibrated
probability deltas.
"""

from __future__ import annotations

import logging
from typing import Any, Sequence

import numpy as np
import pandas as pd

try:  # SHAP is optional; we fall back to global importance without it.
    import shap  # type: ignore

    _HAS_SHAP = True
except Exception:  # noqa: BLE001 - any import failure => no shap
    _HAS_SHAP = False


__all__ = ["explain_prediction", "format_attributions", "HAS_SHAP"]

HAS_SHAP = _HAS_SHAP


def _as_row_df(x_row: Any, feature_names: Sequence[str]) -> pd.DataFrame:
    """Coerce ``x_row`` into a 1-row DataFrame with columns in model order."""
    feature_names = list(feature_names)
    if isinstance(x_row, pd.DataFrame):
        return x_row.loc[:, feature_names].astype(float).head(1).reset_index(drop=True)
    if isinstance(x_row, pd.Series):
        return x_row.reindex(feature_names).astype(float).to_frame().T.reset_index(drop=True)
    arr = np.asarray(x_row, dtype=float).reshape(1, -1)
    return pd.DataFrame(arr, columns=feature_names)


def _split_pipeline(model: Any) -> tuple[Any | None, Any]:
    """Return ``(preprocessor_or_None, final_estimator)``.

    For an sklearn ``Pipeline`` the preprocessor is everything but the last
    step (imputer, scaler, ...); the final estimator is the last step. For a
    bare estimator the preprocessor is None.
    """
    steps = getattr(model, "steps", None)
    if steps:
        final = steps[-1][1]
        if len(steps) == 1:
            return None, final
        try:
            from sklearn.pipeline import Pipeline

            return Pipeline(steps[:-1]), final
        except Exception:  # noqa: BLE001
            return None, final
    return None, model


def _is_linear(est: Any) -> bool:
    name = type(est).__name__
    return hasattr(est, "coef_") and name.startswith(
        ("LogisticRegression", "Linear", "Ridge", "SGD", "Perceptron")
    )


def _positive_class_vector(sv: Any, n_features: int) -> np.ndarray:
    """Normalize a ``TreeExplainer.shap_values`` result to a 1-D per-feature
    vector of contributions toward the positive class for a single sample.

    Across shap/sklearn versions the shape can be:
      * a list ``[class0, class1]`` of ``(n_samples, n_features)`` arrays,
      * a 3-D array ``(n_samples, n_features, n_classes)``,
      * a 2-D array ``(n_samples, n_features)`` (single-output boosters),
      * a 1-D array ``(n_features,)``.
    We always reduce to the positive-class row.
    """
    if isinstance(sv, list):
        arr = np.asarray(sv[1] if len(sv) > 1 else sv[0])
    else:
        arr = np.asarray(sv)
        if arr.ndim == 3:  # (n_samples, n_features, n_classes)
            arr = arr[..., -1]
    if arr.ndim == 2:
        arr = arr[0]
    arr = np.ravel(arr)
    if arr.shape[0] != n_features and arr.size % n_features == 0:
        arr = arr.reshape(-1, n_features)[-1]
    return arr.astype(float)


def _pack(
    feature_names: Sequence[str],
    values: dict[str, float],
    contrib: np.ndarray,
    base_value: float | None,
    method: str,
    top_k: int,
) -> dict:
    feature_names = list(feature_names)
    contrib = np.asarray(contrib, dtype=float).ravel()
    order = np.argsort(-np.abs(contrib))
    attributions = []
    for i in order[: top_k if top_k and top_k > 0 else len(order)]:
        f = feature_names[i]
        attributions.append(
            {
                "feature": f,
                "value": float(values.get(f, float("nan"))),
                "contribution": float(contrib[i]),
            }
        )
    return {"method": method, "base_value": base_value, "attributions": attributions}


def explain_prediction(
    model: Any,
    x_row: Any,
    feature_names: Sequence[str],
    *,
    top_k: int = 12,
    logger: logging.Logger | None = None,
) -> dict:
    """Explain one prediction: which features drove the score, and how.

    Args:
        model: A fitted estimator or sklearn ``Pipeline`` (any family we train).
        x_row: The single feature row — a 1-row DataFrame, a Series, or a 1-D
            array-like in ``feature_names`` order. NaNs are allowed (imputer /
            native-NaN models handle them).
        feature_names: Feature order the model expects (from its meta.json).
        top_k: Keep this many features, ranked by ``|contribution|``. ``<= 0``
            keeps all of them.
        logger: Optional logger for a warning if SHAP fails and we fall back.

    Returns:
        ``{"method": str, "base_value": float | None, "attributions": [...]}``
        where each attribution is ``{"feature", "value", "contribution"}``,
        sorted by descending ``|contribution|``. ``method`` records which path
        produced the numbers (``shap (TreeExplainer)``, ``linear (coef*x)``,
        ``global feature_importances_``, or ``unavailable``).
    """
    feature_names = list(feature_names)
    row_df = _as_row_df(x_row, feature_names)
    values = row_df.iloc[0].to_dict()
    pre, est = _split_pipeline(model)

    # Push the row through any preprocessing so the explainer sees model input.
    try:
        x_trans = pre.transform(row_df) if pre is not None else row_df.to_numpy(dtype=float)
    except Exception:  # noqa: BLE001
        x_trans = row_df.to_numpy(dtype=float)
    x_trans = np.asarray(x_trans, dtype=float)

    # ---- linear models: SHAP is closed-form (coef * x on scaled space) ----
    if _is_linear(est):
        coef = np.asarray(est.coef_, dtype=float).ravel()
        contrib = coef * x_trans[0]
        base = float(np.asarray(getattr(est, "intercept_", [0.0])).ravel()[0])
        return _pack(feature_names, values, contrib, base,
                     "linear (coef*x, log-odds)", top_k)

    # ---- tree ensembles: SHAP TreeExplainer ----
    if _HAS_SHAP:
        try:
            explainer = shap.TreeExplainer(est)
            sv = explainer.shap_values(x_trans, check_additivity=False)
            contrib = _positive_class_vector(sv, len(feature_names))
            ev = explainer.expected_value
            if isinstance(ev, (list, tuple, np.ndarray)):
                ev = float(np.ravel(ev)[-1])
            else:
                ev = float(ev)
            return _pack(feature_names, values, contrib, ev,
                         "shap (TreeExplainer)", top_k)
        except Exception as exc:  # noqa: BLE001
            if logger is not None:
                logger.warning(
                    f"shap failed for {type(est).__name__}: "
                    f"{type(exc).__name__}: {exc}; falling back to global importance"
                )

    # ---- fallback: global feature importance (not per-prediction) ----
    imp = getattr(est, "feature_importances_", None)
    if imp is not None:
        return _pack(feature_names, values, np.asarray(imp, dtype=float), None,
                     "global feature_importances_ (NOT per-prediction)", top_k)
    coef = getattr(est, "coef_", None)
    if coef is not None:
        return _pack(feature_names, values, np.asarray(coef, dtype=float).ravel(), None,
                     "global |coef| (NOT per-prediction)", top_k)
    return {"method": "unavailable", "base_value": None, "attributions": []}


def format_attributions(expl: dict, *, indent: str = "        ") -> str:
    """Render an :func:`explain_prediction` result as aligned log lines."""
    method = expl.get("method", "?")
    base = expl.get("base_value")
    head = f"{indent}explain [{method}]"
    if base is not None:
        head += f"  base={base:+.4f}"
    lines = [head]
    for a in expl.get("attributions", []):
        v = a["value"]
        v_str = f"{v:+.4f}" if v == v else "   NaN"  # NaN != NaN
        arrow = "↑insider" if a["contribution"] >= 0 else "↓insider"
        lines.append(
            f"{indent}  {a['feature']:<44} "
            f"contrib={a['contribution']:+.4f} {arrow}  value={v_str}"
        )
    if len(lines) == 1:
        lines.append(f"{indent}  (no attribution available)")
    return "\n".join(lines)
