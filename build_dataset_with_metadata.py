"""Merge the 17 metadata features onto the cached training dataset.

Takes the behavioral feature matrix (``cache/training_data.parquet``) and
joins the metadata feature matrix (``cache/metadata_features.parquet``,
produced by ``metadata_layer.py``) onto it, one metadata vector per
(wallet, market) pair. The result is a single wide dataset that downstream
model scripts (xgboost*.py, catboost*.py, stacking) can train on without
re-touching any API.

Only the 17 model features in ``METADATA_FEATURE_COLUMNS`` are carried
over; the bookkeeping columns (``as_of_ts``, ``funder``,
``first_deposit_ts``, ``_error``) are dropped.

The join is a strict 1:1 left-join keyed on lowercased
(wallet, market_slug). The script hard-fails if any base row is missing a
metadata match, if the metadata side has duplicate keys, or if the row
count changes — so silent drift or leakage via row drops is impossible.

Output:
  cache/training_data_with_metadata.parquet

Usage:
  python build_dataset_with_metadata.py
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from polycluster.metadata import METADATA_FEATURE_COLUMNS

ROOT = Path(__file__).resolve().parent
CACHE = ROOT / "cache"
BASE_PARQUET = CACHE / "training_data.parquet"
METADATA_PARQUET = CACHE / "metadata_features.parquet"
OUTPUT_PARQUET = CACHE / "training_data_with_metadata.parquet"

KEY_COLS = ["wallet", "market_slug"]


def _normalize_keys(df: pd.DataFrame, name: str) -> pd.DataFrame:
    """Lowercase the wallet column so the two sides join exactly."""
    missing = [c for c in KEY_COLS if c not in df.columns]
    if missing:
        raise SystemExit(f"{name} is missing key column(s): {missing}")
    df = df.copy()
    df["wallet"] = df["wallet"].str.lower()
    return df


def main() -> None:
    if not BASE_PARQUET.exists():
        raise SystemExit(f"missing base dataset: {BASE_PARQUET}")
    if not METADATA_PARQUET.exists():
        raise SystemExit(
            f"missing metadata features: {METADATA_PARQUET} "
            f"(run metadata_layer.py first)"
        )

    base = _normalize_keys(pd.read_parquet(BASE_PARQUET), BASE_PARQUET.name)
    meta = _normalize_keys(pd.read_parquet(METADATA_PARQUET), METADATA_PARQUET.name)
    print(f"[build] base     {BASE_PARQUET.name}  shape={base.shape}")
    print(f"[build] metadata {METADATA_PARQUET.name}  shape={meta.shape}")

    feature_cols = list(METADATA_FEATURE_COLUMNS)
    missing_feats = [c for c in feature_cols if c not in meta.columns]
    if missing_feats:
        raise SystemExit(f"metadata file is missing feature column(s): {missing_feats}")

    # Guard against accidentally overwriting behavioral columns that happen
    # to share a name with a metadata feature.
    collisions = [c for c in feature_cols if c in base.columns]
    if collisions:
        raise SystemExit(
            f"metadata feature column(s) already present in base dataset: "
            f"{collisions} (would be silently overwritten)"
        )

    # Strict 1:1 keying: the metadata side must be unique on (wallet, slug).
    dup_mask = meta.duplicated(subset=KEY_COLS, keep=False)
    if dup_mask.any():
        dups = meta.loc[dup_mask, KEY_COLS].drop_duplicates()
        raise SystemExit(
            f"metadata file has {len(dups)} duplicated (wallet, market_slug) "
            f"key(s); join would not be 1:1:\n{dups.to_string(index=False)}"
        )

    meta_slim = meta[KEY_COLS + feature_cols]

    n_before = len(base)
    merged = base.merge(meta_slim, on=KEY_COLS, how="left", validate="one_to_one")

    if len(merged) != n_before:
        raise SystemExit(
            f"row count changed during merge: {n_before} -> {len(merged)}"
        )

    # Every base row must have found a metadata match. Probe one feature
    # column for NaNs introduced purely by a failed join (vs. a genuine NaN
    # feature value, which we can't distinguish here) -> instead check that
    # the join key existed on the metadata side for every base row.
    base_keys = set(map(tuple, base[KEY_COLS].itertuples(index=False, name=None)))
    meta_keys = set(map(tuple, meta_slim[KEY_COLS].itertuples(index=False, name=None)))
    unmatched = base_keys - meta_keys
    if unmatched:
        sample = list(unmatched)[:10]
        raise SystemExit(
            f"{len(unmatched)} base (wallet, market_slug) pair(s) have no "
            f"metadata row; sample: {sample}"
        )

    OUTPUT_PARQUET.parent.mkdir(parents=True, exist_ok=True)
    merged.to_parquet(OUTPUT_PARQUET, index=False)

    print(f"[build] added {len(feature_cols)} metadata feature columns:")
    for i, c in enumerate(feature_cols, 1):
        nan_frac = merged[c].isna().mean()
        print(f"  {i:2d}. {c:48s}  NaN={nan_frac:5.1%}")
    print(f"[build] wrote {OUTPUT_PARQUET}  shape={merged.shape}")


if __name__ == "__main__":
    main()
