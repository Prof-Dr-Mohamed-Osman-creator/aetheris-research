# ==========================================
# Aetheris Research - Dataset merging (pure, testable)
# ==========================================

from __future__ import annotations

import pandas as pd


def merge_datasets(left: pd.DataFrame, right: pd.DataFrame, left_keys: list[str],
                   right_keys: list[str], how: str = "left") -> pd.DataFrame:
    """Merge two datasets on matching key columns, harmonising key dtypes."""
    left, right = left.copy(), right.copy()
    for lk, rk in zip(left_keys, right_keys):
        if pd.api.types.is_numeric_dtype(left[lk]) and pd.api.types.is_numeric_dtype(right[rk]):
            continue
        left[lk] = left[lk].astype(str).str.strip()
        right[rk] = right[rk].astype(str).str.strip()
    merged = left.merge(right, left_on=left_keys, right_on=right_keys, how=how,
                        suffixes=("", "_2"))
    # drop duplicated key columns coming from the right side
    dup = [rk for lk, rk in zip(left_keys, right_keys) if rk != lk and rk in merged.columns]
    return merged.drop(columns=dup)
