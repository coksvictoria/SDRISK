"""Apply internal row-band remediation without exposing removed-row details."""

from typing import cast

import pandas as pd

from sdrisk.results import RowStore

_BAND_RANK = {"low": 0, "medium": 1, "high": 2}


def remove_above_band(
    synthetic: pd.DataFrame,
    row_store: RowStore,
    *,
    max_band: str,
) -> pd.DataFrame:
    """Return synthetic rows at or below a risk band, without returning flags.

    The output is a cleaned dataframe only. It does not include dropped row
    indices, counts, or an aligned risk-flag vector.
    """
    if not isinstance(max_band, str) or max_band not in _BAND_RANK:
        raise ValueError("max_band must be one of: low, medium, high")
    if len(synthetic) != len(row_store._profiles):
        raise ValueError("RowStore profile row count must match the synthetic row count")
    maximum = _BAND_RANK[max_band]
    keep_positions = [
        position
        for position, profile in enumerate(row_store._profiles)
        if _BAND_RANK[str(profile["composite_band"])] <= maximum
    ]
    cleaned = synthetic.iloc[keep_positions].reset_index(drop=True).copy()
    return cast(pd.DataFrame, cleaned)
