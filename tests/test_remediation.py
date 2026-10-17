"""Tests for safe removal of rows above the configured risk band."""

import pandas as pd
import pytest

from sdrisk.remediation import remove_above_band
from sdrisk.results import RowStore


def test_remediation_returns_only_cleaned_data_without_removed_row_identifiers() -> None:
    synthetic = pd.DataFrame(
        {"record_key": ["keep-low", "keep-medium", "remove-high"], "value": [1, 2, 3]}
    )
    store = RowStore()
    for band in ("low", "medium", "high"):
        store.add_profile({"composite_band": band, "dcr_train": 0.123})

    cleaned = remove_above_band(synthetic, store, max_band="medium")

    assert cleaned["value"].tolist() == [1, 2]
    assert list(cleaned.columns) == list(synthetic.columns)
    assert "remove-high" not in cleaned["record_key"].tolist()


def test_remediation_requires_profiles_aligned_to_synthetic_rows() -> None:
    store = RowStore()
    store.add_profile({"composite_band": "low"})

    with pytest.raises(ValueError, match="row count"):
        remove_above_band(
            pd.DataFrame({"value": [1, 2]}),
            store,
            max_band="medium",
        )


def test_remediation_rejects_invalid_band() -> None:
    with pytest.raises(ValueError, match="max_band"):
        remove_above_band(pd.DataFrame({"value": [1]}), RowStore(), max_band="critical")

