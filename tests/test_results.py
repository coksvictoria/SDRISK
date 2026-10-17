"""Tests for result and internal row-store models."""

import pytest

from sdrisk.results import MetricResult, RowStore


def test_metric_result_requires_reason_when_not_assessed() -> None:
    with pytest.raises(ValueError, match="reason"):
        MetricResult(
            name="membership",
            level="table",
            value=None,
            baseline=None,
            ci=None,
            n_records=0,
            n_flagged=None,
            status="not_assessed",
        )


def test_metric_result_rejects_invalid_counts() -> None:
    with pytest.raises(ValueError, match="n_flagged"):
        MetricResult(
            name="match",
            level="table",
            value=0.5,
            baseline=0.1,
            ci=None,
            n_records=2,
            n_flagged=3,
            status="review",
        )


def test_row_store_exposes_only_filtered_band_aggregates() -> None:
    store = RowStore()
    store.add_profile({"composite_band": "low", "private_value": 7})
    store.add_profile({"composite_band": "high", "private_value": 9})

    from sdrisk.output_filter import OutputFilter

    aggregate = store.aggregate_band_counts(OutputFilter(min_cell_size=2, rounding_base=1))

    assert aggregate["low"].value is None
    assert aggregate["low"].status == "suppressed"
    assert aggregate["high"].value is None
    assert aggregate["high"].status == "suppressed"
    assert not hasattr(store, "to_dict")
