"""Tests for disclosure-safe output filtering."""

from hypothesis import given
from hypothesis import strategies as st

from sdrisk.output_filter import OutputFilter
from sdrisk.results import MetricResult


def metric_result(
    *, n_records: int = 30, n_flagged: int = 12, value: float = 0.4
) -> MetricResult:
    return MetricResult(
        name="risk",
        level="table",
        value=value,
        baseline=0.1,
        ci=(0.2, 0.6),
        n_records=n_records,
        n_flagged=n_flagged,
        status="review",
    )


def test_count_is_suppressed_below_minimum_and_rounded_otherwise() -> None:
    output_filter = OutputFilter(min_cell_size=6, rounding_base=3)

    assert output_filter.filter_count(5).value is None
    assert output_filter.filter_count(5).status == "suppressed"
    assert output_filter.filter_count(10).value == 9


def test_metric_is_suppressed_if_a_contributing_count_is_small() -> None:
    output_filter = OutputFilter(min_cell_size=6, rounding_base=3)

    result = output_filter.filter_metric_result(metric_result(n_flagged=2))

    assert result.status == "suppressed"
    assert result.reason
    assert result.value is None
    assert result.baseline is None
    assert result.n_records is None
    assert result.n_flagged is None


def test_metric_counts_are_rounded_before_export() -> None:
    result = OutputFilter(min_cell_size=6, rounding_base=3).filter_metric_result(
        metric_result(n_records=31, n_flagged=12)
    )

    assert result.n_records == 30
    assert result.n_flagged == 12


def test_secondary_suppression_blocks_recovery_from_total() -> None:
    result = OutputFilter(min_cell_size=6, rounding_base=1).filter_count_table(
        {"small": 2, "other": 10, "third": 20, "total": 32},
        total_key="total",
    )

    assert result["small"].status == "suppressed"
    assert result["other"].status == "suppressed"
    assert result["third"].value == 20


@given(st.integers(min_value=0, max_value=100_000))
def test_no_unsuppressed_count_is_below_minimum(count: int) -> None:
    filtered = OutputFilter(min_cell_size=6, rounding_base=3).filter_count(count)

    assert filtered.value is None or filtered.value >= 6


@given(
    n_records=st.integers(min_value=0, max_value=500),
    n_flagged=st.integers(min_value=0, max_value=500),
)
def test_small_metric_counts_never_leave_an_unsuppressed_rate(
    n_records: int, n_flagged: int
) -> None:
    n_flagged = min(n_flagged, n_records)
    source = metric_result(n_records=n_records, n_flagged=n_flagged)

    filtered = OutputFilter(min_cell_size=6, rounding_base=3).filter_metric_result(source)

    if n_records < 6 or n_flagged < 6:
        assert filtered.status == "suppressed"
        assert filtered.value is None
