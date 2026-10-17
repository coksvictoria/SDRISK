"""Tests for deterministic row subsampling."""

import pandas as pd
import pytest

from sdrisk.distance.subsample import subsample_rows
from sdrisk.provenance import make_rng


def test_subsampling_respects_limit_and_records_fraction() -> None:
    frame = pd.DataFrame({"value": range(10)})

    result = subsample_rows(frame, max_rows=4, rng=make_rng(41))

    assert len(result.data) == 4
    assert result.n_original == 10
    assert result.n_sampled == 4
    assert result.sampling_fraction == 0.4


def test_subsampling_is_reproducible_and_preserves_source_order() -> None:
    frame = pd.DataFrame({"value": range(20)})

    first = subsample_rows(frame, max_rows=6, rng=make_rng(9))
    second = subsample_rows(frame, max_rows=6, rng=make_rng(9))

    pd.testing.assert_frame_equal(first.data, second.data)
    assert first.data.index.is_monotonic_increasing


def test_small_and_empty_inputs_are_returned_without_expansion() -> None:
    frame = pd.DataFrame({"value": [1, 2]})

    result = subsample_rows(frame, max_rows=5, rng=make_rng(0))

    pd.testing.assert_frame_equal(result.data, frame)
    assert result.sampling_fraction == 1.0


def test_empty_input_records_zero_sampling_fraction() -> None:
    empty = pd.DataFrame({"value": pd.Series(dtype=int)})

    result = subsample_rows(empty, max_rows=5, rng=make_rng(0))

    assert result.n_original == 0
    assert result.n_sampled == 0
    assert result.sampling_fraction == 0.0


def test_subsampling_requires_positive_limit() -> None:
    with pytest.raises(ValueError, match="positive"):
        subsample_rows(pd.DataFrame({"value": [1]}), max_rows=0, rng=make_rng(0))
