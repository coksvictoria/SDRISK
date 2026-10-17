"""Tests for chunked and blocked nearest-neighbor search."""

import numpy as np
import pandas as pd
import pytest

from sdrisk.distance.gower import GowerDistance
from sdrisk.distance.nn_search import nearest_neighbor_distances
from sdrisk.schema import Schema, Variable


def setup_distance() -> tuple[GowerDistance, pd.DataFrame, pd.DataFrame]:
    schema = Schema(
        variables=[
            Variable(name="x", vtype="numeric", role=["qi"]),
            Variable(name="block", vtype="categorical", role=["qi"]),
        ]
    )
    reference = pd.DataFrame(
        {"x": [0.0, 10.0, 4.0], "block": ["a", "a", "b"]}
    )
    query = pd.DataFrame({"x": [1.0, 8.0, 5.0], "block": ["a", "a", "b"]})
    distance = GowerDistance(schema, columns=["x"]).fit(reference)
    return distance, reference, query


def test_chunked_nearest_neighbors_match_brute_force_reference() -> None:
    distance, reference, query = setup_distance()
    pairwise = distance.pairwise(query, reference)

    actual = nearest_neighbor_distances(
        query, reference, distance, query_chunk_size=1
    )

    np.testing.assert_allclose(actual, pairwise.min(axis=1))


def test_chunk_size_does_not_change_nearest_neighbor_results() -> None:
    distance, reference, query = setup_distance()

    one_at_a_time = nearest_neighbor_distances(
        query, reference, distance, query_chunk_size=1
    )
    all_at_once = nearest_neighbor_distances(
        query, reference, distance, query_chunk_size=len(query)
    )

    np.testing.assert_allclose(one_at_a_time, all_at_once)


def test_blocking_restricts_candidates_to_matching_blocks() -> None:
    distance, reference, query = setup_distance()

    actual = nearest_neighbor_distances(
        query,
        reference,
        distance,
        blocking_columns=["block"],
        fallback_to_global=False,
    )

    np.testing.assert_allclose(actual, np.array([0.1, 0.2, 0.1]))


def test_blocking_falls_back_globally_when_query_has_no_block_match() -> None:
    distance, reference, _ = setup_distance()
    query = pd.DataFrame({"x": [5.0], "block": ["missing"]})

    actual = nearest_neighbor_distances(query, reference, distance)
    no_fallback = nearest_neighbor_distances(
        query, reference, distance, blocking_columns=["block"], fallback_to_global=False
    )

    assert actual.shape == (1,)
    assert np.isfinite(actual[0])
    assert np.isnan(no_fallback[0])


def test_nearest_neighbor_search_handles_empty_query_and_rejects_empty_reference() -> None:
    distance, reference, _ = setup_distance()
    empty_query = pd.DataFrame({"x": pd.Series(dtype=float), "block": pd.Series(dtype=str)})

    result = nearest_neighbor_distances(empty_query, reference, distance)

    assert result.shape == (0,)
    with pytest.raises(ValueError, match="must not be empty"):
        nearest_neighbor_distances(
            pd.DataFrame({"x": [1.0], "block": ["a"]}),
            reference.iloc[:0],
            distance,
        )


def test_leave_one_out_nearest_neighbors_excludes_only_the_aligned_self_row() -> None:
    schema = Schema(variables=[Variable(name="x", vtype="numeric", role=["qi"])])
    data = pd.DataFrame({"x": [0.0, 2.0, 5.0]})
    distance = GowerDistance(schema, columns=["x"]).fit(data)

    actual = nearest_neighbor_distances(data, data, distance, exclude_self=True)

    np.testing.assert_allclose(actual, np.array([0.4, 0.4, 0.6]))


def test_leave_one_out_nearest_neighbor_requires_aligned_frame_lengths() -> None:
    distance, reference, query = setup_distance()

    with pytest.raises(ValueError, match="same number of rows"):
        nearest_neighbor_distances(
            query, reference.iloc[:2], distance, exclude_self=True
        )
