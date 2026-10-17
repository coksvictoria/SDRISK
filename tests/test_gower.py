"""Hand-computable tests for mixed-type Gower distance."""

import numpy as np
import pandas as pd
import pytest

from sdrisk.distance.gower import GowerDistance
from sdrisk.schema import Schema, Variable


def test_gower_distance_matches_hand_computed_mixed_type_values() -> None:
    schema = Schema(
        variables=[
            Variable(name="x", vtype="numeric", role=["qi"], weight=2.0),
            Variable(name="group", vtype="categorical", role=["qi"], weight=1.0),
        ]
    )
    reference = pd.DataFrame({"x": [0.0, 10.0], "group": ["a", "b"]})
    distance = GowerDistance(schema, columns=["x", "group"]).fit(reference)
    query = pd.DataFrame({"x": [5.0], "group": ["a"]})

    actual = distance.pairwise(query, reference)

    np.testing.assert_allclose(actual, np.array([[1 / 3, 2 / 3]]))


def test_gower_ignores_missing_dimensions_and_returns_nan_when_none_overlap() -> None:
    schema = Schema(
        variables=[
            Variable(name="x", vtype="numeric", role=["qi"]),
            Variable(name="group", vtype="categorical", role=["qi"]),
        ]
    )
    reference = pd.DataFrame({"x": [0.0, 10.0], "group": ["a", "b"]})
    distance = GowerDistance(schema, columns=["x", "group"]).fit(reference)
    query = pd.DataFrame({"x": [np.nan, np.nan], "group": ["a", None]})

    actual = distance.pairwise(query, reference)

    assert actual[0, 0] == 0.0
    assert actual[0, 1] == 1.0
    assert np.isnan(actual[1]).all()


def test_constant_numeric_column_uses_binary_mismatch() -> None:
    schema = Schema(variables=[Variable(name="x", vtype="numeric", role=["qi"])])
    distance = GowerDistance(schema, columns=["x"]).fit(pd.DataFrame({"x": [2.0, 2.0]}))

    actual = distance.pairwise(
        pd.DataFrame({"x": [2.0, 3.0]}), pd.DataFrame({"x": [2.0, 3.0]})
    )

    np.testing.assert_allclose(actual, np.array([[0.0, 1.0], [1.0, 0.0]]))


def test_gower_distance_supports_dates() -> None:
    schema = Schema(variables=[Variable(name="date", vtype="date", role=["qi"])])
    reference = pd.DataFrame({"date": ["2024-01-01", "2024-01-03"]})
    distance = GowerDistance(schema).fit(reference)

    actual = distance.pairwise(
        pd.DataFrame({"date": ["2024-01-02"]}), reference
    )

    np.testing.assert_allclose(actual, np.array([[0.5, 0.5]]))


def test_gower_requires_fit_and_rejects_identifier_features() -> None:
    schema = Schema(
        variables=[Variable(name="id", vtype="numeric", role=["id"])]
    )

    with pytest.raises(ValueError, match="identifier"):
        GowerDistance(schema, columns=["id"])

    distance = GowerDistance(
        Schema(variables=[Variable(name="x", vtype="numeric", role=["qi"])]),
        columns=["x"],
    )
    with pytest.raises(RuntimeError, match="fit"):
        distance.pairwise(pd.DataFrame({"x": [1]}), pd.DataFrame({"x": [1]}))
