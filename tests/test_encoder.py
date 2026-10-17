"""Hand-computable tests for mixed-type preprocessing."""

import numpy as np
import pandas as pd
import pytest

from sdrisk.prep.encoder import MixedTypeEncoder
from sdrisk.schema import Schema, Variable


def test_encoder_scales_numeric_values_and_indicates_missingness() -> None:
    schema = Schema(
        variables=[
            Variable(name="age", vtype="numeric", role=["qi"]),
            Variable(name="region", vtype="categorical", role=["qi"]),
        ]
    )
    training = pd.DataFrame({"age": [0.0, 10.0, None], "region": ["north", "south", None]})
    encoder = MixedTypeEncoder(schema, columns=["age", "region"]).fit(training)

    encoded = encoder.transform(pd.DataFrame({"age": [5.0, None], "region": ["south", None]}))

    expected = np.array(
        [
            [0.5, 0.0, 0.0, 1.0, 0.0, 0.0],
            [0.5, 1.0, 0.0, 0.0, 1.0, 0.0],
        ]
    )
    np.testing.assert_allclose(encoded, expected)


def test_encoder_marks_unseen_categories_without_losing_the_row() -> None:
    schema = Schema(variables=[Variable(name="kind", vtype="categorical")])
    encoder = MixedTypeEncoder(schema, columns=["kind"]).fit(
        pd.DataFrame({"kind": ["known"]})
    )

    encoded = encoder.transform(pd.DataFrame({"kind": ["new"]}))

    np.testing.assert_array_equal(encoded, np.array([[0.0, 0.0, 1.0]]))


def test_encoder_handles_dates_and_constant_numeric_columns() -> None:
    schema = Schema(
        variables=[
            Variable(name="constant", vtype="numeric"),
            Variable(name="date", vtype="date"),
        ]
    )
    training = pd.DataFrame(
        {"constant": [4.0, 4.0], "date": ["2024-01-01", "2024-01-03"]}
    )
    encoder = MixedTypeEncoder(schema, columns=["constant", "date"]).fit(training)

    encoded = encoder.transform(
        pd.DataFrame({"constant": [9.0], "date": ["2024-01-02"]})
    )

    np.testing.assert_allclose(encoded, np.array([[0.0, 0.0, 0.5, 0.0]]))


def test_encoder_uses_declared_order_for_ordinal_categories() -> None:
    schema = Schema(variables=[Variable(name="grade", vtype="ordinal")])
    training = pd.DataFrame(
        {
            "grade": pd.Categorical(
                ["low", "high", "medium"],
                categories=["low", "medium", "high"],
                ordered=True,
            )
        }
    )
    encoder = MixedTypeEncoder(schema, columns=["grade"]).fit(training)

    encoded = encoder.transform(pd.DataFrame({"grade": ["medium"]}))

    np.testing.assert_allclose(encoded, np.array([[0.5, 0.0]]))


def test_encoder_rejects_unordered_textual_ordinal_values() -> None:
    schema = Schema(variables=[Variable(name="grade", vtype="ordinal")])

    with pytest.raises(ValueError, match="ordered categorical"):
        MixedTypeEncoder(schema, columns=["grade"]).fit(
            pd.DataFrame({"grade": ["low", "high"]})
        )


def test_encoder_rejects_identifier_columns_and_missing_columns() -> None:
    schema = Schema(
        variables=[
            Variable(name="record_id", vtype="categorical", role=["id"]),
            Variable(name="age", vtype="numeric", role=["qi"]),
        ]
    )

    with pytest.raises(ValueError, match="identifier"):
        MixedTypeEncoder(schema, columns=["record_id"])
    with pytest.raises(ValueError, match="missing"):
        MixedTypeEncoder(schema, columns=["age"]).fit(pd.DataFrame({"other": [1]}))
