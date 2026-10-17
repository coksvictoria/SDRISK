"""Tests for schema validation."""

import pytest
from pydantic import ValidationError

from sdrisk.schema import Schema, Variable


def test_schema_accepts_valid_variables_and_multiple_roles() -> None:
    schema = Schema(
        variables=[
            Variable(name="age", vtype="numeric", role=["qi"], tolerance=1.0),
            Variable(name="region", vtype="categorical", role=["qi", "stratum"]),
        ]
    )

    assert schema.variable_names == ("age", "region")
    assert schema.variables[1].role == ["qi", "stratum"]


@pytest.mark.parametrize(
    "kwargs",
    [
        {"name": "", "vtype": "numeric"},
        {"name": "x", "vtype": "unsupported"},
        {"name": "x", "vtype": "numeric", "weight": 0},
        {"name": "x", "vtype": "numeric", "tolerance": -1},
        {"name": "x", "vtype": "categorical", "tolerance": 0.1},
    ],
)
def test_variable_rejects_invalid_values(kwargs: dict[str, object]) -> None:
    with pytest.raises(ValidationError):
        Variable(**kwargs)


def test_schema_rejects_duplicate_variable_names() -> None:
    with pytest.raises(ValidationError, match="unique"):
        Schema(
            variables=[
                Variable(name="age", vtype="numeric"),
                Variable(name="age", vtype="categorical"),
            ]
        )
