"""Weighted Gower distance for mixed-type tabular data."""

from collections.abc import Sequence

import numpy as np
import pandas as pd

from sdrisk.prep.values import numeric_values, ordered_categories
from sdrisk.schema import Schema


class GowerDistance:
    """Compute weighted pairwise Gower dissimilarities.

    Numeric, ordinal, and date differences are scaled using ranges fitted on
    the reference data and capped at one. Categorical values use exact
    agreement. Variables missing from either row are omitted from that pair's
    weighted denominator; pairs with no comparable variables receive NaN.
    """

    def __init__(self, schema: Schema, *, columns: Sequence[str] | None = None) -> None:
        schema_variables = {variable.name: variable for variable in schema.variables}
        if columns is None:
            selected = [
                variable.name
                for variable in schema.variables
                if "qi" in variable.role
            ]
        else:
            selected = list(columns)
        if not selected:
            raise ValueError("select at least one feature column")
        if len(selected) != len(set(selected)):
            raise ValueError("feature columns must be unique")
        unknown = [name for name in selected if name not in schema_variables]
        if unknown:
            raise ValueError(f"unknown schema columns: {', '.join(unknown)}")
        forbidden = [
            name
            for name in selected
            if {"id", "ignore"} & set(schema_variables[name].role)
        ]
        if forbidden:
            raise ValueError(
                f"identifier or ignored columns cannot be features: {', '.join(forbidden)}"
            )
        self._variables = tuple(schema_variables[name] for name in selected)
        self._schema_variables = schema_variables
        self._ranges: dict[str, tuple[float, float] | None] = {}
        self._ordinal_categories: dict[str, tuple[object, ...] | None] = {}
        self._fitted = False

    @property
    def columns(self) -> tuple[str, ...]:
        """Return selected feature columns in distance order."""
        return tuple(variable.name for variable in self._variables)

    def validate_columns(self, columns: Sequence[str]) -> tuple[str, ...]:
        """Validate feature or blocking columns against the schema."""
        selected = tuple(columns)
        if not selected:
            raise ValueError("at least one column is required")
        if len(selected) != len(set(selected)):
            raise ValueError("columns must be unique")
        unknown = [name for name in selected if name not in self._schema_variables]
        if unknown:
            raise ValueError(f"unknown schema columns: {', '.join(unknown)}")
        forbidden = [
            name
            for name in selected
            if {"id", "ignore"} & set(self._schema_variables[name].role)
        ]
        if forbidden:
            raise ValueError(
                f"identifier or ignored columns cannot be features: {', '.join(forbidden)}"
            )
        return selected

    def validate_frame(self, data: pd.DataFrame, *, label: str = "dataframe") -> None:
        """Ensure a candidate dataframe contains all selected distance columns."""
        if not isinstance(data, pd.DataFrame):
            raise TypeError(f"{label} must be a pandas DataFrame")
        if not data.columns.is_unique:
            raise ValueError(f"{label} column names must be unique")
        missing = [name for name in self.columns if name not in data.columns]
        if missing:
            raise ValueError(f"{label} is missing columns: {', '.join(missing)}")

    def fit(self, reference: pd.DataFrame) -> "GowerDistance":
        """Fit numeric ranges and ordinal levels on reference data."""
        self.validate_frame(reference, label="reference")
        if reference.empty:
            raise ValueError("reference dataframe must not be empty")
        ranges: dict[str, tuple[float, float] | None] = {}
        ordinal_levels: dict[str, tuple[object, ...] | None] = {}
        for variable in self._variables:
            series = reference[variable.name]
            levels = ordered_categories(series, variable)
            ordinal_levels[variable.name] = levels
            if variable.vtype == "categorical":
                ranges[variable.name] = None
                continue
            values = numeric_values(
                series, variable, ordinal_categories=levels
            )
            observed = values[np.isfinite(values)]
            ranges[variable.name] = (
                (float(observed.min()), float(observed.max()))
                if len(observed)
                else None
            )
        self._ranges = ranges
        self._ordinal_categories = ordinal_levels
        self._fitted = True
        return self

    def pairwise(self, left: pd.DataFrame, right: pd.DataFrame) -> np.ndarray:
        """Return a matrix of weighted dissimilarities for every row pair."""
        if not self._fitted:
            raise RuntimeError("Gower distance must be fit before pairwise calculation")
        self.validate_frame(left, label="left dataframe")
        self.validate_frame(right, label="right dataframe")
        numerator = np.zeros((len(left), len(right)), dtype=np.float64)
        denominator = np.zeros_like(numerator)

        for variable in self._variables:
            left_series = left[variable.name].reset_index(drop=True)
            right_series = right[variable.name].reset_index(drop=True)
            if variable.vtype == "categorical":
                left_codes, right_codes = _factorized_codes(left_series, right_series)
                comparable = (left_codes[:, None] >= 0) & (right_codes[None, :] >= 0)
                dissimilarity = (left_codes[:, None] != right_codes[None, :]).astype(
                    np.float64
                )
            else:
                levels = self._ordinal_categories[variable.name]
                left_values = numeric_values(
                    left_series, variable, ordinal_categories=levels
                )
                right_values = numeric_values(
                    right_series, variable, ordinal_categories=levels
                )
                left_valid = np.isfinite(left_values)
                right_valid = np.isfinite(right_values)
                comparable = left_valid[:, None] & right_valid[None, :]
                reference_range = self._ranges[variable.name]
                if reference_range is None:
                    continue
                minimum, maximum = reference_range
                difference = np.abs(left_values[:, None] - right_values[None, :])
                span = maximum - minimum
                if span == 0:
                    dissimilarity = (difference != 0).astype(np.float64)
                else:
                    dissimilarity = np.minimum(difference / span, 1.0)

            np.multiply(dissimilarity, variable.weight, out=dissimilarity)
            np.copyto(dissimilarity, 0.0, where=~comparable)
            numerator += dissimilarity
            np.add(denominator, variable.weight, out=denominator, where=comparable)

        result = np.full_like(numerator, np.nan)
        np.divide(numerator, denominator, out=result, where=denominator > 0)
        return result


def _factorized_codes(
    left: pd.Series, right: pd.Series
) -> tuple[np.ndarray, np.ndarray]:
    try:
        joined = pd.concat((left, right), ignore_index=True)
        codes, _ = pd.factorize(joined, sort=False, use_na_sentinel=True)
    except (TypeError, ValueError) as error:
        raise ValueError("categorical values must be scalar and hashable") from error
    return codes[: len(left)], codes[len(left) :]
