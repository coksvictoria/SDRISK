"""Fitted encoders for mixed-type tabular columns."""

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Literal

import numpy as np
import pandas as pd

from sdrisk.prep.values import is_missing, numeric_values, ordered_categories
from sdrisk.schema import Schema, Variable

_NumericKind = Literal["numeric"]
_CategoricalKind = Literal["categorical"]


@dataclass(frozen=True)
class _ColumnState:
    variable: Variable
    kind: _NumericKind | _CategoricalKind
    minimum: float = 0.0
    maximum: float = 0.0
    median: float = 0.0
    categories: tuple[object, ...] = ()
    category_indices: Mapping[object, int] | None = None


class MixedTypeEncoder:
    """Scale numeric/date/ordinal fields and one-hot encode categorical fields.

    Numeric missing values are imputed with the training median and represented
    by an additional missingness indicator. Categorical missing and unseen
    values receive dedicated one-hot positions.
    """

    def __init__(self, schema: Schema, *, columns: Sequence[str]) -> None:
        if not columns:
            raise ValueError("at least one column must be selected")
        if len(columns) != len(set(columns)):
            raise ValueError("selected columns must be unique")
        variables = {variable.name: variable for variable in schema.variables}
        missing = [name for name in columns if name not in variables]
        if missing:
            raise ValueError(f"unknown schema columns: {', '.join(missing)}")
        forbidden = [
            name
            for name in columns
            if {"id", "ignore"} & set(variables[name].role)
        ]
        if forbidden:
            raise ValueError(
                f"identifier or ignored columns cannot be features: {', '.join(forbidden)}"
            )
        self._variables = tuple(variables[name] for name in columns)
        self._states: tuple[_ColumnState, ...] | None = None
        self._feature_names: tuple[str, ...] = ()

    def fit(self, data: pd.DataFrame) -> "MixedTypeEncoder":
        """Fit scaling ranges and categorical levels using training rows."""
        self._validate_frame(data)
        if data.empty:
            raise ValueError("cannot fit encoder on an empty dataframe")
        states: list[_ColumnState] = []
        feature_names: list[str] = []
        for variable in self._variables:
            series = data[variable.name]
            if variable.vtype == "categorical":
                categories = _observed_categories(series, variable.name)
                indices = {category: index for index, category in enumerate(categories)}
                states.append(
                    _ColumnState(
                        variable=variable,
                        kind="categorical",
                        categories=categories,
                        category_indices=indices,
                    )
                )
                feature_names.extend(
                    f"{variable.name}__level_{index}"
                    for index in range(len(categories) + 2)
                )
                continue

            levels = ordered_categories(series, variable)
            if variable.vtype == "ordinal" and levels is None:
                _require_numeric_ordinal(series, variable)
            values = numeric_values(series, variable, ordinal_categories=levels)
            observed = values[np.isfinite(values)]
            minimum = float(observed.min()) if len(observed) else 0.0
            maximum = float(observed.max()) if len(observed) else 0.0
            median = float(np.median(observed)) if len(observed) else 0.0
            states.append(
                _ColumnState(
                    variable=variable,
                    kind="numeric",
                    minimum=minimum,
                    maximum=maximum,
                    median=median,
                    categories=levels or (),
                )
            )
            feature_names.extend(
                (f"{variable.name}__scaled", f"{variable.name}__missing")
            )
        self._states = tuple(states)
        self._feature_names = tuple(feature_names)
        return self

    def transform(self, data: pd.DataFrame) -> np.ndarray:
        """Encode a dataframe using previously fitted training statistics."""
        self._require_fitted()
        self._validate_frame(data)
        encoded = np.zeros((len(data), len(self._feature_names)), dtype=np.float64)
        offset = 0
        for state in self._states or ():
            series = data[state.variable.name]
            if state.kind == "categorical":
                categories = state.categories
                category_indices = state.category_indices
                if category_indices is None:
                    raise RuntimeError("categorical encoder state is incomplete")
                for row_index, value in enumerate(series.tolist()):
                    if is_missing(value):
                        column_index = len(categories)
                    else:
                        try:
                            column_index = category_indices[value]
                        except (KeyError, TypeError):
                            column_index = len(categories) + 1
                    encoded[row_index, offset + column_index] = 1.0
                offset += len(categories) + 2
                continue

            values = numeric_values(
                series,
                state.variable,
                ordinal_categories=state.categories or None,
            )
            missing = ~np.isfinite(values)
            imputed = np.where(missing, state.median, values)
            span = state.maximum - state.minimum
            if span == 0:
                scaled = np.zeros(len(data), dtype=np.float64)
            else:
                scaled = (imputed - state.minimum) / span
            encoded[:, offset] = scaled
            encoded[:, offset + 1] = missing.astype(np.float64)
            offset += 2
        return encoded

    def fit_transform(self, data: pd.DataFrame) -> np.ndarray:
        """Fit on and encode a training dataframe."""
        return self.fit(data).transform(data)

    @property
    def feature_names(self) -> tuple[str, ...]:
        """Return generated feature names after fitting."""
        self._require_fitted()
        return self._feature_names

    def _validate_frame(self, data: pd.DataFrame) -> None:
        if not isinstance(data, pd.DataFrame):
            raise TypeError("data must be a pandas DataFrame")
        if not data.columns.is_unique:
            raise ValueError("dataframe column names must be unique")
        missing = [
            variable.name
            for variable in self._variables
            if variable.name not in data.columns
        ]
        if missing:
            raise ValueError(f"dataframe is missing columns: {', '.join(missing)}")

    def _require_fitted(self) -> None:
        if self._states is None:
            raise RuntimeError("encoder must be fit before transform")


def _observed_categories(series: pd.Series, name: str) -> tuple[object, ...]:
    categories: list[object] = []
    indices: dict[object, int] = {}
    for value in series.tolist():
        if is_missing(value):
            continue
        try:
            if value not in indices:
                indices[value] = len(categories)
                categories.append(value)
        except TypeError as error:
            raise ValueError(f"{name!r} categorical values must be hashable") from error
    return tuple(categories)


def _require_numeric_ordinal(series: pd.Series, variable: Variable) -> None:
    try:
        numeric_values(series, variable)
    except ValueError as error:
        raise ValueError(
            f"{variable.name!r} textual ordinal values require an ordered categorical dtype"
        ) from error
