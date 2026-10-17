"""Shared conversion of numeric, ordinal, and date values."""

from collections.abc import Sequence
from typing import Any, cast

import numpy as np
import pandas as pd

from sdrisk.schema import Variable

_NANOSECONDS_PER_DAY = 86_400_000_000_000


def numeric_values(
    series: pd.Series,
    variable: Variable,
    *,
    ordinal_categories: Sequence[object] | None = None,
) -> np.ndarray:
    """Convert a supported ordered variable to finite floats or NaN."""
    if variable.vtype == "date":
        try:
            dates = pd.to_datetime(series, errors="raise", utc=True)
        except (TypeError, ValueError) as error:
            raise ValueError(f"{variable.name!r} contains invalid date values") from error
        values = dates.astype("int64").to_numpy(dtype=np.float64)
        values /= _NANOSECONDS_PER_DAY
        values[dates.isna().to_numpy()] = np.nan
        return values

    if variable.vtype == "ordinal" and ordinal_categories is not None:
        category_ranks = {
            category: float(index) for index, category in enumerate(ordinal_categories)
        }
        converted = np.full(len(series), np.nan, dtype=np.float64)
        for index, value in enumerate(series.tolist()):
            if is_missing(value):
                continue
            try:
                converted[index] = category_ranks[value]
            except (KeyError, TypeError) as error:
                raise ValueError(
                    f"{variable.name!r} contains a value outside its ordered categories"
                ) from error
        return converted

    try:
        converted_series = pd.to_numeric(series, errors="raise")
        values = converted_series.to_numpy(dtype=np.float64, na_value=np.nan)
    except (TypeError, ValueError) as error:
        message = (
            "must be numeric or use an ordered pandas categorical"
            if variable.vtype == "ordinal"
            else "must contain numeric values"
        )
        raise ValueError(f"{variable.name!r} {message}") from error
    if np.isinf(values).any():
        raise ValueError(f"{variable.name!r} must not contain infinite values")
    return values


def ordered_categories(series: pd.Series, variable: Variable) -> tuple[object, ...] | None:
    """Return declared order for ordered pandas categoricals, if present."""
    if variable.vtype != "ordinal" or not isinstance(series.dtype, pd.CategoricalDtype):
        return None
    if not series.cat.ordered:
        return None
    return tuple(series.cat.categories.tolist())


def is_missing(value: object) -> bool:
    """Return whether a scalar pandas value is missing."""
    missing = pd.isna(cast(Any, value))
    if not isinstance(missing, (bool, np.bool_)):
        raise ValueError("tabular values must be scalar")
    return bool(missing)
