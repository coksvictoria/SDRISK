"""Exact-value blocking helpers for candidate selection."""

from collections.abc import Sequence

import numpy as np
import pandas as pd

from sdrisk.prep.values import is_missing


class _MissingKey:
    pass


_MISSING = _MissingKey()


def block_key(values: Sequence[object]) -> tuple[object, ...]:
    """Create a stable key that treats all pandas missing values alike."""
    key: list[object] = []
    for value in values:
        if is_missing(value):
            key.append(_MISSING)
        else:
            try:
                hash(value)
            except TypeError as error:
                raise ValueError("blocking values must be hashable") from error
            key.append(value)
    return tuple(key)


def build_block_index(
    reference: pd.DataFrame, columns: Sequence[str]
) -> dict[tuple[object, ...], np.ndarray]:
    """Index reference-row positions by their exact blocking values."""
    if not columns:
        raise ValueError("blocking requires at least one column")
    if len(columns) != len(set(columns)):
        raise ValueError("blocking columns must be unique")
    if not reference.columns.is_unique:
        raise ValueError("reference column names must be unique")
    missing = [column for column in columns if column not in reference.columns]
    if missing:
        raise ValueError(f"reference is missing blocking columns: {', '.join(missing)}")
    mutable_index: dict[tuple[object, ...], list[int]] = {}
    selected = reference.loc[:, list(columns)]
    for row_index, values in enumerate(selected.itertuples(index=False, name=None)):
        mutable_index.setdefault(block_key(values), []).append(row_index)
    return {
        key: np.asarray(row_indices, dtype=np.intp)
        for key, row_indices in mutable_index.items()
    }
