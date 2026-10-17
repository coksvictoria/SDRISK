"""Chunked exact nearest-neighbor distances."""

from collections.abc import Sequence

import numpy as np
import pandas as pd

from sdrisk.distance.blocking import block_key, build_block_index
from sdrisk.distance.gower import GowerDistance


def nearest_neighbor_distances(
    query: pd.DataFrame,
    reference: pd.DataFrame,
    distance: GowerDistance,
    *,
    query_chunk_size: int = 8,
    blocking_columns: Sequence[str] | None = None,
    fallback_to_global: bool = True,
    exclude_self: bool = False,
) -> np.ndarray:
    """Find exact nearest-reference distances with bounded pairwise matrices.

    When blocking is enabled, each query is compared with reference rows whose
    blocking values match exactly. If a block is empty, the default behavior is
    to fall back to the full reference set; disable that fallback to return NaN
    for unmatched blocks. With `exclude_self=True`, query and reference rows must
    be positionally aligned and the same-position reference is omitted.
    """
    if isinstance(query_chunk_size, bool) or not isinstance(query_chunk_size, int):
        raise ValueError("query_chunk_size must be a positive integer")
    if query_chunk_size < 1:
        raise ValueError("query_chunk_size must be a positive integer")
    distance.validate_frame(query, label="query dataframe")
    distance.validate_frame(reference, label="reference dataframe")
    if exclude_self and len(query) != len(reference):
        raise ValueError(
            "exclude_self requires query and reference to have the same number of rows"
        )
    if reference.empty:
        raise ValueError("reference dataframe must not be empty")
    if query.empty:
        return np.empty(0, dtype=np.float64)

    blocking = tuple(blocking_columns or ())
    index: dict[tuple[object, ...], np.ndarray] | None = None
    if blocking:
        distance.validate_columns(blocking)
        index = build_block_index(reference, blocking)

    results = np.full(len(query), np.nan, dtype=np.float64)
    all_reference_positions = np.arange(len(reference), dtype=np.intp)
    for chunk_start in range(0, len(query), query_chunk_size):
        chunk_end = min(chunk_start + query_chunk_size, len(query))
        if index is None:
            matrix = distance.pairwise(query.iloc[chunk_start:chunk_end], reference)
            if exclude_self:
                matrix[
                    np.arange(chunk_end - chunk_start),
                    np.arange(chunk_start, chunk_end),
                ] = np.inf
            results[chunk_start:chunk_end] = _row_minima(matrix)
            continue

        for query_position in range(chunk_start, chunk_end):
            row = query.iloc[query_position]
            key = block_key(row.loc[list(blocking)].tolist())
            candidate_positions = index.get(key)
            if candidate_positions is None:
                if not fallback_to_global:
                    continue
                candidate_positions = all_reference_positions
            matrix = distance.pairwise(
                query.iloc[[query_position]],
                reference.iloc[candidate_positions],
            )
            if exclude_self:
                self_positions = np.flatnonzero(candidate_positions == query_position)
                matrix[0, self_positions] = np.inf
            results[query_position] = _row_minima(matrix)[0]
    return results


def _row_minima(matrix: np.ndarray) -> np.ndarray:
    finite = np.isfinite(matrix)
    minima: np.ndarray = np.asarray(
        np.min(np.where(finite, matrix, np.inf), axis=1), dtype=np.float64
    )
    minima[np.isinf(minima)] = np.nan
    return minima
