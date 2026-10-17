"""Reproducible row subsampling."""

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class SubsampledData:
    """A sampled dataframe and its auditable sampling fraction."""

    data: pd.DataFrame
    n_original: int
    n_sampled: int
    sampling_fraction: float


def subsample_rows(
    data: pd.DataFrame, *, max_rows: int, rng: np.random.Generator
) -> SubsampledData:
    """Sample without replacement while retaining source row order."""
    if isinstance(max_rows, bool) or not isinstance(max_rows, int) or max_rows < 1:
        raise ValueError("max_rows must be a positive integer")
    if not isinstance(data, pd.DataFrame):
        raise TypeError("data must be a pandas DataFrame")
    if not isinstance(rng, np.random.Generator):
        raise TypeError("rng must be a numpy.random.Generator")

    n_original = len(data)
    n_sampled = min(max_rows, n_original)
    if n_sampled == n_original:
        sampled = data.copy()
    else:
        positions = np.sort(rng.choice(n_original, size=n_sampled, replace=False))
        sampled = data.iloc[positions].copy()
    fraction = n_sampled / n_original if n_original else 0.0
    return SubsampledData(
        data=sampled,
        n_original=n_original,
        n_sampled=n_sampled,
        sampling_fraction=fraction,
    )
