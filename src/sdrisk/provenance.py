"""Deterministic hashing and run provenance helpers."""

import hashlib
import json
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from pydantic import BaseModel

from sdrisk import __version__


def config_sha256(config: Mapping[str, Any] | BaseModel) -> str:
    """Hash configuration data using canonical JSON serialization."""
    if isinstance(config, BaseModel):
        payload: object = config.model_dump(mode="json")
    else:
        payload = config
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def file_sha256(path: str | Path, *, chunk_size: int = 1024 * 1024) -> str:
    """Compute a file's SHA-256 digest without loading it all into memory."""
    if chunk_size < 1:
        raise ValueError("chunk_size must be positive")
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def dataframe_sha256(frame: pd.DataFrame) -> str:
    """Hash dataframe contents, column order, and dtypes without writing data."""
    metadata = json.dumps(
        {"columns": list(frame.columns), "dtypes": [str(dtype) for dtype in frame.dtypes]},
        ensure_ascii=False,
        separators=(",", ":"),
    ).encode("utf-8")
    row_hashes = pd.util.hash_pandas_object(frame, index=False, categorize=True)
    digest = hashlib.sha256()
    digest.update(metadata)
    digest.update(row_hashes.to_numpy().tobytes())
    return digest.hexdigest()


def make_rng(seed: int) -> np.random.Generator:
    """Create the package's explicit NumPy random generator."""
    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
        raise ValueError("seed must be a non-negative integer")
    return np.random.default_rng(seed)


@dataclass(frozen=True)
class RunProvenance:
    """Record hashes and explicit run settings without copying input data."""

    config_hash: str
    input_hashes: Mapping[str, str]
    package_version: str
    seed: int
    sampling_fractions: Mapping[str, float]


def build_provenance(
    config: Mapping[str, Any] | BaseModel,
    input_files: Mapping[str, str | Path],
    *,
    seed: int,
    sampling_fractions: Mapping[str, float],
) -> RunProvenance:
    """Build audit provenance from configuration, input files, and sampling."""
    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
        raise ValueError("seed must be a non-negative integer")
    for name, fraction in sampling_fractions.items():
        if not 0 <= fraction <= 1:
            raise ValueError(f"sampling fraction for {name!r} must be between 0 and 1")
    input_hashes = {
        name: file_sha256(path) for name, path in sorted(input_files.items())
    }
    return RunProvenance(
        config_hash=config_sha256(config),
        input_hashes=input_hashes,
        package_version=__version__,
        seed=seed,
        sampling_fractions=dict(sampling_fractions),
    )
