"""Tests for deterministic provenance helpers."""

from pathlib import Path

import numpy as np

from sdrisk.provenance import config_sha256, file_sha256, make_rng


def test_config_hash_ignores_mapping_key_order() -> None:
    assert config_sha256({"a": 1, "b": [2, 3]}) == config_sha256(
        {"b": [2, 3], "a": 1}
    )


def test_file_hash_is_sha256_of_file_contents(tmp_path: Path) -> None:
    path = tmp_path / "input.csv"
    path.write_bytes(b"public test fixture")

    assert file_sha256(path) == "8a9d59ae6798b4543dc7cb81d0df1417c105bdd58bd506b7c539ed823fb1b3a8"


def test_rng_is_reproducible_for_explicit_seed() -> None:
    first = make_rng(42)
    second = make_rng(42)

    np.testing.assert_array_equal(first.integers(0, 100, 5), second.integers(0, 100, 5))
