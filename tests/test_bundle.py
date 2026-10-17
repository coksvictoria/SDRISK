"""Tests for dataset bundle validation."""

import pandas as pd
import pytest

from sdrisk.bundle import DatasetBundle


def test_bundle_accepts_aligned_train_synthetic_and_disjoint_test() -> None:
    bundle = DatasetBundle(
        real_train=pd.DataFrame({"age": [20, 30]}),
        synthetic=pd.DataFrame({"age": [21, 31]}),
        real_test=pd.DataFrame({"age": [40]}),
    )

    assert len(bundle.real_train) == 2
    assert bundle.real_test is not None


def test_bundle_rejects_column_mismatch() -> None:
    with pytest.raises(ValueError, match="columns"):
        DatasetBundle(
            real_train=pd.DataFrame({"age": [20]}),
            synthetic=pd.DataFrame({"years": [21]}),
        )


def test_bundle_rejects_dtype_mismatch() -> None:
    with pytest.raises(ValueError, match="dtype"):
        DatasetBundle(
            real_train=pd.DataFrame({"age": [20]}),
            synthetic=pd.DataFrame({"age": ["20"]}),
        )


def test_bundle_rejects_exact_train_test_overlap_including_missing_values() -> None:
    frame = pd.DataFrame({"age": [20, None], "region": ["north", "south"]})

    with pytest.raises(ValueError, match="disjoint"):
        DatasetBundle(
            real_train=frame,
            synthetic=frame.copy(),
            real_test=pd.DataFrame(
                {
                    "age": pd.Series([float("nan")], dtype="float64"),
                    "region": ["south"],
                }
            ),
        )


def test_bundle_rejects_duplicate_column_names() -> None:
    with pytest.raises(ValueError, match="unique"):
        DatasetBundle(
            real_train=pd.DataFrame([[1, 2]], columns=["age", "age"]),
            synthetic=pd.DataFrame([[1, 2]], columns=["age", "age"]),
        )


def test_bundle_rejects_empty_tables_without_columns() -> None:
    with pytest.raises(ValueError, match="at least one column"):
        DatasetBundle(
            real_train=pd.DataFrame(index=[0]),
            synthetic=pd.DataFrame(index=[0]),
        )
