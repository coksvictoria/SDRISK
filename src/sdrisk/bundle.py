"""Validated input data bundles."""

from dataclasses import dataclass

import pandas as pd


@dataclass(frozen=True)
class DatasetBundle:
    """Hold aligned real and synthetic tabular data.

    The bundle checks column names and pandas dtypes across all inputs. An exact
    row appearing in both real partitions is rejected because the holdout must
    be disjoint from the generator's training data.
    """

    real_train: pd.DataFrame
    synthetic: pd.DataFrame
    real_test: pd.DataFrame | None = None

    def __post_init__(self) -> None:
        frames: dict[str, pd.DataFrame] = {
            "real_train": self.real_train,
            "synthetic": self.synthetic,
        }
        if self.real_test is not None:
            frames["real_test"] = self.real_test

        for name, frame in frames.items():
            if not isinstance(frame, pd.DataFrame):
                raise TypeError(f"{name} must be a pandas DataFrame")
            if not frame.columns.is_unique:
                raise ValueError(f"{name} column names must be unique")
            if any(not isinstance(column, str) or not column for column in frame.columns):
                raise ValueError(f"{name} column names must be non-empty strings")

        expected_columns = list(self.real_train.columns)
        if not expected_columns:
            raise ValueError("dataframes must contain at least one column")
        expected_dtypes = tuple(str(dtype) for dtype in self.real_train.dtypes)
        for name, frame in frames.items():
            if list(frame.columns) != expected_columns:
                raise ValueError(
                    f"{name} columns must match real_train in name and order"
                )
            actual_dtypes = tuple(str(dtype) for dtype in frame.dtypes)
            if actual_dtypes != expected_dtypes:
                raise ValueError(f"{name} dtypes must match real_train")

        if self.real_test is not None and _has_exact_row_overlap(
            self.real_train, self.real_test
        ):
            raise ValueError("real_train and real_test must be disjoint")


def _has_exact_row_overlap(left: pd.DataFrame, right: pd.DataFrame) -> bool:
    if left.empty or right.empty:
        return False
    left_hashes = pd.util.hash_pandas_object(
        left, index=False, categorize=True
    )
    right_hashes = pd.util.hash_pandas_object(
        right, index=False, categorize=True
    )
    common_hashes = set(left_hashes) & set(right_hashes)
    if not common_hashes:
        return False

    columns = list(left.columns)
    left_candidates = left.loc[left_hashes.isin(common_hashes), columns].drop_duplicates()
    right_candidates = right.loc[
        right_hashes.isin(common_hashes), columns
    ].drop_duplicates()
    return not left_candidates.merge(
        right_candidates, on=columns, how="inner", validate="one_to_one"
    ).empty
