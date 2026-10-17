"""Equivalence-class k-profile identity metric."""

import numpy as np
import pandas as pd

from sdrisk.bundle import DatasetBundle
from sdrisk.metrics.identity.common import (
    assessed_result,
    filter_result,
    not_assessed,
    qi_sets,
    wilson_interval,
)
from sdrisk.policy import PolicyProfile
from sdrisk.results import MetricResult
from sdrisk.schema import Schema


def k_profile(
    bundle: DatasetBundle, schema: Schema, profile: PolicyProfile
) -> list[MetricResult]:
    """Compare synthetic and original low-k record shares.

    Definition
        For each policy-configured k cutoff, report the share of synthetic
        records whose exact QI equivalence class contains fewer than k records.
    Inputs
        `real_train` and `synthetic`, for each configured QI set.
    Baseline
        The corresponding share among real-training rows.
    Interpretation
        A larger synthetic low-k share than the original baseline indicates
        more synthetic records in small exact QI classes.
    Weaknesses
        Exact class sizes ignore numeric tolerance and may be sensitive to
        discretization, missingness, and QI selection.
    Citation
        NIST SDNist benchmark and evaluation tools,
        https://github.com/usnistgov/SDNist. This is a policy-configured
        low-k profile, not an SDNist score.
    """
    qsets = qi_sets(schema, profile)
    if bundle.real_train.empty or bundle.synthetic.empty:
        return [
            filter_result(
                not_assessed(
                    "k_profile",
                    "real_train and synthetic must both contain records",
                    group=qi_set.name,
                ),
                profile,
            )
            for qi_set in qsets
        ]

    results: list[MetricResult] = []
    for qi_set in qsets:
        train_sizes = _row_class_sizes(bundle.real_train, qi_set.columns)
        synthetic_sizes = _row_class_sizes(bundle.synthetic, qi_set.columns)
        for threshold in profile.identity.k_profile_thresholds:
            train_flagged = train_sizes < threshold
            synthetic_flagged = synthetic_sizes < threshold
            n_flagged = int(synthetic_flagged.sum())
            n_baseline_flagged = int(train_flagged.sum())
            name = f"k_profile_below_{threshold}"
            result = assessed_result(
                name,
                n_flagged / len(bundle.synthetic),
                n_baseline_flagged / len(bundle.real_train),
                wilson_interval(n_flagged, len(bundle.synthetic)),
                len(bundle.synthetic),
                n_flagged,
                group=qi_set.name,
            )
            results.append(
                filter_result(
                    result, profile, contributing_counts=[n_baseline_flagged]
                )
            )
    return results


def _row_class_sizes(data: pd.DataFrame, columns: list[str]) -> np.ndarray:
    selected = data.loc[:, list(columns)]
    sizes = selected.groupby(
        list(columns),
        dropna=False,
        sort=False,
        observed=True,
    ).transform("size")
    result: np.ndarray = np.asarray(sizes.to_numpy(), dtype=np.int64)
    return result
