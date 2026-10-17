"""Distance-to-closest-record identity metrics."""

from collections.abc import Sequence

import numpy as np
import pandas as pd

from sdrisk.bundle import DatasetBundle
from sdrisk.distance.gower import GowerDistance
from sdrisk.distance.nn_search import nearest_neighbor_distances
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


def dcr_overfitting(
    bundle: DatasetBundle, schema: Schema, profile: PolicyProfile
) -> list[MetricResult]:
    """Compare synthetic nearest distances to training and holdout references.

    Definition
        Among synthetic rows whose train and holdout nearest distances are
        finite and non-tied, share closer to `real_train` than to `real_test`.
        Numerical ties are excluded from the denominator.
    Inputs
        `real_train`, `synthetic`, and disjoint `real_test`.
    Baseline
        0.5, expected when synthetic records are equally close to both samples.
    Interpretation
        A share materially above 0.5 is consistent with closer proximity to the
        generator's training data; this diagnostic alone does not prove copying.
    Weaknesses
        Results depend on QIs, distance weights, holdout representativeness, and
        excluding ties. This reports the raw closer-to-training share specified
        by the project brief, not SDMetrics' transformed protection score.
    Citation
        SDMetrics DCROverfittingProtection metric documentation,
        https://docs.sdv.dev/sdmetrics/data-metrics/privacy/dcroverfittingprotection.
    """
    qsets = qi_sets(schema, profile)
    if bundle.real_test is None:
        return [
            filter_result(
                not_assessed(
                    "dcr_overfit_share",
                    "no holdout supplied",
                    group=qi_set.name,
                ),
                profile,
            )
            for qi_set in qsets
        ]
    if (
        bundle.real_train.empty
        or bundle.synthetic.empty
        or bundle.real_test.empty
    ):
        return [
            filter_result(
                not_assessed(
                    "dcr_overfit_share",
                    "real_train, synthetic, and real_test must all contain records",
                    group=qi_set.name,
                ),
                profile,
            )
            for qi_set in qsets
        ]

    results: list[MetricResult] = []
    for qi_set in qsets:
        distance = GowerDistance(schema, columns=qi_set.columns).fit(bundle.real_train)
        train_distances = nearest_neighbor_distances(
            bundle.synthetic, bundle.real_train, distance
        )
        test_distances = nearest_neighbor_distances(
            bundle.synthetic, bundle.real_test, distance
        )
        comparable = (
            np.isfinite(train_distances)
            & np.isfinite(test_distances)
            & ~np.isclose(train_distances, test_distances, rtol=1e-12, atol=1e-12)
        )
        n_records = int(comparable.sum())
        if n_records == 0:
            results.append(
                filter_result(
                    not_assessed(
                        "dcr_overfit_share",
                        "all comparable train and holdout distances tie or are missing",
                        group=qi_set.name,
                    ),
                    profile,
                )
            )
            continue
        train_closer = comparable & (train_distances < test_distances)
        n_flagged = int(train_closer.sum())
        result = assessed_result(
            "dcr_overfit_share",
            n_flagged / n_records,
            0.5,
            wilson_interval(n_flagged, n_records),
            n_records,
            n_flagged,
            group=qi_set.name,
        )
        results.append(filter_result(result, profile))
    return results


def dcr_vs_random(
    bundle: DatasetBundle,
    schema: Schema,
    profile: PolicyProfile,
    rng: np.random.Generator,
) -> list[MetricResult]:
    """Compare synthetic nearest-distance median to a marginal-only control.

    Definition
        Median synthetic-to-training nearest distance divided by the median
        nearest distance for an independent sample drawn from each training
        column's empirical marginal distribution.
    Inputs
        `real_train` and `synthetic`, plus an explicit seeded NumPy generator.
    Baseline
        1.0, the ratio for the marginal-only control itself.
    Interpretation
        A value below one means synthetic rows are closer to training records
        than the independent marginal sample under the chosen distance.
    Weaknesses
        Independent marginals destroy dependencies and can create implausible
        controls. A zero random median leaves the ratio undefined; exact median
        components are not exposed by the result.
    Citation
        SDMetrics DCRBaselineProtection metric documentation,
        https://docs.sdv.dev/sdmetrics/data-metrics/privacy/dcrbaselineprotection.
    """
    if not isinstance(rng, np.random.Generator):
        raise TypeError("rng must be a numpy.random.Generator")
    qsets = qi_sets(schema, profile)
    if bundle.real_train.empty or bundle.synthetic.empty:
        return [
            filter_result(
                not_assessed(
                    "dcr_vs_random_ratio",
                    "real_train and synthetic must both contain records",
                    group=qi_set.name,
                ),
                profile,
            )
            for qi_set in qsets
        ]

    results: list[MetricResult] = []
    for qi_set in qsets:
        random_sample = _marginal_sample(
            bundle.real_train.loc[:, list(qi_set.columns)],
            len(bundle.synthetic),
            rng,
        )
        distance = GowerDistance(schema, columns=qi_set.columns).fit(bundle.real_train)
        synthetic_distances = nearest_neighbor_distances(
            bundle.synthetic, bundle.real_train, distance
        )
        random_distances = nearest_neighbor_distances(
            random_sample, bundle.real_train, distance
        )
        synthetic_median = _finite_median(synthetic_distances)
        random_median = _finite_median(random_distances)
        if synthetic_median is None:
            result = not_assessed(
                "dcr_vs_random_ratio",
                "synthetic nearest distances have no comparable values",
                group=qi_set.name,
            )
        elif random_median is None:
            result = not_assessed(
                "dcr_vs_random_ratio",
                "random marginal nearest distances have no comparable values",
                group=qi_set.name,
            )
        elif random_median == 0:
            synthetic_component = "zero" if synthetic_median == 0 else "positive"
            result = not_assessed(
                "dcr_vs_random_ratio",
                "random marginal baseline median is zero; synthetic median component "
                f"is {synthetic_component}; the ratio is undefined",
                group=qi_set.name,
            )
        else:
            result = assessed_result(
                "dcr_vs_random_ratio",
                synthetic_median / random_median,
                1.0,
                None,
                len(bundle.synthetic),
                None,
                group=qi_set.name,
            )
        results.append(filter_result(result, profile))
    return results


def _marginal_sample(
    reference: pd.DataFrame, n_rows: int, rng: np.random.Generator
) -> pd.DataFrame:
    sampled: pd.DataFrame = pd.DataFrame(index=range(n_rows))
    for column in reference.columns:
        positions = rng.integers(0, len(reference), size=n_rows)
        sampled[column] = reference[column].iloc[positions].reset_index(drop=True)
    return sampled


def _finite_median(values: Sequence[float] | np.ndarray) -> float | None:
    finite = np.asarray(values, dtype=np.float64)
    finite = finite[np.isfinite(finite)]
    return float(np.median(finite)) if len(finite) else None
