"""Exact and unique exact-match disclosure metrics."""

from sdrisk.bundle import DatasetBundle
from sdrisk.metrics.identity.common import (
    assessed_result,
    filter_result,
    key_counts,
    matching_rows,
    not_assessed,
    qi_sets,
    row_keys,
    wilson_interval,
)
from sdrisk.policy import PolicyProfile
from sdrisk.results import MetricResult
from sdrisk.schema import Schema


def exact_match_rate(
    bundle: DatasetBundle, schema: Schema, profile: PolicyProfile
) -> list[MetricResult]:
    """Calculate exact match rate for each configured quasi-identifier set.

    Definition
        Share of real-training rows having an exact synthetic match on the QI
        columns. A numeric variable's configured schema tolerance is applied.
    Inputs
        `real_train`, `synthetic`, and `real_test`; the holdout provides a
        train-to-test control baseline.
    Baseline
        Share of real-training rows matching a holdout row on the same QIs.
    Interpretation
        A higher value than the holdout baseline indicates more exact or
        tolerance-based matches in the synthetic data.
    Weaknesses
        Results depend on the selected QIs and numeric tolerances; aggregate
        rates can hide subgroup-specific matches.
    Citation
        NIST SDNist benchmark and evaluation tools,
        https://github.com/usnistgov/SDNist; see also Raab, Nowok, and Dibben
        (2024), Appendix 1, https://arxiv.org/abs/2406.16826.
    """
    qsets = qi_sets(schema, profile)
    if bundle.real_test is None:
        return [
            filter_result(
                not_assessed(
                    "exact_match_rate",
                    "no holdout supplied for the required control baseline",
                    group=qi_set.name,
                ),
                profile,
            )
            for qi_set in qsets
        ]
    if bundle.real_train.empty or bundle.synthetic.empty or bundle.real_test.empty:
        return [
            filter_result(
                not_assessed(
                    "exact_match_rate",
                    "real_train, synthetic, and real_test must all contain records",
                    group=qi_set.name,
                ),
                profile,
            )
            for qi_set in qsets
        ]

    results: list[MetricResult] = []
    for qi_set in qsets:
        matched = matching_rows(
            bundle.real_train, bundle.synthetic, qi_set.columns, schema
        )
        baseline_matches = matching_rows(
            bundle.real_train, bundle.real_test, qi_set.columns, schema
        )
        n_flagged = int(matched.sum())
        n_baseline_matches = int(baseline_matches.sum())
        result = assessed_result(
            "exact_match_rate",
            n_flagged / len(bundle.real_train),
            n_baseline_matches / len(bundle.real_train),
            wilson_interval(n_flagged, len(bundle.real_train)),
            len(bundle.real_train),
            n_flagged,
            group=qi_set.name,
        )
        results.append(
            filter_result(result, profile, contributing_counts=[n_baseline_matches])
        )
    return results


def unique_exact_match_rate(
    bundle: DatasetBundle, schema: Schema, profile: PolicyProfile
) -> list[MetricResult]:
    """Calculate the unique exact-match rate with a holdout control.

    Definition
        Among rows unique on their exact real-training QI key, the share that
        matches at least one synthetic row; numeric match tolerances follow the
        schema.
    Inputs
        `real_train`, `synthetic`, and `real_test`.
    Baseline
        Equivalent match share for QI-unique holdout rows against the synthetic
        data.
    Interpretation
        A higher rate than the holdout control indicates that training-unique
        records are more often reproduced than comparable holdout records.
    Weaknesses
        Exact uniqueness is affected by QI selection and data quality. Numeric
        tolerance matching is not used to define equivalence-class uniqueness.
    Citation
        NIST SDNist benchmark and evaluation tools,
        https://github.com/usnistgov/SDNist.
    """
    qsets = qi_sets(schema, profile)
    if bundle.real_test is None:
        return [
            filter_result(
                not_assessed(
                    "uem_rate",
                    "no holdout supplied for the required control baseline",
                    group=qi_set.name,
                ),
                profile,
            )
            for qi_set in qsets
        ]
    if bundle.real_train.empty or bundle.synthetic.empty or bundle.real_test.empty:
        return [
            filter_result(
                not_assessed(
                    "uem_rate",
                    "real_train, synthetic, and real_test must all contain records",
                    group=qi_set.name,
                ),
                profile,
            )
            for qi_set in qsets
        ]

    results: list[MetricResult] = []
    for qi_set in qsets:
        train_keys = row_keys(bundle.real_train, qi_set.columns)
        test_keys = row_keys(bundle.real_test, qi_set.columns)
        train_counts = key_counts(bundle.real_train, qi_set.columns)
        test_counts = key_counts(bundle.real_test, qi_set.columns)
        unique_train_positions = [
            position for position, key in enumerate(train_keys) if train_counts[key] == 1
        ]
        unique_test_positions = [
            position for position, key in enumerate(test_keys) if test_counts[key] == 1
        ]
        if not unique_train_positions or not unique_test_positions:
            results.append(
                filter_result(
                    not_assessed(
                        "uem_rate",
                        "no unique training or holdout records for this QI set",
                        group=qi_set.name,
                    ),
                    profile,
                )
            )
            continue

        unique_train = bundle.real_train.iloc[unique_train_positions]
        unique_test = bundle.real_test.iloc[unique_test_positions]
        train_matches = matching_rows(
            unique_train, bundle.synthetic, qi_set.columns, schema
        )
        test_matches = matching_rows(
            unique_test, bundle.synthetic, qi_set.columns, schema
        )
        n_flagged = int(train_matches.sum())
        n_baseline_matches = int(test_matches.sum())
        result = assessed_result(
            "uem_rate",
            n_flagged / len(unique_train),
            n_baseline_matches / len(unique_test),
            wilson_interval(n_flagged, len(unique_train)),
            len(unique_train),
            n_flagged,
            group=qi_set.name,
        )
        results.append(
            filter_result(
                result,
                profile,
                contributing_counts=[len(unique_test), n_baseline_matches],
            )
        )
    return results
