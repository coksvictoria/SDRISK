"""Replicated-unique identity disclosure metric."""

from sdrisk.bundle import DatasetBundle
from sdrisk.metrics.identity.common import (
    assessed_result,
    filter_result,
    key_counts,
    not_assessed,
    qi_sets,
    row_keys,
    wilson_interval,
)
from sdrisk.policy import PolicyProfile
from sdrisk.results import MetricResult
from sdrisk.schema import Schema


def replicated_uniques(
    bundle: DatasetBundle, schema: Schema, profile: PolicyProfile
) -> list[MetricResult]:
    """Calculate replicated uniques (RepU) with UiO as its baseline.

    Definition
        Percentage of original records that are unique on the QIs in both the
        original and synthetic data: 100 * count(d.q == 1 and s.q == 1) / N_d.
        The accompanying baseline, UiO, is 100 * count(d.q == 1) / N_d.
    Inputs
        `real_train` and `synthetic`, evaluated separately for each configured
        QI set.
    Baseline
        Percentage unique in the original data (UiO).
    Interpretation
        RepU near UiO indicates that many original unique QI combinations also
        remain unique in synthetic output. RepU is not an estimate of all
        disclosure risk.
    Weaknesses
        It depends strongly on QI choice and does not account for attribute or
        membership disclosure.
    Citation
        Raab, Nowok, and Dibben (2024), *Practical privacy metrics for synthetic
        data*, Appendix 1, equations (1) and (4),
        https://arxiv.org/abs/2406.16826.
    """
    qsets = qi_sets(schema, profile)
    if bundle.real_train.empty or bundle.synthetic.empty:
        return [
            filter_result(
                not_assessed(
                    "repu_pct",
                    "real_train and synthetic must both contain records",
                    group=qi_set.name,
                ),
                profile,
            )
            for qi_set in qsets
        ]

    results: list[MetricResult] = []
    for qi_set in qsets:
        train_keys = row_keys(bundle.real_train, qi_set.columns)
        train_counts = key_counts(bundle.real_train, qi_set.columns)
        synthetic_counts = key_counts(bundle.synthetic, qi_set.columns)
        unique_original = sum(train_counts[key] == 1 for key in train_keys)
        replicated_unique = sum(
            train_counts[key] == 1 and synthetic_counts[key] == 1 for key in train_keys
        )
        n_original = len(bundle.real_train)
        value = 100.0 * replicated_unique / n_original
        baseline = 100.0 * unique_original / n_original
        ci = wilson_interval(replicated_unique, n_original, scale=100.0)
        result = assessed_result(
            "repu_pct",
            value,
            baseline,
            ci,
            n_original,
            replicated_unique,
            group=qi_set.name,
        )
        results.append(
            filter_result(result, profile, contributing_counts=[unique_original])
        )
    return results
