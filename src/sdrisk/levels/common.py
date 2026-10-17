"""Shared helpers for lifting table metrics to analytic groups."""

from dataclasses import replace
from typing import Literal

import numpy as np

from sdrisk.bundle import DatasetBundle
from sdrisk.metrics.attribute import (
    cap_disclosure,
    disco,
    homogeneity,
    ml_inference_gain,
    numeric_tolerance_recovery,
    tcap_disclosure,
)
from sdrisk.metrics.identity import (
    dcr_overfitting,
    dcr_vs_random,
    exact_match_rate,
    k_profile,
    replicated_uniques,
    unique_exact_match_rate,
)
from sdrisk.metrics.identity.common import filter_result
from sdrisk.policy import PolicyProfile
from sdrisk.results import MetricResult
from sdrisk.schema import Schema

AnalysisLevel = Literal["subset", "cluster"]


def tier1_results(
    bundle: DatasetBundle,
    schema: Schema,
    profile: PolicyProfile,
    rng: np.random.Generator,
) -> list[MetricResult]:
    """Compute the implemented table-level Tier 1 metrics for one group."""
    results = [
        *exact_match_rate(bundle, schema, profile),
        *unique_exact_match_rate(bundle, schema, profile),
        *replicated_uniques(bundle, schema, profile),
        *k_profile(bundle, schema, profile),
        *dcr_overfitting(bundle, schema, profile),
        *dcr_vs_random(bundle, schema, profile, rng),
        *cap_disclosure(bundle, schema, profile),
        *tcap_disclosure(bundle, schema, profile),
        *disco(bundle, schema, profile),
        *ml_inference_gain(bundle, schema, profile, rng),
        *numeric_tolerance_recovery(bundle, schema, profile),
        *homogeneity(bundle, schema, profile),
    ]
    return results


def lift_results(
    results: list[MetricResult],
    profile: PolicyProfile,
    *,
    level: AnalysisLevel,
    group: str,
    contributing_counts: tuple[int, ...],
) -> list[MetricResult]:
    """Set a result's analytic level and filter all group-size contributors."""
    lifted_results: list[MetricResult] = []
    for result in results:
        label = (
            f"{group}:{result.group}"
            if result.group and result.status not in ("suppressed", "not_assessed")
            else group
            if result.status not in ("suppressed", "not_assessed")
            else None
        )
        lifted = replace(result, level=level, group=label)
        lifted_results.append(
            filter_result(
                lifted,
                profile,
                contributing_counts=contributing_counts,
            )
        )
    return lifted_results
