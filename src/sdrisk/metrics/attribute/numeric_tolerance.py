"""Numeric target recovery within schema-declared tolerance."""

import math

import numpy as np

from sdrisk.bundle import DatasetBundle
from sdrisk.metrics.attribute.common import (
    assessed_or_not,
    get_qi_sets,
    missing_sensitive_result,
    result_group,
    validate_metric_inputs,
)
from sdrisk.metrics.identity.common import (
    assessed_result,
    filter_result,
    grouped_target_counts,
    row_keys,
    wilson_interval,
)
from sdrisk.policy import PolicyProfile
from sdrisk.prep.values import numeric_values
from sdrisk.results import MetricResult
from sdrisk.schema import Schema


def numeric_tolerance_recovery(
    bundle: DatasetBundle, schema: Schema, profile: PolicyProfile
) -> list[MetricResult]:
    """Measure recovery of numeric sensitive values within configured tolerance.

    Definition
        A real numeric target is recovered when at least one synthetic target
        in the exact-matching QI class lies within the schema tolerance. Unmatched
        QI classes count as unrecovered.
    Inputs
        `real_train` and `synthetic`, a numeric sensitive variable with a
        schema-declared tolerance, and each configured QI set.
    Baseline
        Expected recovery probability if a target value is guessed independently
        from the empirical marginal distribution of `real_train`.
    Interpretation
        A higher recovery share than baseline means QI-linked synthetic values
        provide more within-tolerance guesses than marginal-only sampling.
    Weaknesses
        The result depends on policy-selected tolerance and exact QI matching;
        it does not estimate membership disclosure.
    Citation
        Project-defined numeric-tolerance disclosure check, AGENTS.md §7.2;
        tolerance must be supplied in the variable schema.
    """
    if not profile.variables.sensitive:
        return missing_sensitive_result("numeric_tolerance_recovery", profile)
    if bundle.real_train.empty or bundle.synthetic.empty:
        return [
            assessed_or_not(
                "numeric_tolerance_recovery",
                "real_train and synthetic must both contain records",
                profile,
            )
        ]

    results: list[MetricResult] = []
    for qi_set in get_qi_sets(schema, profile):
        targets = validate_metric_inputs(bundle, schema, profile, qi_set)
        for target in targets:
            group = result_group(qi_set, target)
            if target.vtype != "numeric":
                results.append(
                    assessed_or_not(
                        "numeric_tolerance_recovery",
                        "numeric tolerance recovery only applies to numeric targets",
                        profile,
                        group=group,
                    )
                )
                continue
            if target.tolerance is None:
                results.append(
                    assessed_or_not(
                        "numeric_tolerance_recovery",
                        "numeric target has no schema-declared tolerance",
                        profile,
                        group=group,
                    )
                )
                continue

            real_values = numeric_values(bundle.real_train[target.name], target)
            synthetic_values = numeric_values(bundle.synthetic[target.name], target)
            finite_original = np.isfinite(real_values)
            n_original = int(finite_original.sum())
            if n_original == 0:
                results.append(
                    assessed_or_not(
                        "numeric_tolerance_recovery",
                        "real_train has no finite numeric target values",
                        profile,
                        group=group,
                    )
                )
                continue
            if not np.isfinite(synthetic_values).any():
                results.append(
                    assessed_or_not(
                        "numeric_tolerance_recovery",
                        "synthetic has no finite numeric target values",
                        profile,
                        group=group,
                    )
                )
                continue

            real_qi = row_keys(bundle.real_train, qi_set.columns)
            synthetic_qi = row_keys(bundle.synthetic, qi_set.columns)
            synthetic_by_qi: dict[tuple[object, ...], list[float]] = {}
            for qi_key, value in zip(synthetic_qi, synthetic_values, strict=True):
                if math.isfinite(float(value)):
                    synthetic_by_qi.setdefault(qi_key, []).append(float(value))

            finite_positions = np.flatnonzero(finite_original)
            recovered = 0
            marginal_probabilities: list[float] = []
            tolerance = target.tolerance
            for position in finite_positions:
                target_value = real_values[position]
                candidates = synthetic_by_qi.get(real_qi[position], [])
                if candidates and np.any(
                    np.abs(np.asarray(candidates) - target_value) <= tolerance
                ):
                    recovered += 1
                marginal_probabilities.append(
                    float(np.mean(np.abs(real_values[finite_original] - target_value) <= tolerance))
                )
            baseline = float(np.mean(marginal_probabilities))
            synthetic_groups = grouped_target_counts(
                bundle.synthetic, qi_set.columns, target.name
            )
            original_groups = grouped_target_counts(
                bundle.real_train, qi_set.columns, target.name
            )
            used_groups = tuple(original_groups)
            q_class_counts = [
                sum(group.values())
                for mapping in (original_groups, synthetic_groups)
                for key, group in mapping.items()
                if key in used_groups
            ]
            result = assessed_result(
                "numeric_tolerance_recovery",
                recovered / n_original,
                baseline,
                wilson_interval(recovered, n_original),
                n_original,
                recovered,
                group=group,
            )
            results.append(
                filter_result(
                    result,
                    profile,
                    contributing_counts=q_class_counts,
                )
            )
    return results
