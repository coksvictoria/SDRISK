"""Sensitive-value homogeneity in quasi-identifier equivalence classes."""

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
    contributing_cell_counts,
    filter_result,
    grouped_target_counts,
    wilson_interval,
)
from sdrisk.policy import PolicyProfile
from sdrisk.results import MetricResult
from sdrisk.schema import Schema


def homogeneity(
    bundle: DatasetBundle, schema: Schema, profile: PolicyProfile
) -> list[MetricResult]:
    """Compare homogeneous synthetic QI classes with their original-data baseline.

    Definition
        Share of QI classes in synthetic data whose size is at least the
        configured minimum cell size and whose records all have one exact
        sensitive value. The baseline is the equivalent share in `real_train`.
    Inputs
        Non-empty `real_train` and `synthetic` tables, each configured sensitive
        variable, and each configured QI set.
    Baseline
        Homogeneous eligible-class share in original training data.
    Interpretation
        A larger synthetic share indicates more synthetic QI classes with a
        single sensitive value than in the original data.
    Weaknesses
        Exact equality treats continuous values as distinct and class-level
        shares weight small and large eligible classes equally.
    Citation
        Project-defined class-level summary related to l-diversity and
        DiSCO; Raab, Nowok, and Dibben (2024), Appendix 1, equations (5)-(6),
        https://arxiv.org/abs/2406.16826.
    """
    if not profile.variables.sensitive:
        return missing_sensitive_result("homogeneity_share", profile)
    if bundle.real_train.empty or bundle.synthetic.empty:
        return [
            assessed_or_not(
                "homogeneity_share",
                "real_train and synthetic must both contain records",
                profile,
            )
        ]

    minimum_size = profile.output.min_cell_size
    results: list[MetricResult] = []
    for qi_set in get_qi_sets(schema, profile):
        targets = validate_metric_inputs(bundle, schema, profile, qi_set)
        for target in targets:
            original_counts = grouped_target_counts(
                bundle.real_train, qi_set.columns, target.name
            )
            synthetic_counts = grouped_target_counts(
                bundle.synthetic, qi_set.columns, target.name
            )
            eligible_original = {
                key: cells
                for key, cells in original_counts.items()
                if sum(cells.values()) >= minimum_size
            }
            eligible_synthetic = {
                key: cells
                for key, cells in synthetic_counts.items()
                if sum(cells.values()) >= minimum_size
            }
            group = result_group(qi_set, target)
            if not eligible_original or not eligible_synthetic:
                results.append(
                    assessed_or_not(
                        "homogeneity_share",
                        "no real and synthetic QI classes meet the configured minimum size",
                        profile,
                        group=group,
                    )
                )
                continue

            n_homogeneous_synthetic = sum(
                len(cells) == 1 for cells in eligible_synthetic.values()
            )
            n_homogeneous_original = sum(
                len(cells) == 1 for cells in eligible_original.values()
            )
            n_synthetic_classes = len(eligible_synthetic)
            result = assessed_result(
                "homogeneity_share",
                n_homogeneous_synthetic / n_synthetic_classes,
                n_homogeneous_original / len(eligible_original),
                wilson_interval(n_homogeneous_synthetic, n_synthetic_classes),
                n_synthetic_classes,
                n_homogeneous_synthetic,
                group=group,
            )
            relevant_groups = tuple(
                set(eligible_original) | set(eligible_synthetic)
            )
            results.append(
                filter_result(
                    result,
                    profile,
                    contributing_counts=[
                        *contributing_cell_counts(
                            original_counts,
                            synthetic_counts,
                            relevant_groups,
                        ),
                        len(eligible_original),
                    ],
                )
            )
    return results
