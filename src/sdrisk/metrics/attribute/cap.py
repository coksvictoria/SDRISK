"""CAP and TCAP-style correct-attribution measures."""

from collections import Counter

from sdrisk.bundle import DatasetBundle
from sdrisk.metrics.attribute.common import (
    assessed_or_not,
    get_qi_sets,
    marginal_guess_accuracy,
    missing_sensitive_result,
    result_group,
    validate_metric_inputs,
)
from sdrisk.metrics.identity.common import (
    assessed_result,
    contributing_cell_counts,
    filter_result,
    grouped_target_counts,
    row_keys,
    target_keys,
    wilson_interval,
)
from sdrisk.policy import PolicyProfile
from sdrisk.results import MetricResult
from sdrisk.schema import Schema


def cap_disclosure(
    bundle: DatasetBundle, schema: Schema, profile: PolicyProfile
) -> list[MetricResult]:
    """Calculate a DCAP-d-style expected correct-attribution rate.

    Definition
        For each original row, use its exact QI class to obtain the synthetic
        conditional distribution of the target, then average the probability
        assigned to that row's true target. QI classes absent from synthetic
        data contribute zero; the denominator is all original rows.
    Inputs
        `real_train` and `synthetic`, one categorical or ordinal target and
        QI set at a time.
    Baseline
        Expected accuracy of an independent target guess sampled from the
        `real_train` target's empirical marginal distribution.
    Interpretation
        Values above baseline indicate that QI-linked synthetic targets give
        better expected attribution than a marginal-only random guess.
    Weaknesses
        Exact classes and target categories can be sparse; results depend on
        the selected QIs and do not measure membership disclosure.
    Citation
        Raab, Nowok, and Dibben (2024), Section 3.3.2, discussion of CAP and
        DCAP denominator variants, https://arxiv.org/abs/2406.16826.
    """
    if not profile.variables.sensitive:
        return missing_sensitive_result("cap_accuracy", profile)
    if bundle.real_train.empty or bundle.synthetic.empty:
        return [
            assessed_or_not(
                "cap_accuracy",
                "real_train and synthetic must both contain records",
                profile,
            )
        ]

    results: list[MetricResult] = []
    for qi_set in get_qi_sets(schema, profile):
        targets = validate_metric_inputs(bundle, schema, profile, qi_set)
        for target in targets:
            group = result_group(qi_set, target)
            if target.vtype not in ("categorical", "ordinal"):
                results.append(
                    assessed_or_not(
                        "cap_accuracy",
                        "CAP requires a categorical or ordinal target; "
                        "use numeric_tolerance for numeric targets",
                        profile,
                        group=group,
                    )
                )
                continue

            original_counts = grouped_target_counts(
                bundle.real_train, qi_set.columns, target.name
            )
            synthetic_counts = grouped_target_counts(
                bundle.synthetic, qi_set.columns, target.name
            )
            original_labels = target_keys(bundle.real_train, target.name)
            marginal_counts = Counter(original_labels)
            n_original = len(original_labels)
            if n_original == 0:
                results.append(
                    assessed_or_not(
                        "cap_accuracy",
                        "real_train has no target values",
                        profile,
                        group=group,
                    )
                )
                continue

            correct_probability = 0.0
            for qi_key, label in zip(
                row_keys(bundle.real_train, qi_set.columns),
                original_labels,
                strict=True,
            ):
                conditional = synthetic_counts.get(qi_key)
                if conditional:
                    correct_probability += conditional[label] / sum(conditional.values())
            value = correct_probability / n_original
            baseline = marginal_guess_accuracy(marginal_counts, original_labels)
            result = assessed_result(
                "cap_accuracy",
                value,
                baseline,
                None,
                n_original,
                None,
                group=group,
            )
            results.append(
                filter_result(
                    result,
                    profile,
                    contributing_counts=contributing_cell_counts(
                        original_counts,
                        synthetic_counts,
                        tuple(original_counts),
                    ),
                )
            )
    return results


def tcap_disclosure(
    bundle: DatasetBundle, schema: Schema, profile: PolicyProfile
) -> list[MetricResult]:
    """Calculate conditional correct attribution for deterministic synthetic QIs.

    Definition
        Among original rows whose QI classes occur in synthetic data and have
        exactly one synthetic target value, report the share whose target
        matches that deterministic synthetic prediction.
    Inputs
        `real_train` and `synthetic`, one categorical or ordinal target and
        QI set at a time.
    Baseline
        Expected accuracy of an independent guess from the original target's
        empirical marginal, evaluated on the same deterministic-class rows.
    Interpretation
        This is conditional accuracy only; unlike DiSCO, it is not divided by
        the full original-data size.
    Weaknesses
        The conditional denominator can make values appear large when few
        original rows have deterministic synthetic predictions.
    Citation
        Raab, Nowok, and Dibben (2024), Section 3.3.2, TCAP denominator
        discussion, https://arxiv.org/abs/2406.16826.
    """
    if not profile.variables.sensitive:
        return missing_sensitive_result("tcap_accuracy", profile)
    if bundle.real_train.empty or bundle.synthetic.empty:
        return [
            assessed_or_not(
                "tcap_accuracy",
                "real_train and synthetic must both contain records",
                profile,
            )
        ]

    results: list[MetricResult] = []
    for qi_set in get_qi_sets(schema, profile):
        targets = validate_metric_inputs(bundle, schema, profile, qi_set)
        for target in targets:
            group = result_group(qi_set, target)
            if target.vtype not in ("categorical", "ordinal"):
                results.append(
                    assessed_or_not(
                        "tcap_accuracy",
                        "TCAP requires a categorical or ordinal target",
                        profile,
                        group=group,
                    )
                )
                continue

            original_counts = grouped_target_counts(
                bundle.real_train, qi_set.columns, target.name
            )
            synthetic_counts = grouped_target_counts(
                bundle.synthetic, qi_set.columns, target.name
            )
            deterministic = {
                qi_key: next(iter(cells))
                for qi_key, cells in synthetic_counts.items()
                if len(cells) == 1
            }
            eligible = [
                (qi_key, label)
                for qi_key, label in zip(
                    row_keys(bundle.real_train, qi_set.columns),
                    target_keys(bundle.real_train, target.name),
                    strict=True,
                )
                if qi_key in deterministic
            ]
            if not eligible:
                results.append(
                    assessed_or_not(
                        "tcap_accuracy",
                        "no original rows belong to a deterministic synthetic QI class",
                        profile,
                        group=group,
                    )
                )
                continue

            original_labels = target_keys(bundle.real_train, target.name)
            marginal_counts = Counter(original_labels)
            n_correct = sum(
                label == deterministic[qi_key] for qi_key, label in eligible
            )
            baseline = marginal_guess_accuracy(
                marginal_counts, [label for _, label in eligible]
            )
            result = assessed_result(
                "tcap_accuracy",
                n_correct / len(eligible),
                baseline,
                wilson_interval(n_correct, len(eligible)),
                len(eligible),
                n_correct,
                group=group,
            )
            results.append(
                filter_result(
                    result,
                    profile,
                    contributing_counts=[
                        *contributing_cell_counts(
                            original_counts,
                            {},
                            tuple(original_counts),
                        ),
                        *contributing_cell_counts(
                            {},
                            synthetic_counts,
                            tuple(deterministic),
                        ),
                    ],
                )
            )
    return results
