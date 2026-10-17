"""Shared helpers for table-level attribute-disclosure metrics."""

from collections import Counter
from collections.abc import Sequence

from sdrisk.bundle import DatasetBundle
from sdrisk.metrics.identity.common import (
    filter_result,
    not_assessed,
    qi_sets,
    sensitive_variables,
    validate_bundle_columns,
)
from sdrisk.policy import PolicyProfile, QISet
from sdrisk.results import MetricResult
from sdrisk.schema import Schema, Variable


def missing_sensitive_result(name: str, profile: PolicyProfile) -> list[MetricResult]:
    """Return an explicit result when a profile has no sensitive targets."""
    return [
        filter_result(
            not_assessed(name, "no sensitive variables configured"),
            profile,
        )
    ]


def validate_metric_inputs(
    bundle: DatasetBundle,
    schema: Schema,
    profile: PolicyProfile,
    qi_set: QISet,
) -> tuple[Variable, ...]:
    """Validate selected columns and return configured sensitive targets."""
    selected = (*qi_set.columns, *profile.variables.sensitive)
    validate_bundle_columns(bundle, selected)
    return sensitive_variables(schema, profile, qi_set)


def result_group(qi_set: QISet, target: Variable) -> str:
    """Build the stable aggregate label for a QI-set/target pair."""
    return f"{qi_set.name}:{target.name}"


def marginal_guess_accuracy(
    counts: Counter[tuple[object, ...]],
    labels: Sequence[tuple[object, ...]],
) -> float:
    """Return expected accuracy of an independent guess from an empirical marginal."""
    total = sum(counts.values())
    if total == 0:
        raise ValueError("cannot calculate a marginal guess from no target values")
    return sum(counts[label] / total for label in labels) / len(labels)


def assessed_or_not(
    name: str,
    reason: str,
    profile: PolicyProfile,
    *,
    group: str | None = None,
) -> MetricResult:
    """Build and filter a not-assessed metric result."""
    return filter_result(not_assessed(name, reason, group=group), profile)


def get_qi_sets(schema: Schema, profile: PolicyProfile) -> tuple[QISet, ...]:
    """Expose the common configured-QI validation to attribute metrics."""
    return qi_sets(schema, profile)
