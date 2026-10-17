"""Metric applicability checks based on dataset bundle inputs."""

from collections.abc import Collection

from sdrisk.bundle import DatasetBundle
from sdrisk.results import Level, MetricResult

INPUT_NAMES = frozenset({"real_train", "synthetic", "real_test"})


def check_applicability(
    metric_name: str,
    requires: Collection[str],
    bundle: DatasetBundle,
    *,
    level: Level = "table",
) -> MetricResult | None:
    """Return a not-assessed result when a metric's required input is absent."""
    unknown_inputs = set(requires) - INPUT_NAMES
    if unknown_inputs:
        raise ValueError(f"unknown required inputs: {', '.join(sorted(unknown_inputs))}")
    missing = sorted(
        name
        for name in requires
        if getattr(bundle, name) is None
    )
    if not missing:
        return None
    return MetricResult(
        name=metric_name,
        level=level,
        value=None,
        baseline=None,
        ci=None,
        n_records=None,
        n_flagged=None,
        status="not_assessed",
        reason=f"missing required input: {', '.join(missing)}",
    )
