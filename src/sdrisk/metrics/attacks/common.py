"""Shared support for holdout-controlled attribute-guess attacks."""

from collections import defaultdict
from collections.abc import Sequence

from sdrisk.bundle import DatasetBundle
from sdrisk.metrics.identity.common import assessed_result, filter_result, not_assessed
from sdrisk.policy import PolicyProfile
from sdrisk.results import MetricResult


def not_assessed_attack(name: str, reason: str, group: str, profile: PolicyProfile) -> MetricResult:
    """Create a disclosure-filtered not-assessed attack result."""
    return filter_result(
        not_assessed(name, reason, group=group),
        profile,
    )


def success_result(
    name: str,
    group: str,
    bundle: DatasetBundle,
    profile: PolicyProfile,
    *,
    successes: int,
    control_successes: int,
    reason: str,
) -> MetricResult:
    """Report the synthetic-guess success rate against training and holdout."""
    result = assessed_result(
        name,
        successes / len(bundle.synthetic),
        control_successes / len(bundle.synthetic),
        _wilson_interval(successes, len(bundle.synthetic)),
        len(bundle.synthetic),
        successes,
        group=group,
        experimental=True,
        reason=reason,
    )
    if bundle.real_test is None:
        raise RuntimeError("attack control result requires real_test")
    return filter_result(
        result,
        profile,
        contributing_counts=(len(bundle.real_test), control_successes),
    )


def positions_by_key(keys: Sequence[tuple[object, ...]]) -> dict[tuple[object, ...], list[int]]:
    """Index row positions for normalized exact keys."""
    positions: dict[tuple[object, ...], list[int]] = defaultdict(list)
    for position, key in enumerate(keys):
        positions[key].append(position)
    return dict(positions)


def _wilson_interval(successes: int, n_records: int) -> tuple[float, float]:
    """Return a 95% Wilson interval for an attack success proportion."""
    from sdrisk.metrics.identity.common import wilson_interval

    interval = wilson_interval(successes, n_records)
    if interval is None:
        raise ValueError("Wilson interval requires at least one attack guess")
    return interval
