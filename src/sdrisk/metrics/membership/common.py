"""Shared scoring and bootstrap helpers for membership metrics."""

import numpy as np
from sklearn.metrics import auc, roc_curve

from sdrisk.metrics.identity.common import (
    assessed_result,
    filter_result,
    not_assessed,
)
from sdrisk.policy import PolicyProfile
from sdrisk.results import MetricResult

_BOOTSTRAP_REPLICATES = 200
_FALSE_POSITIVE_RATES = (0.01, 0.001)
_METRIC_SUFFIXES = (
    "auc",
    "advantage",
    "tpr_at_1pct_fpr",
    "tpr_at_0_1pct_fpr",
)
_BASELINES = (0.5, 0.0, 0.01, 0.001)


def membership_not_assessed(
    name_prefix: str,
    reason: str,
    group: str,
    profile: PolicyProfile,
) -> list[MetricResult]:
    """Return an explicit result for every unassessable membership statistic."""
    return [
        filter_result(
            not_assessed(f"{name_prefix}_{suffix}", reason, group=group),
            profile,
        )
        for suffix in _METRIC_SUFFIXES
    ]


def membership_results(
    member_scores: np.ndarray,
    nonmember_scores: np.ndarray,
    *,
    name_prefix: str,
    group: str,
    profile: PolicyProfile,
    rng: np.random.Generator,
    experimental: bool = False,
    reason: str = "",
) -> list[MetricResult]:
    """Summarize membership scores with a stratified bootstrap."""
    if not isinstance(rng, np.random.Generator):
        raise TypeError("rng must be a numpy.random.Generator")
    members = np.asarray(member_scores, dtype=np.float64)
    nonmembers = np.asarray(nonmember_scores, dtype=np.float64)
    members = members[np.isfinite(members)]
    nonmembers = nonmembers[np.isfinite(nonmembers)]
    if members.size == 0 or nonmembers.size == 0:
        raise ValueError("membership scores must include both finite classes")

    observed = _roc_metrics(members, nonmembers)
    bootstrap = np.empty((_BOOTSTRAP_REPLICATES, len(_METRIC_SUFFIXES)))
    for iteration in range(_BOOTSTRAP_REPLICATES):
        sampled_members = members[rng.integers(0, members.size, size=members.size)]
        sampled_nonmembers = nonmembers[rng.integers(0, nonmembers.size, size=nonmembers.size)]
        bootstrap[iteration] = _roc_metrics(sampled_members, sampled_nonmembers)
    intervals = np.quantile(bootstrap, (0.025, 0.975), axis=0)
    n_records = int(members.size + nonmembers.size)
    contributing_counts = (int(members.size), int(nonmembers.size))

    results: list[MetricResult] = []
    for index, suffix in enumerate(_METRIC_SUFFIXES):
        result = assessed_result(
            f"{name_prefix}_{suffix}",
            observed[index],
            _BASELINES[index],
            (float(intervals[0, index]), float(intervals[1, index])),
            n_records,
            None,
            group=group,
            experimental=experimental,
            reason=reason or "Decision classification is deferred to the M7 decision layer.",
        )
        results.append(
            filter_result(
                result,
                profile,
                contributing_counts=contributing_counts,
            )
        )
    return results


def _roc_metrics(
    member_scores: np.ndarray, nonmember_scores: np.ndarray
) -> tuple[float, float, float, float]:
    """Return AUC, advantage, and TPR at the two configured FPR operating points."""
    scores = np.concatenate((member_scores, nonmember_scores))
    labels = np.concatenate(
        (
            np.ones(member_scores.size, dtype=np.int8),
            np.zeros(nonmember_scores.size, dtype=np.int8),
        )
    )
    false_positive_rate, true_positive_rate, _ = roc_curve(labels, scores, drop_intermediate=False)
    area = float(auc(false_positive_rate, true_positive_rate))
    advantage = float(np.max(true_positive_rate - false_positive_rate))
    low_fpr_tprs = tuple(
        float(np.max(true_positive_rate[false_positive_rate <= limit]))
        for limit in _FALSE_POSITIVE_RATES
    )
    return area, advantage, low_fpr_tprs[0], low_fpr_tprs[1]
