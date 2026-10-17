"""Policy-based metric adjudication and overall recommendations."""

from dataclasses import dataclass, replace
from typing import Literal

from sdrisk.policy import PolicyProfile, Threshold
from sdrisk.results import MetricResult

Recommendation = Literal["Release", "Release with modifications", "Refuse"]
Confidence = Literal["high", "medium", "low"]

_BASELINE_EXCESS_METRICS = frozenset(
    {
        "cap_accuracy",
        "tcap_accuracy",
        "disco_pct",
        "ml_inference_gain",
        "numeric_tolerance_recovery",
        "homogeneity_share",
        "cluster_exposure_ratio",
        "singling_out_success",
        "linkability_success",
    }
)
_LOWER_IS_RISK = frozenset({"dcr_vs_random_ratio"})


@dataclass(frozen=True)
class AssessmentDecision:
    """Summarize classified results, recommendation, confidence, and rationale."""

    results: tuple[MetricResult, ...]
    recommendation: Recommendation
    confidence: Confidence
    rationale: tuple[str, ...]


def decide_results(
    results: list[MetricResult] | tuple[MetricResult, ...],
    profile: PolicyProfile,
    *,
    has_holdout: bool,
) -> AssessmentDecision:
    """Classify results against configured cutoffs and produce a recommendation.

    Threshold values are applied to each metric's documented risk signal:
    direct metric value for higher-is-risk metrics, baseline excess for
    disclosure-gain diagnostics, and baseline shortfall for DCR-vs-random.
    Metrics without configured cutoffs remain in review.
    """
    classified = tuple(_classify(result, profile.thresholds) for result in results)
    statuses = {result.status for result in classified}
    rationale: list[str] = []
    if "fail" in statuses:
        recommendation: Recommendation = "Refuse"
        rationale.append("At least one assessed metric reached its configured fail threshold.")
    elif "review" in statuses or "suppressed" in statuses or "not_assessed" in statuses:
        recommendation = "Release with modifications"
        rationale.append(
            "One or more results require review, are suppressed, or were not assessed."
        )
    else:
        recommendation = "Release"
        rationale.append("All assessed metrics were below their configured review thresholds.")

    if (
        profile.release_tier == "public"
        and profile.policy.require_test_for_public_tier
        and not has_holdout
    ):
        if recommendation == "Release":
            recommendation = "Release with modifications"
        rationale.append("Public-tier policy requires a real_test holdout before Release.")

    confidence = _confidence(classified, has_holdout)
    return AssessmentDecision(
        results=classified,
        recommendation=recommendation,
        confidence=confidence,
        rationale=tuple(rationale),
    )


def _classify(
    result: MetricResult,
    thresholds: dict[str, Threshold],
) -> MetricResult:
    if result.status in ("suppressed", "not_assessed"):
        return result
    threshold_aliases = {
        "disco_pct": "disco_gain",
        "mia_distance_auc": "mia_auc",
        "mia_density_auc": "mia_auc",
    }
    threshold_name = threshold_aliases.get(result.name, result.name)
    threshold = thresholds.get(threshold_name)
    if threshold is None:
        reason = _append_reason(
            result.reason, "No threshold configured; manual review is required."
        )
        return replace(result, status="review", reason=reason)
    signal = _risk_signal(result)
    if signal is None:
        reason = _append_reason(
            result.reason, "Risk signal is unavailable; manual review is required."
        )
        return replace(result, status="review", reason=reason)
    if signal >= threshold.fail:
        status: Literal["pass", "review", "fail"] = "fail"
        explanation = f"Risk signal {signal:.6g} reached the fail threshold."
    elif signal >= threshold.review:
        status = "review"
        explanation = f"Risk signal {signal:.6g} reached the review threshold."
    else:
        status = "pass"
        explanation = "Risk signal is below the configured review threshold."
    return replace(result, status=status, reason=_append_reason(result.reason, explanation))


def _risk_signal(result: MetricResult) -> float | None:
    if result.value is None:
        return None
    if result.name in _BASELINE_EXCESS_METRICS:
        if result.baseline is None:
            return None
        return max(0.0, result.value - result.baseline)
    if result.name in _LOWER_IS_RISK:
        if result.baseline is None:
            return None
        return max(0.0, result.baseline - result.value)
    return result.value


def _confidence(
    results: tuple[MetricResult, ...],
    has_holdout: bool,
) -> Confidence:
    if not results:
        return "low"
    assessed = sum(result.status in ("pass", "review", "fail") for result in results)
    coverage = assessed / len(results)
    if not has_holdout or coverage < 0.5:
        return "low"
    if coverage < 1.0 or any(
        result.name == "cluster_stability_ari"
        and result.value is not None
        and result.value < 1.0
        for result in results
    ):
        return "medium"
    return "high"


def _append_reason(current: str, added: str) -> str:
    return f"{current} {added}".strip()
