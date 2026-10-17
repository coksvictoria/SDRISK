"""Tests for policy-based recommendations and confidence."""

from sdrisk.decision import decide_results
from sdrisk.policy import PolicyProfile
from sdrisk.results import MetricResult


def make_profile(*, release_tier: str = "in_lab") -> PolicyProfile:
    return PolicyProfile.model_validate(
        {
            "name": "decision-test",
            "release_tier": release_tier,
            "output": {"min_cell_size": 2, "rounding_base": 1},
            "variables": {
                "qi_sets": [{"name": "core", "columns": ["key"]}],
                "sensitive": [],
                "strata": [],
            },
            "identity": {"k_profile_thresholds": [2, 5]},
            "distance": {
                "max_rows_exact": 100,
                "subsample_rows": 100,
                "near_match_threshold": 0.05,
            },
            "clusters": {"k": 2, "min_cluster_size": 2, "seeds": 2},
            "thresholds": {
                "exact_match_rate": {"review": 0.01, "fail": 0.05},
                "mia_auc": {"review": 0.55, "fail": 0.65},
            },
            "policy": {"require_test_for_public_tier": True},
        }
    )


def metric(
    name: str,
    value: float | None,
    *,
    status: str = "review",
    reason: str = "",
) -> MetricResult:
    return MetricResult(
        name=name,
        level="table",
        value=value,
        baseline=0.0 if value is not None else None,
        ci=None,
        n_records=20 if value is not None else None,
        n_flagged=None,
        status=status,  # type: ignore[arg-type]
        reason=reason,
    )


def test_fail_maps_to_refuse_and_uses_configured_cutoffs() -> None:
    decision = decide_results(
        [metric("exact_match_rate", 0.06)],
        make_profile(),
        has_holdout=True,
    )

    assert decision.results[0].status == "fail"
    assert decision.recommendation == "Refuse"


def test_review_threshold_maps_to_release_with_modifications() -> None:
    decision = decide_results(
        [metric("exact_match_rate", 0.02)],
        make_profile(),
        has_holdout=True,
    )

    assert decision.results[0].status == "review"
    assert decision.recommendation == "Release with modifications"


def test_all_pass_maps_to_release_when_public_holdout_is_present() -> None:
    decision = decide_results(
        [metric("exact_match_rate", 0.001)],
        make_profile(release_tier="public"),
        has_holdout=True,
    )

    assert decision.results[0].status == "pass"
    assert decision.recommendation == "Release"
    assert decision.confidence == "high"


def test_not_assessed_result_caps_recommendation_and_lowers_confidence() -> None:
    decision = decide_results(
        [
            metric(
                "mia_distance_auc",
                None,
                status="not_assessed",
                reason="no holdout supplied",
            )
        ],
        make_profile(release_tier="public"),
        has_holdout=False,
    )

    assert decision.results[0].status == "not_assessed"
    assert decision.recommendation == "Release with modifications"
    assert decision.confidence == "low"


def test_missing_threshold_is_explicit_review_not_an_implicit_pass() -> None:
    decision = decide_results(
        [metric("unconfigured_metric", 0.0)],
        make_profile(),
        has_holdout=True,
    )

    assert decision.results[0].status == "review"
    assert "no threshold" in decision.results[0].reason.lower()
    assert decision.recommendation == "Release with modifications"


def test_membership_auc_uses_the_profile_mia_threshold() -> None:
    decision = decide_results(
        [metric("mia_distance_auc", 0.7)],
        make_profile(),
        has_holdout=True,
    )

    assert decision.results[0].status == "fail"


def test_public_holdout_requirement_caps_an_otherwise_passing_run() -> None:
    decision = decide_results(
        [metric("exact_match_rate", 0.001)],
        make_profile(release_tier="public"),
        has_holdout=False,
    )

    assert decision.recommendation == "Release with modifications"
    assert any("holdout" in reason.lower() for reason in decision.rationale)
