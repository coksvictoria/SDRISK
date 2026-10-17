"""Tests for aggregate-only JSON and HTML reports."""

import json
from dataclasses import replace

from sdrisk.decision import AssessmentDecision
from sdrisk.output_filter import OutputFilter
from sdrisk.policy import PolicyProfile
from sdrisk.provenance import RunProvenance
from sdrisk.report.html_report import render_html_report
from sdrisk.report.json_report import render_json_report
from sdrisk.results import MetricResult, RowStore, RunResult


def make_profile() -> PolicyProfile:
    return PolicyProfile.model_validate(
        {
            "name": "report-test",
            "release_tier": "in_lab",
            "output": {"min_cell_size": 3, "rounding_base": 1},
            "variables": {
                "qi_sets": [{"name": "core", "columns": ["key"]}],
                "sensitive": [],
                "strata": [],
            },
            "identity": {"k_profile_thresholds": [2]},
            "distance": {
                "max_rows_exact": 100,
                "subsample_rows": 100,
                "near_match_threshold": 0.05,
            },
            "clusters": {"k": 2, "min_cluster_size": 2, "seeds": 2},
            "thresholds": {"risk": {"review": 0.1, "fail": 0.5}},
            "policy": {"require_test_for_public_tier": False},
        }
    )


def make_run_result() -> RunResult:
    profile = make_profile()
    result = MetricResult(
        name="risk",
        level="table",
        value=0.3,
        baseline=0.1,
        ci=(0.2, 0.4),
        n_records=20,
        n_flagged=9,
        status="review",
    )
    row_store = RowStore()
    for band, count in (("low", 10), ("medium", 8), ("high", 2)):
        for _ in range(count):
            row_store.add_profile(
                {
                    "composite_band": band,
                    "dcr_train": 0.123456,
                    "exact_match": True,
                }
            )
    decision = AssessmentDecision(
        results=(result,),
        recommendation="Release with modifications",
        confidence="medium",
        rationale=("A metric requires review.",),
    )
    return RunResult(
        table_results=(result,),
        subset_results=(),
        cluster_results=(),
        row_store=row_store,
        row_status="assessed",
        provenance=RunProvenance(
            config_hash="config-digest",
            input_hashes={"synthetic": "file-digest"},
            package_version="0.0.0",
            seed=17,
            sampling_fractions={},
        ),
        decision=decision,
        profile=profile,
    )


def test_json_report_is_filtered_and_contains_aggregate_row_bands_only() -> None:
    payload = render_json_report(make_run_result())
    report = json.loads(payload)
    serialized = payload.lower()

    assert report["row_bands"]["total"]["value"] == 20
    assert report["row_bands"]["high"]["status"] == "suppressed"
    assert "dcr_train" not in serialized
    assert '"exact_match":' not in serialized
    assert "0.123456" not in serialized
    assert "row_profiles" not in serialized
    assert "baseline" in serialized
    assert "not prove" in serialized
    assert "ask_submitter" not in serialized
    assert "fail_meaning" not in serialized
    assert "measure" not in report["results"]["table"][0]


def test_html_report_uses_the_same_safe_aggregates() -> None:
    report = render_html_report(make_run_result()).lower()

    assert "release with modifications" in report
    assert "dcr_train" not in report
    assert "exact_match</" not in report
    assert "0.123456" not in report
    assert "passing these checks" in report
    assert "ask the submitter" not in report
    assert "what a fail means" not in report


def test_subset_report_aggregates_metrics_and_preserves_qi_guard_counts() -> None:
    result = make_run_result()
    subset_results = [
        MetricResult(
            name="subset_min_n_guard",
            level="subset",
            group="qi:core",
            value=None,
            baseline=None,
            ci=None,
            n_records=None,
            n_flagged=None,
            status="suppressed",
            reason="subset partition is below the configured minimum cell size",
        )
        for _ in range(3)
    ]
    subset_results.extend(
        MetricResult(
            name="exact_match_rate",
            level="subset",
            group="qi:core:core",
            value=value,
            baseline=0.05,
            ci=None,
            n_records=10,
            n_flagged=None,
            status=status,
        )
        for value, status in (
            (0.1, "pass"),
            (0.2, "pass"),
            (0.3, "pass"),
            (0.4, "review"),
            (0.5, "review"),
            (0.6, "fail"),
        )
    )
    subset_results.extend(
        MetricResult(
            name="cap_accuracy",
            level="subset",
            group="qi:core:core:income",
            value=value,
            baseline=0.4,
            ci=None,
            n_records=10,
            n_flagged=None,
            status="review",
        )
        for value in (0.7, 0.8, 0.9)
    )
    result = replace(result, subset_results=tuple(subset_results))

    report = json.loads(render_json_report(result))
    subset = report["results"]["subset"]

    assert len(subset) == 1
    assert subset[0]["group"] == "qi:core"
    assert subset[0]["subset_min_n_guard"]["suppressed_groups"] == {
        "value": 3,
        "status": "reported",
        "reason": "",
    }
    metrics = {metric["name"]: metric for metric in subset[0]["metrics"]}
    metric = metrics["exact_match_rate"]
    assert metric["name"] == "exact_match_rate"
    assert metric["n_groups"]["value"] == 6
    assert metric["median_value"] == 0.35
    assert metric["median_baseline"] == 0.05
    assert metric["status_counts"]["pass"]["value"] == 3
    assert metric["status_counts"]["review"]["status"] == "suppressed"
    assert metric["status_counts"]["fail"]["status"] == "suppressed"
    assert metrics["cap_accuracy"]["median_value"] == 0.8
    assert report["applicability"] == [
        {
            "level": "subset",
            "status": "suppressed",
            "count": {"value": 3, "status": "reported", "reason": ""},
        }
    ]
    assert "ask_submitter" not in render_json_report(result)
    html = render_html_report(result).lower()
    assert "qi:core" in html
    assert "median value" in html
    assert "ask the submitter" not in html


def test_suppressed_generic_group_label_survives_output_filter() -> None:
    result = MetricResult(
        name="subset_min_n_guard",
        level="subset",
        group="qi:core",
        value=None,
        baseline=None,
        ci=None,
        n_records=None,
        n_flagged=None,
        status="suppressed",
        reason="subset partition is below the configured minimum cell size",
    )

    filtered = OutputFilter(min_cell_size=3, rounding_base=1).filter_metric_result(
        result
    )

    assert filtered.group == "qi:core"
