"""Disclosure-safe JSON report and audit-log serialization."""

import json
from collections import Counter, defaultdict
from dataclasses import asdict
from pathlib import Path
from statistics import median
from typing import Any

from sdrisk.output_filter import FilteredCount, OutputFilter
from sdrisk.policy import QISet
from sdrisk.results import MetricResult, RunResult

LIMITATIONS = (
    "Passing these checks lowers concern but does not prove the data cannot be "
    "disclosive. Thresholds are policy choices. Attacks requiring generator "
    "access were not run."
)
def build_report_payload(result: RunResult) -> dict[str, Any]:
    """Build aggregate-only content after applying the output filter."""
    output_filter = OutputFilter(
        min_cell_size=result.profile.output.min_cell_size,
        rounding_base=result.profile.output.rounding_base,
    )
    result_groups = {
        "table": result.table_results,
        "subset": result.subset_results,
        "cluster": result.cluster_results,
    }
    metrics = {
        "table": [
            _metric_payload(item, output_filter) for item in result.table_results
        ],
        "subset": _subset_payload(
            result.subset_results,
            result.profile.variables.qi_sets,
            result.profile.variables.strata,
            output_filter,
        ),
        "cluster": [
            _metric_payload(item, output_filter) for item in result.cluster_results
        ],
    }
    applicability = _applicability_summary(result_groups, output_filter)
    row_bands: dict[str, Any] = {"status": result.row_status}
    if result.row_store is None:
        row_bands["reason"] = result.row_reason
    else:
        counts = result.row_store.aggregate_band_counts(output_filter)
        row_bands.update(
            {
                name: _count_payload(count)
                for name, count in counts.items()
            }
        )
    return {
        "profile": {
            "name": result.profile.name,
            "release_tier": result.profile.release_tier,
        },
        "provenance": asdict(result.provenance),
        "applicability": applicability,
        "results": metrics,
        "row_bands": row_bands,
        "decision": {
            "recommendation": result.decision.recommendation,
            "confidence": result.decision.confidence,
            "rationale": list(result.decision.rationale),
            "checker_decision": None,
            "checker_rationale": None,
        },
        "limitations": LIMITATIONS,
    }


def render_json_report(result: RunResult) -> str:
    """Return formatted JSON containing only filtered aggregate results."""
    return json.dumps(
        build_report_payload(result),
        indent=2,
        sort_keys=True,
        allow_nan=False,
    )


def write_json_report(result: RunResult, path: str | Path) -> Path:
    """Write a disclosure-safe JSON report to `path`."""
    output_path = Path(path)
    output_path.write_text(render_json_report(result) + "\n", encoding="utf-8")
    return output_path


def write_audit_log(result: RunResult, path: str | Path) -> Path:
    """Write run provenance and the checker-facing recommendation."""
    payload = {
        "provenance": asdict(result.provenance),
        "decision": {
            "recommendation": result.decision.recommendation,
            "confidence": result.decision.confidence,
            "rationale": list(result.decision.rationale),
            "checker_decision": None,
            "checker_rationale": None,
        },
    }
    output_path = Path(path)
    output_path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return output_path


def _metric_payload(
    result: MetricResult,
    output_filter: OutputFilter,
) -> dict[str, Any]:
    filtered = output_filter.filter_metric_result(result)
    return {
        "name": filtered.name,
        "level": filtered.level,
        "group": filtered.group,
        "value": filtered.value,
        "baseline": filtered.baseline,
        "ci": list(filtered.ci) if filtered.ci is not None else None,
        "n_records": filtered.n_records,
        "n_flagged": filtered.n_flagged,
        "status": filtered.status,
        "reason": (
            filtered.reason
            if filtered.status in ("suppressed", "not_assessed")
            else ""
        ),
        "experimental": filtered.experimental,
    }


def _subset_payload(
    results: tuple[MetricResult, ...],
    qi_sets: list[QISet],
    strata: list[str],
    output_filter: OutputFilter,
) -> list[dict[str, Any]]:
    grouped: dict[str | None, list[MetricResult]] = defaultdict(list)
    for result in results:
        filtered = output_filter.filter_metric_result(result)
        group = _subset_group_name(filtered.group, qi_sets, strata)
        grouped[group].append(filtered)

    summaries: list[dict[str, Any]] = []
    for group, group_results in sorted(
        grouped.items(), key=lambda item: item[0] or ""
    ):
        guard_results = [
            item for item in group_results if item.name == "subset_min_n_guard"
        ]
        metric_results: dict[str, list[MetricResult]] = defaultdict(list)
        for item in group_results:
            if item.name != "subset_min_n_guard":
                metric_results[item.name].append(item)
        summaries.append(
            {
                "group": group,
                "subset_min_n_guard": {
                    "suppressed_groups": _count_payload(
                        output_filter.filter_count(len(guard_results))
                    ),
                    "reasons": sorted(
                        {item.reason for item in guard_results if item.reason}
                    ),
                },
                "aggregation": "unweighted median across reportable subset groups",
                "metrics": [
                    _subset_metric_payload(name, items, output_filter)
                    for name, items in sorted(metric_results.items())
                ],
            }
        )
    return summaries


def _subset_group_name(
    group: str | None,
    qi_sets: list[QISet],
    strata: list[str],
) -> str | None:
    """Normalize generic labels to one configured QI set or stratum."""
    if group is None:
        return None
    for qi_set in qi_sets:
        qi_prefix = f"qi:{qi_set.name}"
        if group == qi_prefix or group.startswith(f"{qi_prefix}:"):
            return qi_prefix
    for stratum in strata:
        stratum_prefix = f"stratum:{stratum}:"
        if group.startswith(stratum_prefix):
            qiset_label = group.removeprefix(stratum_prefix)
            for qi_set in qi_sets:
                if qiset_label == qi_set.name or qiset_label.startswith(
                    f"{qi_set.name}:"
                ):
                    return f"{stratum_prefix}{qi_set.name}"
    return group


def _subset_metric_payload(
    name: str,
    results: list[MetricResult],
    output_filter: OutputFilter,
) -> dict[str, Any]:
    status_counts = Counter(item.status for item in results)
    raw_counts: dict[str, int] = {}
    for status, count in status_counts.items():
        if count > 0:
            raw_counts[status] = count
    raw_counts["total"] = sum(raw_counts.values())
    filtered_counts = output_filter.filter_count_table(
        raw_counts,
        total_key="total",
    )
    value_pairs = [
        (item.value, item.baseline)
        for item in results
        if item.value is not None
        and item.baseline is not None
        and item.status not in ("suppressed", "not_assessed")
    ]
    filtered_value_count = output_filter.filter_count(len(value_pairs))
    can_report_median = filtered_value_count.status == "reported"
    return {
        "name": name,
        "n_groups": _count_payload(filtered_counts["total"]),
        "status_counts": {
            status: _count_payload(filtered_counts[status])
            for status in status_counts
        },
        "n_groups_with_value_and_baseline": _count_payload(filtered_value_count),
        "median_value": (
            round(float(median(value for value, _ in value_pairs)), 4)
            if can_report_median
            else None
        ),
        "median_baseline": (
            round(float(median(baseline for _, baseline in value_pairs)), 4)
            if can_report_median
            else None
        ),
        "experimental": any(item.experimental for item in results),
        "reasons": sorted(
            {
                item.reason
                for item in results
                if item.status in ("suppressed", "not_assessed") and item.reason
            }
        ),
    }


def _applicability_summary(
    result_groups: dict[str, tuple[MetricResult, ...]],
    output_filter: OutputFilter,
) -> list[dict[str, Any]]:
    counts: Counter[tuple[str, str]] = Counter()
    for level, results in result_groups.items():
        for result in results:
            filtered = output_filter.filter_metric_result(result)
            if filtered.status in ("suppressed", "not_assessed"):
                counts[(level, filtered.status)] += 1
    return [
        {
            "level": level,
            "status": status,
            "count": _count_payload(output_filter.filter_count(count)),
        }
        for (level, status), count in sorted(counts.items())
    ]


def _count_payload(count: FilteredCount) -> dict[str, Any]:
    return {"value": count.value, "status": count.status, "reason": count.reason}
