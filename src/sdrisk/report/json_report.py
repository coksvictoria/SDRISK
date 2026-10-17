"""Disclosure-safe JSON report and audit-log serialization."""

import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

from sdrisk.output_filter import FilteredCount, OutputFilter
from sdrisk.results import MetricResult, RunResult

LIMITATIONS = (
    "Passing these checks lowers concern but does not prove the data cannot be "
    "disclosive. Thresholds are policy choices. Attacks requiring generator "
    "access were not run."
)
_METRIC_NOTES: dict[str, tuple[str, str, str]] = {
    "exact_match_rate": (
        (
            "Share of training records with an exact match in the synthetic data "
            "on the configured QIs."
        ),
        (
            "A fail means exact training-record matches exceeded policy limits; "
            "it does not establish intent."
        ),
        (
            "Ask which QIs and tolerances are appropriate and whether generation "
            "or post-processing can reduce copying."
        ),
    ),
    "uem_rate": (
        (
            "Share of training records unique on the configured QIs that have an exact "
            "synthetic match."
        ),
        (
            "A fail means some identifiable training records are reproduced more often "
            "than policy allows."
        ),
        (
            "Ask whether the affected QIs are necessary and whether the synthetic output "
            "can be regenerated."
        ),
    ),
    "repu_pct": (
        "Percentage of original-data unique QI records that are also unique in the synthetic data.",
        "A fail means replicated uniqueness exceeds the configured policy threshold.",
        (
            "Ask whether QI detail can be coarsened or the synthetic data regenerated "
            "with less local copying."
        ),
    ),
    "dcr_overfit_share": (
        "Share of synthetic records closer to training records than to disjoint holdout records.",
        "A fail indicates potential training-specific overfitting under the configured distance.",
        "Ask how the generator was trained and whether an independent rerun changes this pattern.",
    ),
    "dcr_vs_random_ratio": (
        (
            "Median synthetic-to-training nearest-neighbor distance relative to a "
            "marginal-only reference."
        ),
        "A low risk signal can indicate synthetic records unusually close to training data.",
        (
            "Ask whether the distance variables, weights, and random reference are "
            "suitable for this release."
        ),
    ),
    "cap_accuracy": (
        "Accuracy of guessing a sensitive value from QI-matched synthetic records.",
        "A fail means guessing accuracy is materially above its configured baseline threshold.",
        "Ask whether the target is sensitive and whether QIs or target detail can be reduced.",
    ),
    "tcap_accuracy": (
        "Sensitive-value guessing accuracy among QI classes with a single synthetic target value.",
        "A fail indicates concentrated attribute disclosure in uniform synthetic QI classes.",
        "Ask whether these QI classes should be generalized or excluded from the release.",
    ),
    "disco_pct": (
        "Share of synthetic records meeting the configured disclosure-identity condition.",
        "A fail means the measured disclosure gain exceeded its configured threshold.",
        "Ask which key variables drive the result and whether they are needed at this detail.",
    ),
    "ml_inference_gain": (
        "Improvement in sensitive-attribute prediction over the configured baseline.",
        "A fail indicates the synthetic data supports stronger inference than the baseline.",
        (
            "Ask whether the target and predictors are appropriate and whether less "
            "detail can be released."
        ),
    ),
    "numeric_tolerance_recovery": (
        "Share of numeric targets recovered within the schema's configured tolerance.",
        "A fail indicates more values are recoverable within tolerance than policy permits.",
        (
            "Ask whether the tolerance reflects real-world disclosure and whether "
            "precision can be reduced."
        ),
    ),
    "homogeneity_share": (
        "Share of eligible synthetic QI classes containing only one sensitive value.",
        (
            "A fail indicates sensitive values may be inferable within more classes "
            "than the baseline allows."
        ),
        "Ask whether the QI classes or sensitive target should be coarsened.",
    ),
    "cluster_stability_ari": (
        "Agreement between seeded cluster fits, measured by adjusted Rand index.",
        "A low value means cluster-level results may depend on model initialization.",
        (
            "Ask for caution interpreting cluster findings and consider rerunning "
            "with validated settings."
        ),
    ),
    "cluster_exposure_ratio": (
        "Experimental ratio comparing a cluster's synthetic share with its training share.",
        "A high ratio relative to the holdout baseline may indicate localized over-representation.",
        (
            "Ask whether the subgroup is expected and validate this candidate "
            "diagnostic before relying on it."
        ),
    ),
    "singling_out_success": (
        "Success rate of simplified exact-QI guesses against training records.",
        (
            "A higher rate than the holdout baseline suggests training records are "
            "easier to single out."
        ),
        (
            "Ask which QIs enable unique guesses; this simplified attack is not a full "
            "Anonymeter implementation."
        ),
    ),
    "linkability_success": (
        "Success rate of simplified exact-block synthetic links against training records.",
        (
            "A higher rate than the holdout baseline suggests synthetic links identify "
            "training records."
        ),
        "Ask whether both QI blocks are necessary; this does not model external auxiliary tables.",
    ),
}
_MEMBERSHIP_SUFFIX_NOTES: dict[str, tuple[str, str, str]] = {
    "auc": (
        "How well the membership score separates training members from holdout non-members.",
        "A fail indicates separation above the configured chance-adjusted AUC threshold.",
        (
            "Ask whether the holdout is representative and whether the attacker "
            "assumptions fit the release."
        ),
    ),
    "advantage": (
        "Maximum true-positive-rate advantage over the false-positive rate.",
        "A fail indicates an attacker can achieve an elevated membership advantage.",
        "Ask whether the attack's distance or density representation matches a plausible attacker.",
    ),
    "tpr_at_1pct_fpr": (
        "Membership true-positive rate at a one-percent false-positive rate.",
        "A fail indicates elevated detection at this operating point.",
        "Ask whether that false-positive operating point reflects the intended threat model.",
    ),
    "tpr_at_0_1pct_fpr": (
        "Membership true-positive rate at a one-tenth-percent false-positive rate.",
        "A fail indicates elevated detection at this low-false-positive operating point.",
        "Ask whether rare-target membership is within the release's threat model.",
    ),
}


def build_report_payload(result: RunResult) -> dict[str, Any]:
    """Build aggregate-only content after applying the output filter."""
    output_filter = OutputFilter(
        min_cell_size=result.profile.output.min_cell_size,
        rounding_base=result.profile.output.rounding_base,
    )
    metrics = {
        "table": [_metric_payload(item, output_filter) for item in result.table_results],
        "subset": [_metric_payload(item, output_filter) for item in result.subset_results],
        "cluster": [_metric_payload(item, output_filter) for item in result.cluster_results],
    }
    applicability = [
        {
            "name": metric["name"],
            "level": metric["level"],
            "status": metric["status"],
            "reason": metric["reason"],
        }
        for items in metrics.values()
        for metric in items
        if metric["status"] in ("suppressed", "not_assessed")
    ]
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
    measure, fail_meaning, ask = _metric_note(filtered.name)
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
        "reason": filtered.reason,
        "experimental": filtered.experimental,
        "measure": measure,
        "fail_meaning": fail_meaning,
        "ask_submitter": ask,
    }


def _count_payload(count: FilteredCount) -> dict[str, Any]:
    return {"value": count.value, "status": count.status, "reason": count.reason}


def _metric_note(name: str) -> tuple[str, str, str]:
    direct = _METRIC_NOTES.get(name)
    if direct is not None:
        return direct
    if name.startswith("mia_distance_"):
        prefix = "Distance-based membership inference"
    elif name.startswith("mia_density_"):
        prefix = "Experimental KDE density-ratio membership inference"
    else:
        prefix = ""
    suffix = name.removeprefix("mia_distance_").removeprefix("mia_density_")
    membership_note = _MEMBERSHIP_SUFFIX_NOTES.get(suffix)
    if prefix and membership_note is not None:
        measure, fail_meaning, ask = membership_note
        return (f"{prefix}: {measure}", fail_meaning, ask)
    return (
        f"Supporting diagnostic `{name}`; interpret it beside its baseline and applicability.",
        "A fail means the configured threshold was reached; it is not proof of disclosure.",
        (
            "Review the metric definition, baseline, and limitations, then request "
            "clarification as needed."
        ),
    )
