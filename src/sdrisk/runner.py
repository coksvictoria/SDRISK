"""Orchestrate table, subset, cluster, and internal row-level assessment."""

from collections.abc import Mapping
from pathlib import Path
from typing import Literal

from sdrisk.bundle import DatasetBundle
from sdrisk.decision import decide_results
from sdrisk.levels.cluster import analyze_clusters
from sdrisk.levels.common import tier1_results
from sdrisk.levels.row import build_row_store
from sdrisk.levels.subset import analyze_subsets
from sdrisk.metrics.attacks.linkability import linkability
from sdrisk.metrics.attacks.singling_out import singling_out
from sdrisk.metrics.membership.density_ratio import density_ratio_mia
from sdrisk.metrics.membership.distance import distance_mia
from sdrisk.output_filter import OutputFilter
from sdrisk.policy import PolicyProfile
from sdrisk.provenance import (
    RunProvenance,
    build_provenance,
    dataframe_sha256,
    make_rng,
)
from sdrisk.results import RunResult
from sdrisk.schema import Schema


def run_assessment(
    bundle: DatasetBundle,
    schema: Schema,
    profile: PolicyProfile,
    *,
    seed: int,
    input_files: Mapping[str, str | Path] | None = None,
) -> RunResult:
    """Run all implemented metrics and return filtered aggregates plus RowStore."""
    rng = make_rng(seed)
    table_results = [
        *tier1_results(bundle, schema, profile, rng),
        *distance_mia(bundle, schema, profile, rng),
        *density_ratio_mia(bundle, schema, profile, rng),
        *singling_out(bundle, schema, profile, rng),
        *linkability(bundle, schema, profile, rng),
    ]
    subset_analysis = analyze_subsets(bundle, schema, profile, rng)
    cluster_analysis = analyze_clusters(bundle, schema, profile, rng)
    subset_results = list(subset_analysis.metrics)
    cluster_results = list(cluster_analysis.results)

    row_store = None
    row_status: Literal["assessed", "not_assessed"] = "not_assessed"
    row_reason = ""
    if profile.row_risk is None:
        row_reason = "row-risk policy is not configured"
    elif len(bundle.real_train) < 2:
        row_reason = "row profiles require at least two real training rows"
    elif bundle.synthetic.empty:
        row_reason = "row profiles require at least one synthetic row"
    elif bundle.real_test is not None and bundle.real_test.empty:
        row_reason = "row profiles require a non-empty supplied real_test"
    else:
        row_store = build_row_store(
            bundle,
            schema,
            profile,
            cluster_analysis=cluster_analysis,
        )
        row_status = "assessed"

    output_filter = OutputFilter(
        min_cell_size=profile.output.min_cell_size,
        rounding_base=profile.output.rounding_base,
    )
    table_results = [
        output_filter.filter_metric_result(result) for result in table_results
    ]
    subset_results = [
        output_filter.filter_metric_result(result) for result in subset_results
    ]
    cluster_results = [
        output_filter.filter_metric_result(result) for result in cluster_results
    ]
    all_results = (*table_results, *subset_results, *cluster_results)
    decision = decide_results(all_results, profile, has_holdout=bundle.real_test is not None)
    provenance = _make_run_provenance(
        bundle,
        schema,
        profile,
        seed=seed,
        input_files=input_files,
    )
    n_table = len(table_results)
    n_subset = len(subset_results)
    classified_results = decision.results
    return RunResult(
        table_results=classified_results[:n_table],
        subset_results=classified_results[n_table : n_table + n_subset],
        cluster_results=classified_results[n_table + n_subset :],
        row_store=row_store,
        row_status=row_status,
        provenance=provenance,
        decision=decision,
        profile=profile,
        row_reason=row_reason,
    )


def _make_run_provenance(
    bundle: DatasetBundle,
    schema: Schema,
    profile: PolicyProfile,
    *,
    seed: int,
    input_files: Mapping[str, str | Path] | None,
) -> RunProvenance:
    """Hash configuration and inputs while recording actual holdout fractions."""
    config = {
        "profile": profile.model_dump(mode="json"),
        "schema": schema.model_dump(mode="json"),
    }
    frames = {
        "real_train": bundle.real_train,
        "synthetic": bundle.synthetic,
    }
    if bundle.real_test is not None:
        frames["real_test"] = bundle.real_test
    if input_files is None:
        files: Mapping[str, str | Path] = {}
        content_hashes = {
            name: dataframe_sha256(frame) for name, frame in frames.items()
        }
    else:
        unknown = set(input_files) - set(frames)
        if unknown:
            raise ValueError(f"input file names do not match bundle: {', '.join(sorted(unknown))}")
        files = input_files
        content_hashes = {
            name: dataframe_sha256(frame)
            for name, frame in frames.items()
            if name not in input_files
        }
    provenance = build_provenance(
        config,
        files,
        seed=seed,
        sampling_fractions=_sampling_fractions(bundle),
    )
    return RunProvenance(
        config_hash=provenance.config_hash,
        input_hashes={**provenance.input_hashes, **content_hashes},
        package_version=provenance.package_version,
        seed=provenance.seed,
        sampling_fractions=provenance.sampling_fractions,
    )


def _sampling_fractions(bundle: DatasetBundle) -> dict[str, float]:
    if bundle.real_test is None or len(bundle.real_test) < 4:
        return {}
    n_test = len(bundle.real_test)
    n_reference = n_test // 2
    return {
        "density_ratio_reference": n_reference / n_test,
        "density_ratio_control": (n_test - n_reference) / n_test,
    }
