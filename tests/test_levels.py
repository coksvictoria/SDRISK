"""Tests for subset, cluster, and internal row-profile levels."""

import json
from dataclasses import asdict

import pandas as pd
import pytest

from sdrisk.bundle import DatasetBundle
from sdrisk.levels.cluster import analyze_clusters
from sdrisk.levels.row import RowProfile, build_row_store
from sdrisk.levels.subset import analyze_subsets
from sdrisk.output_filter import OutputFilter
from sdrisk.policy import PolicyProfile
from sdrisk.provenance import make_rng
from sdrisk.schema import Schema, Variable


def make_profile(
    *,
    min_cell_size: int = 2,
    min_cluster_size: int = 2,
    k: int = 2,
) -> PolicyProfile:
    return PolicyProfile.model_validate(
        {
            "name": "level-test",
            "release_tier": "in_lab",
            "output": {"min_cell_size": min_cell_size, "rounding_base": 1},
            "variables": {
                "qi_sets": [{"name": "core", "columns": ["key", "x"]}],
                "sensitive": ["secret"],
                "strata": ["region"],
            },
            "identity": {"k_profile_thresholds": [2, 5]},
            "distance": {
                "max_rows_exact": 100,
                "subsample_rows": 100,
                "near_match_threshold": 0.05,
            },
            "clusters": {"k": k, "min_cluster_size": min_cluster_size, "seeds": 3},
            "row_risk": {"medium_min_flags": 1, "high_min_flags": 2},
            "thresholds": {"risk": {"review": 0.1, "fail": 0.5}},
            "policy": {"require_test_for_public_tier": True},
        }
    )


def make_schema() -> Schema:
    return Schema(
        variables=[
            Variable(name="key", vtype="categorical", role=["qi"]),
            Variable(name="x", vtype="numeric", role=["qi"]),
            Variable(name="region", vtype="categorical", role=["stratum"]),
            Variable(name="secret", vtype="categorical", role=["sensitive"]),
        ]
    )


def make_cluster_bundle() -> DatasetBundle:
    train = pd.DataFrame(
        {
            "key": ["A"] * 6 + ["B"] * 6,
            "x": [float(value) for value in range(6)] + [float(value) for value in range(100, 106)],
            "region": ["north"] * 6 + ["south"] * 6,
            "secret": ["private-alpha"] * 6 + ["private-beta"] * 6,
        }
    )
    synthetic = pd.DataFrame(
        {
            "key": ["A"] * 10 + ["B"] * 2,
            "x": [float(value) for value in range(6)] + [0.1, 1.1, 2.1, 3.1, 100.0, 101.0],
            "region": ["north"] * 10 + ["south"] * 2,
            "secret": ["private-alpha"] * 10 + ["private-beta"] * 2,
        }
    )
    test = pd.DataFrame(
        {
            "key": ["A"] * 6 + ["B"] * 6,
            "x": [float(value) + 0.25 for value in range(6)]
            + [float(value) + 0.25 for value in range(100, 106)],
            "region": ["north"] * 6 + ["south"] * 6,
            "secret": ["private-alpha"] * 6 + ["private-beta"] * 6,
        }
    )
    return DatasetBundle(train, synthetic, test)


def test_subset_analysis_includes_strata_and_qi_classes_with_small_n_guards() -> None:
    analysis = analyze_subsets(
        make_cluster_bundle(),
        make_schema(),
        make_profile(),
        make_rng(17),
    )

    assert analysis.n_subsets_tested.value == 2
    assert analysis.n_subsets_tested.status == "reported"
    assert analysis.n_subsets_suppressed.value == 28
    assert analysis.correction_method == "none"
    assert "multiple" in analysis.multiple_comparison_caution.lower()
    assert analysis.metrics
    assert all(result.level == "subset" for result in analysis.metrics)
    assert any(
        result.name == "subset_min_n_guard" and result.status == "suppressed"
        for result in analysis.metrics
    )
    assert all("north" not in (result.group or "") for result in analysis.metrics)


def test_cluster_analysis_reports_stability_and_localized_exposure_ratio() -> None:
    analysis = analyze_clusters(
        make_cluster_bundle(),
        make_schema(),
        make_profile(),
        make_rng(23),
    )

    stability = next(
        result for result in analysis.results if result.name == "cluster_stability_ari"
    )
    exposure = [
        result
        for result in analysis.results
        if result.name == "cluster_exposure_ratio" and result.status == "review"
    ]

    assert stability.value is not None
    assert stability.baseline == 1.0
    assert stability.value > 0.9
    assert stability.reason
    assert len(exposure) == 2
    assert max(result.value or 0.0 for result in exposure) > 1.5
    assert all(result.baseline == pytest.approx(1.0) for result in exposure)
    assert all(result.experimental for result in exposure)
    assert any(result.level == "cluster" for result in analysis.results)


def test_cluster_exposure_is_not_assessed_without_a_holdout_baseline() -> None:
    bundle = make_cluster_bundle()
    without_test = DatasetBundle(bundle.real_train, bundle.synthetic)

    analysis = analyze_clusters(
        without_test,
        make_schema(),
        make_profile(),
        make_rng(23),
    )
    exposure = [result for result in analysis.results if result.name == "cluster_exposure_ratio"]

    assert exposure
    assert all(result.status == "not_assessed" for result in exposure)


def test_row_profiles_produce_configured_bands_and_only_filtered_aggregates() -> None:
    schema = make_schema()
    train = pd.DataFrame(
        {
            "key": ["A", "B", "C", "D"],
            "x": [0.0, 10.0, 20.0, 30.0],
            "region": ["north", "north", "south", "south"],
            "secret": ["private-alpha", "private-beta", "private-gamma", "private-delta"],
        }
    )
    synthetic = pd.DataFrame(
        {
            "key": ["A", "B", "Z"],
            "x": [0.0, 10.2, 100.0],
            "region": ["north", "north", "south"],
            "secret": ["private-alpha", "private-beta", "private-epsilon"],
        }
    )
    profile = make_profile(min_cell_size=1)
    store = build_row_store(
        DatasetBundle(train, synthetic),
        schema,
        profile,
    )
    counts = store.aggregate_band_counts(OutputFilter(min_cell_size=1, rounding_base=1))

    assert {field.name for field in RowProfile.__dataclass_fields__.values()} == {
        "dcr_train",
        "dcr_percentile",
        "exact_match",
        "near_match",
        "replicated_unique",
        "k_syn",
        "k_real",
        "attribute_disclosure",
        "local_density_ratio",
        "cluster_id",
        "cluster_exposure_ratio",
        "composite_band",
    }
    assert {key: value.value for key, value in counts.items()} == {
        "low": 1,
        "medium": 1,
        "high": 1,
        "total": 3,
    }
    internal_profiles = store._profiles
    assert internal_profiles[0]["exact_match"] is True
    assert internal_profiles[0]["attribute_disclosure"] is True
    assert internal_profiles[0]["composite_band"] == "high"
    assert internal_profiles[1]["composite_band"] == "medium"
    assert internal_profiles[2]["composite_band"] == "low"
    assert "private-alpha" not in repr(store)
    assert not hasattr(store, "to_dict")
    exported = json.dumps({key: asdict(value) for key, value in counts.items()})
    for row_field in ("dcr_train", "attribute_disclosure", "cluster_id"):
        assert row_field not in exported
    assert "private-alpha" not in exported


def test_row_store_suppresses_small_band_counts_and_keeps_only_filtered_values() -> None:
    store = build_row_store(
        DatasetBundle(
            pd.DataFrame(
                {
                    "key": ["A", "B", "C", "D"],
                    "x": [0.0, 10.0, 20.0, 30.0],
                    "region": ["north", "north", "south", "south"],
                    "secret": ["a", "b", "c", "d"],
                }
            ),
            pd.DataFrame(
                {
                    "key": ["A", "B", "Z"],
                    "x": [0.0, 10.2, 100.0],
                    "region": ["north", "north", "south"],
                    "secret": ["a", "b", "z"],
                }
            ),
        ),
        make_schema(),
        make_profile(min_cell_size=2),
    )

    counts = store.aggregate_band_counts(OutputFilter(min_cell_size=2, rounding_base=1))

    assert all(counts[band].status == "suppressed" for band in ("low", "medium", "high"))
    assert counts["total"].value == 3
