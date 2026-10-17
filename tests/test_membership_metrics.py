"""Hand-computable membership-inference metric tests."""

import pandas as pd
import pytest

from sdrisk.bundle import DatasetBundle
from sdrisk.metrics.membership.density_ratio import (
    _split_reference_and_control,
    density_ratio_mia,
)
from sdrisk.metrics.membership.distance import distance_mia
from sdrisk.policy import PolicyProfile
from sdrisk.provenance import make_rng
from sdrisk.schema import Schema, Variable


def make_profile(*, min_cell_size: int = 1) -> PolicyProfile:
    return PolicyProfile.model_validate(
        {
            "name": "mia-test",
            "release_tier": "in_lab",
            "output": {"min_cell_size": min_cell_size, "rounding_base": 1},
            "variables": {
                "qi_sets": [{"name": "core", "columns": ["x"]}],
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
            "thresholds": {"mia_auc": {"review": 0.55, "fail": 0.65}},
            "policy": {"require_test_for_public_tier": True},
        }
    )


def make_numeric_schema() -> Schema:
    return Schema(variables=[Variable(name="x", vtype="numeric", role=["qi"])])


def make_numeric_bundle(*, with_test: bool = True) -> DatasetBundle:
    train = pd.DataFrame({"x": [0.0, 1.0, 2.0, 3.0]})
    synthetic = pd.DataFrame({"x": [0.0, 1.0, 2.0, 3.0]})
    test = pd.DataFrame({"x": [10.0, 11.0, 12.0, 13.0]}) if with_test else None
    return DatasetBundle(train, synthetic, test)


def test_distance_mia_reports_roc_metrics_against_random_baselines() -> None:
    results = distance_mia(
        make_numeric_bundle(),
        make_numeric_schema(),
        make_profile(),
        make_rng(5),
    )
    by_name = {result.name: result for result in results}

    assert by_name["mia_distance_auc"].value == 1.0
    assert by_name["mia_distance_auc"].baseline == 0.5
    assert by_name["mia_distance_advantage"].value == 1.0
    assert by_name["mia_distance_advantage"].baseline == 0.0
    assert by_name["mia_distance_tpr_at_1pct_fpr"].value == 1.0
    assert by_name["mia_distance_tpr_at_1pct_fpr"].baseline == 0.01
    assert by_name["mia_distance_tpr_at_0_1pct_fpr"].value == 1.0
    assert by_name["mia_distance_tpr_at_0_1pct_fpr"].baseline == 0.001
    assert all(result.ci is not None for result in results)


def test_membership_metrics_remain_near_baseline_on_independent_regular_samples() -> None:
    train = pd.DataFrame({"x": [index / 60 for index in range(0, 60, 2)]})
    test = pd.DataFrame({"x": [index / 60 for index in range(1, 60, 2)]})
    synthetic = pd.DataFrame({"x": [(index + 0.5) / 60 for index in range(60)]})
    bundle = DatasetBundle(train, synthetic, test)

    distance = distance_mia(bundle, make_numeric_schema(), make_profile(), make_rng(43))
    density = density_ratio_mia(bundle, make_numeric_schema(), make_profile(), make_rng(43))
    distance_by_name = {result.name: result for result in distance}
    density_by_name = {result.name: result for result in density}

    assert distance_by_name["mia_distance_auc"].value == pytest.approx(0.5)
    assert distance_by_name["mia_distance_advantage"].value == pytest.approx(0.0)
    assert density_by_name["mia_density_auc"].value == pytest.approx(0.5, abs=0.2)
    assert density_by_name["mia_density_advantage"].value < 0.3


def test_distance_mia_bootstrap_is_reproducible() -> None:
    first = distance_mia(
        make_numeric_bundle(),
        make_numeric_schema(),
        make_profile(),
        make_rng(19),
    )
    second = distance_mia(
        make_numeric_bundle(),
        make_numeric_schema(),
        make_profile(),
        make_rng(19),
    )

    assert [(result.value, result.ci) for result in first] == [
        (result.value, result.ci) for result in second
    ]


def test_density_ratio_mia_detects_planted_support_overrepresentation() -> None:
    results = density_ratio_mia(
        make_numeric_bundle(),
        make_numeric_schema(),
        make_profile(),
        make_rng(29),
    )
    by_name = {result.name: result for result in results}

    assert by_name["mia_density_auc"].value is not None
    assert by_name["mia_density_auc"].value > 0.9
    assert by_name["mia_density_auc"].baseline == 0.5
    assert by_name["mia_density_advantage"].baseline == 0.0
    assert all(result.experimental for result in results)


def test_density_ratio_splits_reference_and_control_holdout_rows() -> None:
    test = pd.DataFrame({"x": [0.0, 1.0, 2.0, 3.0, 4.0]})

    reference, control = _split_reference_and_control(test, make_rng(7))

    assert len(reference) == 2
    assert len(control) == 3
    assert set(reference["x"]).isdisjoint(set(control["x"]))
    assert sorted(reference["x"].tolist() + control["x"].tolist()) == test["x"].tolist()


def test_density_ratio_is_not_assessed_with_too_few_holdout_rows() -> None:
    bundle = make_numeric_bundle()
    small_test = pd.DataFrame({"x": [10.0, 11.0, 12.0]})
    small_bundle = DatasetBundle(bundle.real_train, bundle.synthetic, small_test)

    results = density_ratio_mia(
        small_bundle,
        make_numeric_schema(),
        make_profile(),
        make_rng(17),
    )

    assert all(result.status == "not_assessed" for result in results)
    assert all("four" in result.reason for result in results)


def test_membership_filter_suppresses_small_train_and_holdout_partitions() -> None:
    results = distance_mia(
        make_numeric_bundle(),
        make_numeric_schema(),
        make_profile(min_cell_size=5),
        make_rng(31),
    )

    assert results
    assert all(result.status == "suppressed" for result in results)


@pytest.mark.parametrize("metric", [distance_mia, density_ratio_mia])
def test_membership_metrics_are_not_assessed_without_holdout(metric) -> None:
    results = metric(
        make_numeric_bundle(with_test=False),
        make_numeric_schema(),
        make_profile(),
        make_rng(11),
    )

    assert results
    assert all(result.status == "not_assessed" for result in results)
    assert all(result.reason == "no holdout supplied" for result in results)
