"""Hand-computable table-level identity metric tests."""

import pandas as pd
import pytest

from sdrisk.bundle import DatasetBundle
from sdrisk.metrics.identity.dcr import dcr_overfitting, dcr_vs_random
from sdrisk.metrics.identity.exact_match import exact_match_rate, unique_exact_match_rate
from sdrisk.metrics.identity.kprofile import k_profile
from sdrisk.metrics.identity.repu import replicated_uniques
from sdrisk.policy import PolicyProfile
from sdrisk.provenance import make_rng
from sdrisk.schema import Schema, Variable


def make_profile(
    *, min_cell_size: int = 1, qi_columns: tuple[str, ...] = ("key", "x")
) -> PolicyProfile:
    return PolicyProfile.model_validate(
        {
            "name": "identity-test",
            "release_tier": "in_lab",
            "output": {"min_cell_size": min_cell_size, "rounding_base": 1},
            "variables": {
                "qi_sets": [{"name": "core", "columns": list(qi_columns)}],
                "sensitive": [],
                "strata": [],
            },
            "identity": {"k_profile_thresholds": [2, 5, 10]},
            "distance": {
                "max_rows_exact": 100,
                "subsample_rows": 100,
                "near_match_threshold": 0.1,
            },
            "clusters": {"k": 2, "min_cluster_size": 2, "seeds": 1},
            "thresholds": {"exact_match_rate": {"review": 0.1, "fail": 0.5}},
            "policy": {"require_test_for_public_tier": True},
        }
    )


def make_schema(*, tolerance: float | None = 0.5) -> Schema:
    return Schema(
        variables=[
            Variable(name="key", vtype="categorical", role=["qi"]),
            Variable(name="x", vtype="numeric", role=["qi"], tolerance=tolerance),
            Variable(name="record_id", vtype="categorical", role=["id"]),
        ]
    )


def make_bundle(*, with_test: bool = True) -> DatasetBundle:
    real_train = pd.DataFrame(
        {"key": ["A", "A", "B", "C"], "x": [0.0, 0.0, 1.0, 2.0], "record_id": list("abcd")}
    )
    synthetic = pd.DataFrame(
        {"key": ["A", "B", "C", "D"], "x": [0.0, 1.4, 2.0, 3.0], "record_id": list("wxyz")}
    )
    real_test = (
        pd.DataFrame(
            {"key": ["A", "D", "E"], "x": [0.1, 5.0, 6.0], "record_id": ["t1", "t2", "t3"]}
        )
        if with_test
        else None
    )
    return DatasetBundle(real_train, synthetic, real_test)


def test_exact_match_rate_uses_schema_tolerance_and_holdout_baseline() -> None:
    result = exact_match_rate(make_bundle(), make_schema(), make_profile())[0]

    assert result.name == "exact_match_rate"
    assert result.value == 1.0
    assert result.baseline == 0.5
    assert result.n_records == 4
    assert result.n_flagged == 4


def test_unique_exact_match_uses_unique_train_and_test_records() -> None:
    result = unique_exact_match_rate(make_bundle(), make_schema(), make_profile())[0]

    assert result.value == 1.0
    assert result.baseline == pytest.approx(1 / 3)
    assert result.n_records == 2
    assert result.n_flagged == 2


def test_repu_is_reported_against_uio_baseline() -> None:
    result = replicated_uniques(make_bundle(), make_schema(), make_profile())[0]

    assert result.name == "repu_pct"
    assert result.value == 25.0
    assert result.baseline == 50.0
    assert result.n_flagged == 1


def test_k_profile_compares_synthetic_low_k_shares_with_real_profile() -> None:
    results = k_profile(make_bundle(), make_schema(), make_profile())
    by_name = {result.name: result for result in results}

    assert by_name["k_profile_below_2"].value == 1.0
    assert by_name["k_profile_below_2"].baseline == 0.5
    assert by_name["k_profile_below_5"].value == 1.0
    assert by_name["k_profile_below_5"].baseline == 1.0


def test_holdout_dependent_identity_metrics_are_not_assessed_without_test() -> None:
    bundle = make_bundle(with_test=False)

    exact = exact_match_rate(bundle, make_schema(), make_profile())[0]
    unique = unique_exact_match_rate(bundle, make_schema(), make_profile())[0]
    dcr = dcr_overfitting(bundle, make_schema(), make_profile())[0]

    assert exact.status == "not_assessed"
    assert unique.status == "not_assessed"
    assert dcr.status == "not_assessed"
    assert exact.value is None
    assert exact.reason


def test_output_filter_suppresses_small_flag_counts_for_metrics() -> None:
    bundle = make_bundle()
    strict_profile = make_profile(min_cell_size=3)

    result = exact_match_rate(bundle, make_schema(), strict_profile)[0]

    assert result.status == "suppressed"
    assert result.value is None
    assert result.baseline is None


def test_dcr_overfitting_compares_synthetic_nearest_distances() -> None:
    schema = Schema(variables=[Variable(name="x", vtype="numeric", role=["qi"])])
    profile = make_profile(qi_columns=("x",))
    bundle = DatasetBundle(
        real_train=pd.DataFrame({"x": [0.0, 10.0]}),
        synthetic=pd.DataFrame({"x": [0.0, 10.0]}),
        real_test=pd.DataFrame({"x": [20.0, 30.0]}),
    )

    result = dcr_overfitting(bundle, schema, profile)[0]

    assert result.value == 1.0
    assert result.baseline == 0.5
    assert result.n_flagged == 2


def test_dcr_overfitting_does_not_classify_tied_distances() -> None:
    schema = Schema(variables=[Variable(name="x", vtype="numeric", role=["qi"])])
    bundle = DatasetBundle(
        real_train=pd.DataFrame({"x": [0.0, 10.0]}),
        synthetic=pd.DataFrame({"x": [2.0]}),
        real_test=pd.DataFrame({"x": [4.0]}),
    )

    result = dcr_overfitting(bundle, schema, make_profile(qi_columns=("x",)))[0]

    assert result.status == "not_assessed"
    assert "tie" in result.reason


def test_dcr_vs_random_is_reproducible_and_reports_random_baseline() -> None:
    schema = Schema(
        variables=[
            Variable(name="x", vtype="numeric", role=["qi"]),
            Variable(name="group", vtype="categorical", role=["qi"]),
        ]
    )
    bundle = DatasetBundle(
        real_train=pd.DataFrame(
            {"x": [0.0, 10.0, 20.0, 30.0], "group": ["a", "a", "b", "b"]}
        ),
        synthetic=pd.DataFrame(
            {"x": [1.0, 11.0, 21.0, 31.0], "group": ["a", "a", "b", "b"]}
        ),
    )
    profile = PolicyProfile.model_validate(
        {
            "name": "dcr-test",
            "release_tier": "in_lab",
            "output": {"min_cell_size": 1, "rounding_base": 1},
            "variables": {
                "qi_sets": [{"name": "core", "columns": ["x", "group"]}],
                "sensitive": [],
                "strata": [],
            },
            "identity": {"k_profile_thresholds": [2, 5, 10]},
            "distance": {
                "max_rows_exact": 100,
                "subsample_rows": 100,
                "near_match_threshold": 0.1,
            },
            "clusters": {"k": 2, "min_cluster_size": 2, "seeds": 1},
            "thresholds": {"risk": {"review": 0.1, "fail": 0.5}},
            "policy": {"require_test_for_public_tier": True},
        }
    )

    first = dcr_vs_random(bundle, schema, profile, make_rng(0))[0]
    second = dcr_vs_random(bundle, schema, profile, make_rng(0))[0]

    assert first.value is not None
    assert first.value == second.value
    assert first.baseline == 1.0


def test_dcr_vs_random_is_not_assessed_when_random_median_is_zero() -> None:
    schema = Schema(variables=[Variable(name="x", vtype="numeric", role=["qi"])])
    bundle = DatasetBundle(
        real_train=pd.DataFrame({"x": [0.0, 1.0]}),
        synthetic=pd.DataFrame({"x": [0.5, 0.5]}),
    )

    result = dcr_vs_random(
        bundle,
        schema,
        make_profile(qi_columns=("x",)),
        make_rng(0),
    )[0]

    assert result.status == "not_assessed"
    assert result.value is None
    assert "median is zero" in result.reason
