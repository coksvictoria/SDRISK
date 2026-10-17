"""Hand-computable table-level attribute-metric tests."""

import pandas as pd
import pytest

from sdrisk.bundle import DatasetBundle
from sdrisk.metrics.attribute.cap import cap_disclosure, tcap_disclosure
from sdrisk.metrics.attribute.disco import disco
from sdrisk.metrics.attribute.homogeneity import homogeneity
from sdrisk.metrics.attribute.ml_inference import ml_inference_gain
from sdrisk.metrics.attribute.numeric_tolerance import numeric_tolerance_recovery
from sdrisk.policy import PolicyProfile
from sdrisk.provenance import make_rng
from sdrisk.schema import Schema, Variable


def make_profile(
    *,
    min_cell_size: int = 1,
    sensitive: tuple[str, ...] = ("secret", "amount"),
) -> PolicyProfile:
    return PolicyProfile.model_validate(
        {
            "name": "attribute-test",
            "release_tier": "in_lab",
            "output": {"min_cell_size": min_cell_size, "rounding_base": 1},
            "variables": {
                "qi_sets": [{"name": "core", "columns": ["key"]}],
                "sensitive": list(sensitive),
                "strata": [],
            },
            "identity": {"k_profile_thresholds": [2, 5, 10]},
            "distance": {
                "max_rows_exact": 100,
                "subsample_rows": 100,
                "near_match_threshold": 0.1,
            },
            "clusters": {"k": 2, "min_cluster_size": 2, "seeds": 1},
            "thresholds": {"attribute_risk": {"review": 0.1, "fail": 0.5}},
            "policy": {"require_test_for_public_tier": True},
        }
    )


def make_schema(*, amount_tolerance: float | None = 0.5) -> Schema:
    return Schema(
        variables=[
            Variable(name="key", vtype="categorical", role=["qi"]),
            Variable(name="secret", vtype="categorical", role=["sensitive"]),
            Variable(
                name="amount",
                vtype="numeric",
                role=["sensitive"],
                tolerance=amount_tolerance,
            ),
        ]
    )


def make_bundle() -> DatasetBundle:
    real_train = pd.DataFrame(
        {
            "key": ["a", "a", "b", "b", "b", "c"],
            "secret": ["x", "x", "y", "z", "z", "y"],
            "amount": [1.0, 2.0, 5.0, 6.0, 10.0, 11.0],
        }
    )
    synthetic = pd.DataFrame(
        {
            "key": ["a", "a", "a", "b", "b", "c"],
            "secret": ["x", "y", "x", "z", "z", "n"],
            "amount": [1.2, 3.0, 1.2, 5.2, 9.8, 11.2],
        }
    )
    return DatasetBundle(real_train, synthetic)


def for_target(results: list, name: str, target: str):
    return next(
        result
        for result in results
        if result.name == name and result.group is not None and target in result.group
    )


def test_cap_uses_synthetic_conditional_probability_and_marginal_baseline() -> None:
    result = for_target(
        cap_disclosure(make_bundle(), make_schema(), make_profile()),
        "cap_accuracy",
        "secret",
    )

    assert result.value == pytest.approx(5 / 9)
    assert result.baseline == pytest.approx(1 / 3)
    assert result.n_records == 6


def test_tcap_restricts_to_deterministic_synthetic_qi_classes() -> None:
    result = for_target(
        tcap_disclosure(make_bundle(), make_schema(), make_profile()),
        "tcap_accuracy",
        "secret",
    )

    assert result.value == pytest.approx(1 / 2)
    assert result.baseline == pytest.approx(1 / 3)
    assert result.n_records == 4
    assert result.n_flagged == 2


def test_disco_uses_original_denominator_and_dorig_baseline() -> None:
    result = for_target(disco(make_bundle(), make_schema(), make_profile()), "disco_pct", "secret")

    assert result.value == pytest.approx(100 / 3)
    assert result.baseline == 50.0
    assert result.n_records == 6
    assert result.n_flagged == 2


def test_numeric_tolerance_recovery_compares_qi_matched_values_to_marginal_guess() -> None:
    result = for_target(
        numeric_tolerance_recovery(make_bundle(), make_schema(), make_profile()),
        "numeric_tolerance_recovery",
        "amount",
    )

    assert result.value == pytest.approx(4 / 6)
    assert result.baseline == pytest.approx(1 / 6)
    assert result.n_flagged == 4


def test_homogeneity_compares_only_classes_meeting_minimum_size() -> None:
    real_train = pd.DataFrame(
        {
            "key": ["a"] * 4 + ["b"] * 2 + ["c"] + ["d"] * 2,
            "secret": ["x", "x", "y", "y", "z", "z", "n", "w", "w"],
            "amount": [1.0] * 9,
        }
    )
    synthetic = pd.DataFrame(
        {
            "key": ["a"] * 4 + ["b"] * 2 + ["c"] + ["d"] * 2,
            "secret": ["x", "x", "y", "y", "z", "z", "n", "w", "w"],
            "amount": [1.0] * 9,
        }
    )
    result = for_target(
        homogeneity(
            DatasetBundle(real_train, synthetic),
            make_schema(),
            make_profile(min_cell_size=2),
        ),
        "homogeneity_share",
        "secret",
    )

    assert result.value == pytest.approx(2 / 3)
    assert result.baseline == pytest.approx(2 / 3)
    assert result.n_records == 3
    assert result.n_flagged == 2


def test_continuous_target_is_not_assessed_for_categorical_cap() -> None:
    result = next(
        result
        for result in cap_disclosure(make_bundle(), make_schema(), make_profile())
        if result.name == "cap_accuracy" and result.status == "not_assessed"
    )

    assert result.status == "not_assessed"
    assert "categorical or ordinal" in result.reason


def test_numeric_tolerance_is_not_assessed_for_categorical_target() -> None:
    result = next(
        result
        for result in numeric_tolerance_recovery(
            make_bundle(), make_schema(), make_profile()
        )
        if result.name == "numeric_tolerance_recovery"
        and result.status == "not_assessed"
    )

    assert "only applies to numeric targets" in result.reason


def test_cap_result_is_suppressed_when_a_contributing_cell_is_small() -> None:
    result = next(
        result
        for result in cap_disclosure(
            make_bundle(), make_schema(), make_profile(min_cell_size=3)
        )
        if result.name == "cap_accuracy" and result.status == "suppressed"
    )

    assert result.status == "suppressed"
    assert result.value is None
    assert result.baseline is None


def test_ml_inference_gap_compares_synthetic_model_with_marginal_baseline() -> None:
    schema = Schema(
        variables=[
            Variable(name="key", vtype="categorical", role=["qi"]),
            Variable(name="target", vtype="categorical", role=["sensitive"]),
        ]
    )
    profile = make_profile(sensitive=("target",))
    synthetic = pd.DataFrame(
        {"key": ["a"] * 4 + ["b"] * 4, "target": ["A"] * 4 + ["B"] * 4}
    )
    real_train = synthetic.copy()
    real_test = pd.DataFrame(
        {"key": ["a"] * 4 + ["b"] * 4, "target": ["B"] * 4 + ["A"] * 4}
    )
    bundle = DatasetBundle(real_train, synthetic, real_test)

    result = for_target(
        ml_inference_gain(bundle, schema, profile, make_rng(7)),
        "ml_inference_gain",
        "target",
    )

    assert result.value == pytest.approx(1.0)
    assert result.baseline == pytest.approx(0.0)
    assert not result.experimental


def test_ml_inference_without_holdout_uses_experimental_cross_validation() -> None:
    schema = Schema(
        variables=[
            Variable(name="key", vtype="categorical", role=["qi"]),
            Variable(name="target", vtype="categorical", role=["sensitive"]),
        ]
    )
    profile = make_profile(sensitive=("target",))
    real_train = pd.DataFrame(
        {"key": ["a"] * 10 + ["b"] * 10, "target": ["A"] * 10 + ["B"] * 10}
    )
    bundle = DatasetBundle(real_train, real_train.copy())

    result = for_target(
        ml_inference_gain(bundle, schema, profile, make_rng(7)),
        "ml_inference_gain",
        "target",
    )

    assert result.value == pytest.approx(1.0)
    assert result.baseline == pytest.approx(0.5)
    assert result.experimental
    assert "cross-validation" in result.reason
