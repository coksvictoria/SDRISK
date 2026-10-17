"""Tests for public-data benchmark generators and aggregate detection summaries."""

from sdrisk.benchmarking import (
    generate_synthetic,
    load_public_data,
    summarize_detection_rates,
)
from sdrisk.provenance import make_rng


def _training_data():
    return load_public_data(seed=19)[0].real_train


def test_public_benchmark_data_is_bundled_and_train_holdout_are_disjoint() -> None:
    bundle, schema = load_public_data(seed=19)

    assert len(bundle.real_train) == 250
    assert len(bundle.real_test) > 0
    assert "radius_band" in schema.variable_names
    assert set(bundle.real_train["record_id"]).isdisjoint(bundle.real_test["record_id"])


def test_planted_generators_preserve_schema_and_create_fresh_synthetic_ids() -> None:
    train = _training_data()
    scenarios = (
        "independent",
        "bootstrap",
        "noisy_copy",
        "overfit",
        "attribute_leak",
        "localized_leak",
        "heavy_noise",
    )

    for scenario in scenarios:
        generated = generate_synthetic(
            train,
            30,
            scenario=scenario,
            rng=make_rng(11),
        )

        assert list(generated.columns) == list(train.columns)
        assert len(generated) == 30
        assert generated["record_id"].is_unique
        assert generated["record_id"].str.startswith("synthetic-").all()


def test_attribute_leak_generator_is_constant_within_configured_qi_classes() -> None:
    generated = generate_synthetic(
        _training_data(),
        100,
        scenario="attribute_leak",
        rng=make_rng(23),
    )

    assert generated.groupby(["radius_band", "texture_band"])["target"].nunique().max() == 1


def test_localized_generator_oversamples_a_subgroup() -> None:
    train = _training_data()
    generated = generate_synthetic(
        train,
        120,
        scenario="localized_leak",
        rng=make_rng(29),
    )
    affected_band = train["radius_band"].value_counts().idxmin()

    assert (
        (generated["radius_band"] == affected_band).mean()
        > (train["radius_band"] == affected_band).mean()
    )


def test_detection_rates_are_computed_from_hand_checkable_responses() -> None:
    summary = summarize_detection_rates(
        [
            {
                "scenario": "independent",
                "responses": {"exact_match_rate": False, "cluster_exposure_ratio": True},
            },
            {
                "scenario": "independent",
                "responses": {"exact_match_rate": False, "cluster_exposure_ratio": False},
            },
            {
                "scenario": "bootstrap",
                "responses": {"exact_match_rate": True, "cluster_exposure_ratio": True},
            },
            {
                "scenario": "bootstrap",
                "responses": {"exact_match_rate": True, "cluster_exposure_ratio": True},
            },
        ]
    )

    assert summary["independent"]["exact_match_rate"] == {
        "detections": 0,
        "runs": 2,
        "rate": 0.0,
    }
    assert summary["independent"]["cluster_exposure_ratio"]["rate"] == 0.5
    assert summary["bootstrap"]["exact_match_rate"]["rate"] == 1.0
