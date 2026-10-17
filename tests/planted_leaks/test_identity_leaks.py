"""Generated planted-copy checks for table-level identity metrics."""

import pandas as pd

from sdrisk.bundle import DatasetBundle
from sdrisk.metrics.identity.dcr import dcr_overfitting, dcr_vs_random
from sdrisk.metrics.identity.exact_match import exact_match_rate, unique_exact_match_rate
from sdrisk.metrics.identity.repu import replicated_uniques
from sdrisk.policy import PolicyProfile
from sdrisk.provenance import make_rng
from sdrisk.schema import Schema, Variable


def _profile() -> PolicyProfile:
    return PolicyProfile.model_validate(
        {
            "name": "planted-copy-test",
            "release_tier": "in_lab",
            "output": {"min_cell_size": 1, "rounding_base": 1},
            "variables": {
                "qi_sets": [{"name": "core", "columns": ["key", "x"]}],
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
            "thresholds": {
                "exact_match_rate": {"review": 0.1, "fail": 0.5},
                "repu_pct": {"review": 0.5, "fail": 2.0},
                "dcr_overfit_share": {"review": 0.55, "fail": 0.65},
            },
            "policy": {"require_test_for_public_tier": True},
        }
    )


def _schema() -> Schema:
    return Schema(
        variables=[
            Variable(name="key", vtype="categorical", role=["qi"]),
            Variable(name="x", vtype="numeric", role=["qi"], tolerance=0.0),
            Variable(name="record_id", vtype="categorical", role=["id"]),
        ]
    )


def _bundle(copied_unique_rows: int) -> DatasetBundle:
    train = pd.DataFrame(
        {
            "key": ["common", "common", *(f"r{index}" for index in range(2, 20))],
            "x": [0.0, 0.0, *(float(index) for index in range(2, 20))],
            "record_id": [f"train-{index}" for index in range(20)],
        }
    )
    holdout = pd.DataFrame(
        {
            "key": ["common", *(f"t{index}" for index in range(19))],
            "x": [0.0, *(float(20 + index) for index in range(19))],
            "record_id": [f"test-{index}" for index in range(20)],
        }
    )
    copied = train.iloc[2 : 2 + copied_unique_rows].copy()
    novel_count = 19 - copied_unique_rows
    novel = pd.DataFrame(
        {
            "key": [f"new-{index}" for index in range(novel_count)],
            "x": [float(20 + index) + 0.1 for index in range(novel_count)],
            "record_id": [f"synthetic-new-{index}" for index in range(novel_count)],
        }
    )
    shared_control = pd.DataFrame(
        {"key": ["common"], "x": [0.0], "record_id": ["synthetic-control"]}
    )
    synthetic = pd.concat((shared_control, copied, novel), ignore_index=True)
    synthetic["record_id"] = [f"synthetic-{index}" for index in range(len(synthetic))]
    return DatasetBundle(train, synthetic, holdout)


def test_identity_metrics_respond_monotonically_to_planted_copies() -> None:
    schema = _schema()
    profile = _profile()
    copy_counts = [1, 5, 10, 18]
    metric_values: dict[str, list[float]] = {
        "exact": [],
        "uem": [],
        "repu": [],
        "dcr": [],
        "dcr_vs_random": [],
    }

    for copied_count in copy_counts:
        bundle = _bundle(copied_count)
        metric_values["exact"].append(
            exact_match_rate(bundle, schema, profile)[0].value or 0.0
        )
        metric_values["uem"].append(
            unique_exact_match_rate(bundle, schema, profile)[0].value or 0.0
        )
        metric_values["repu"].append(
            replicated_uniques(bundle, schema, profile)[0].value or 0.0
        )
        metric_values["dcr"].append(
            dcr_overfitting(bundle, schema, profile)[0].value or 0.0
        )
        metric_values["dcr_vs_random"].append(
            dcr_vs_random(bundle, schema, profile, make_rng(2026))[0].value or 0.0
        )

    assert metric_values["exact"] == sorted(metric_values["exact"])
    assert metric_values["uem"] == sorted(metric_values["uem"])
    assert metric_values["repu"] == sorted(metric_values["repu"])
    assert metric_values["dcr"] == sorted(metric_values["dcr"])
    assert metric_values["dcr_vs_random"] == sorted(
        metric_values["dcr_vs_random"], reverse=True
    )


def test_independent_control_is_not_reported_as_a_copying_signal() -> None:
    bundle = _bundle(copied_unique_rows=0)
    synthetic = bundle.synthetic.iloc[1:].copy()
    synthetic["key"] = [f"independent-{index}" for index in range(len(synthetic))]
    independent = DatasetBundle(bundle.real_train, synthetic, bundle.real_test)
    schema = _schema()
    profile = _profile()

    exact = exact_match_rate(independent, schema, profile)[0]
    uem = unique_exact_match_rate(independent, schema, profile)[0]
    repu = replicated_uniques(independent, schema, profile)[0]
    dcr = dcr_overfitting(independent, schema, profile)[0]
    dcr_random = dcr_vs_random(independent, schema, profile, make_rng(2026))[0]

    assert exact.status == "suppressed"
    assert uem.status == "suppressed"
    assert repu.status == "suppressed"
    assert dcr.status == "suppressed"
    assert dcr_random.value is not None
    assert dcr_random.value >= dcr_random.baseline
