"""Tests for orchestration of the existing metric levels."""

import pandas as pd

from sdrisk.bundle import DatasetBundle
from sdrisk.policy import PolicyProfile
from sdrisk.runner import run_assessment
from sdrisk.schema import Schema, Variable


def test_runner_builds_all_levels_and_provenance_deterministically() -> None:
    train = pd.DataFrame(
        {
            "key": [f"k{i}" for i in range(12)],
            "x": [float(i) for i in range(12)],
            "secret": [f"s{i % 3}" for i in range(12)],
        }
    )
    synthetic = train.copy()
    test = train.assign(key=[f"t{i}" for i in range(12)])
    bundle = DatasetBundle(train, synthetic, test)
    schema = Schema(
        variables=[
            Variable(name="key", vtype="categorical", role=["qi"]),
            Variable(name="x", vtype="numeric", role=["qi"]),
            Variable(name="secret", vtype="categorical", role=["sensitive"]),
        ]
    )
    profile = PolicyProfile.model_validate(
        {
            "name": "runner-test",
            "release_tier": "in_lab",
            "output": {"min_cell_size": 2, "rounding_base": 1},
            "variables": {
                "qi_sets": [{"name": "core", "columns": ["key", "x"]}],
                "sensitive": ["secret"],
                "strata": [],
            },
            "identity": {"k_profile_thresholds": [2]},
            "row_risk": {"medium_min_flags": 1, "high_min_flags": 2},
            "distance": {
                "max_rows_exact": 100,
                "subsample_rows": 100,
                "near_match_threshold": 0.05,
            },
            "clusters": {"k": 2, "min_cluster_size": 2, "seeds": 2},
            "thresholds": {
                "exact_match_rate": {"review": 0.01, "fail": 0.1},
            },
            "policy": {"require_test_for_public_tier": True},
        }
    )

    result = run_assessment(bundle, schema, profile, seed=31)
    repeat = run_assessment(bundle, schema, profile, seed=31)

    assert result.table_results
    assert result.subset_results
    assert result.cluster_results
    assert result.row_store is not None
    assert result.provenance.config_hash == repeat.provenance.config_hash
    assert result.provenance.input_hashes == repeat.provenance.input_hashes
    assert result.provenance.seed == 31
    assert any(item.name == "mia_distance_auc" for item in result.table_results)
    assert all(item.status in ("pass", "review", "fail", "suppressed", "not_assessed")
               for item in result.decision.results)
