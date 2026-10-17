"""Hand-computable tests for simplified attribute-guess attacks."""

import pandas as pd

from sdrisk.bundle import DatasetBundle
from sdrisk.metrics.attacks.linkability import linkability
from sdrisk.metrics.attacks.singling_out import singling_out
from sdrisk.policy import PolicyProfile
from sdrisk.provenance import make_rng
from sdrisk.schema import Schema, Variable


def make_schema() -> Schema:
    return Schema(
        variables=[
            Variable(name="a", vtype="categorical", role=["qi"]),
            Variable(name="b", vtype="categorical", role=["qi"]),
            Variable(name="record_id", vtype="categorical", role=["id"]),
        ]
    )


def make_profile(*, qi_columns: tuple[str, ...] = ("a", "b")) -> PolicyProfile:
    return PolicyProfile.model_validate(
        {
            "name": "attack-test",
            "release_tier": "in_lab",
            "output": {"min_cell_size": 1, "rounding_base": 1},
            "variables": {
                "qi_sets": [{"name": "pair", "columns": list(qi_columns)}],
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
            "thresholds": {"attack": {"review": 0.1, "fail": 0.5}},
            "policy": {"require_test_for_public_tier": True},
        }
    )


def make_bundle(*, with_test: bool = True) -> DatasetBundle:
    train = pd.DataFrame(
        {
            "a": ["A", "A", "B", "C"],
            "b": ["x", "z", "y", "w"],
            "record_id": ["tr1", "tr2", "tr3", "tr4"],
        }
    )
    synthetic = pd.DataFrame(
        {
            "a": ["A", "B", "Z"],
            "b": ["x", "y", "q"],
            "record_id": ["s1", "s2", "s3"],
        }
    )
    test = (
        pd.DataFrame(
            {
                "a": ["A", "D", "E"],
                "b": ["x", "v", "u"],
                "record_id": ["te1", "te2", "te3"],
            }
        )
        if with_test
        else None
    )
    return DatasetBundle(train, synthetic, test)


def test_singling_out_exact_key_attack_uses_holdout_control_baseline() -> None:
    result = singling_out(
        make_bundle(),
        make_schema(),
        make_profile(),
        make_rng(3),
    )[0]

    assert result.name == "singling_out_success"
    assert result.value == 2 / 3
    assert result.baseline == 1 / 3
    assert result.n_records == 3
    assert result.n_flagged == 2
    assert result.experimental


def test_linkability_uses_two_disjoint_qi_blocks_and_holdout_control() -> None:
    train = pd.DataFrame(
        {
            "a": ["A", "A", "B", "C"],
            "b": ["x", "y", "x", "z"],
            "record_id": ["tr1", "tr2", "tr3", "tr4"],
        }
    )
    synthetic = pd.DataFrame(
        {
            "a": ["A", "C", "D"],
            "b": ["x", "z", "q"],
            "record_id": ["s1", "s2", "s3"],
        }
    )
    test = pd.DataFrame(
        {
            "a": ["A", "C", "F"],
            "b": ["y", "z", "w"],
            "record_id": ["te1", "te2", "te3"],
        }
    )
    result = linkability(
        DatasetBundle(train, synthetic, test),
        make_schema(),
        make_profile(),
        make_rng(3),
    )[0]

    assert result.name == "linkability_success"
    assert result.value == 1 / 3
    assert result.baseline == 1 / 3
    assert result.n_flagged == 1
    assert result.experimental


def test_attack_success_matches_holdout_baseline_on_independent_sample() -> None:
    train = pd.DataFrame(
        {
            "a": ["A", "B", "C", "D"],
            "b": ["x", "y", "z", "w"],
            "record_id": ["tr1", "tr2", "tr3", "tr4"],
        }
    )
    synthetic = pd.DataFrame(
        {
            "a": ["A", "B", "C", "D"],
            "b": ["x", "y", "z", "w"],
            "record_id": ["s1", "s2", "s3", "s4"],
        }
    )
    test = pd.DataFrame(
        {
            "a": ["A", "B", "C", "D"],
            "b": ["x", "y", "z", "w"],
            "record_id": ["te1", "te2", "te3", "te4"],
        }
    )
    bundle = DatasetBundle(train, synthetic, test)

    singling = singling_out(bundle, make_schema(), make_profile(), make_rng(41))[0]
    links = linkability(bundle, make_schema(), make_profile(), make_rng(41))[0]

    assert singling.value == singling.baseline == 1.0
    assert links.value == links.baseline == 1.0


def test_singling_out_is_not_assessed_without_holdout_control() -> None:
    result = singling_out(
        make_bundle(with_test=False),
        make_schema(),
        make_profile(),
        make_rng(3),
    )[0]

    assert result.status == "not_assessed"
    assert "holdout" in result.reason


def test_linkability_is_not_assessed_when_qi_set_has_one_column() -> None:
    result = linkability(
        make_bundle(),
        make_schema(),
        make_profile(qi_columns=("a",)),
        make_rng(3),
    )[0]

    assert result.status == "not_assessed"
    assert "two" in result.reason
