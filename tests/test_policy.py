"""Tests for policy profile parsing and validation."""

from pathlib import Path

import pytest
from pydantic import ValidationError

from sdrisk.policy import PolicyProfile, load_policy_profile


def valid_profile() -> dict[str, object]:
    return {
        "name": "test",
        "release_tier": "public",
        "output": {"min_cell_size": 6, "rounding_base": 3},
        "variables": {
            "qi_sets": [{"name": "core", "columns": ["age"]}],
            "sensitive": ["income"],
            "strata": ["region"],
        },
        "identity": {"k_profile_thresholds": [2, 5, 10]},
        "distance": {
            "max_rows_exact": 1000,
            "subsample_rows": 100,
            "near_match_threshold": 0.05,
        },
        "clusters": {"k": 4, "min_cluster_size": 10, "seeds": 2},
        "thresholds": {"risk": {"review": 0.1, "fail": 0.2}},
        "policy": {"require_test_for_public_tier": True},
    }


def test_policy_profile_parses_required_configuration() -> None:
    profile = PolicyProfile.model_validate(valid_profile())

    assert profile.name == "test"
    assert profile.output.min_cell_size == 6
    assert profile.thresholds["risk"].fail == 0.2


def test_policy_rejects_invalid_threshold_order() -> None:
    data = valid_profile()
    data["thresholds"] = {"risk": {"review": 0.3, "fail": 0.2}}

    with pytest.raises(ValidationError, match="review"):
        PolicyProfile.model_validate(data)


def test_policy_rejects_unknown_fields() -> None:
    data = valid_profile()
    data["surprise"] = True

    with pytest.raises(ValidationError, match="surprise"):
        PolicyProfile.model_validate(data)


def test_policy_requires_positive_unique_k_profile_cutoffs() -> None:
    data = valid_profile()
    data["identity"] = {"k_profile_thresholds": [2, 2]}

    with pytest.raises(ValidationError, match="unique"):
        PolicyProfile.model_validate(data)


def test_policy_validates_configured_row_risk_band_thresholds() -> None:
    data = valid_profile()
    data["row_risk"] = {"medium_min_flags": 1, "high_min_flags": 2}

    profile = PolicyProfile.model_validate(data)

    assert profile.row_risk is not None
    assert profile.row_risk.medium_min_flags == 1
    assert profile.row_risk.high_min_flags == 2


def test_policy_rejects_overlapping_row_risk_thresholds() -> None:
    data = valid_profile()
    data["row_risk"] = {"medium_min_flags": 2, "high_min_flags": 2}

    with pytest.raises(ValidationError, match="high_min_flags"):
        PolicyProfile.model_validate(data)


def test_policy_rejects_row_risk_thresholds_above_available_flags() -> None:
    data = valid_profile()
    data["row_risk"] = {"medium_min_flags": 1, "high_min_flags": 5}

    with pytest.raises(ValidationError, match="cannot exceed four"):
        PolicyProfile.model_validate(data)


def test_load_policy_profile_reads_yaml(tmp_path: Path) -> None:
    import yaml

    path = tmp_path / "policy.yaml"
    path.write_text(yaml.safe_dump(valid_profile()), encoding="utf-8")

    assert load_policy_profile(path).name == "test"


def test_load_policy_profile_reports_invalid_yaml(tmp_path: Path) -> None:
    path = tmp_path / "policy.yaml"
    path.write_text("name: [unclosed", encoding="utf-8")

    with pytest.raises(ValueError, match="YAML"):
        load_policy_profile(path)


def test_load_policy_profile_reports_missing_file(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        load_policy_profile(tmp_path / "missing.yaml")
