"""YAML policy profile models and loading."""

from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


class OutputPolicy(BaseModel):
    """Disclosure-control settings for exported aggregates."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    min_cell_size: int = Field(ge=1)
    rounding_base: int = Field(ge=1)


class QISet(BaseModel):
    """A named set of quasi-identifiers."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    name: str = Field(min_length=1)
    columns: list[str] = Field(min_length=1)

    @field_validator("columns")
    @classmethod
    def columns_are_unique(cls, columns: list[str]) -> list[str]:
        if len(columns) != len(set(columns)):
            raise ValueError("QI set columns must be unique")
        return columns


class VariablePolicy(BaseModel):
    """Profile-declared quasi-identifiers, sensitive variables, and strata."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    qi_sets: list[QISet] = Field(min_length=1)
    sensitive: list[str]
    strata: list[str]

    @model_validator(mode="after")
    def qi_set_names_are_unique(self) -> "VariablePolicy":
        names = [qi_set.name for qi_set in self.qi_sets]
        if len(names) != len(set(names)):
            raise ValueError("QI set names must be unique")
        for label, values in (("sensitive", self.sensitive), ("strata", self.strata)):
            if any(not value for value in values) or len(values) != len(set(values)):
                raise ValueError(f"{label} names must be non-empty and unique")
        return self


class DistancePolicy(BaseModel):
    """Distance-engine limits and matching tolerance."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    max_rows_exact: int = Field(ge=1)
    subsample_rows: int = Field(ge=1)
    near_match_threshold: float = Field(ge=0, allow_inf_nan=False)


class ClusterPolicy(BaseModel):
    """Cluster-analysis configuration."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    k: int = Field(ge=1)
    min_cluster_size: int = Field(ge=1)
    seeds: int = Field(ge=1)


class IdentityPolicy(BaseModel):
    """Identity-metric cutoffs that control k-profile reporting."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    k_profile_thresholds: list[int] = Field(min_length=1)

    @field_validator("k_profile_thresholds")
    @classmethod
    def thresholds_are_positive_and_unique(cls, values: list[int]) -> list[int]:
        if any(isinstance(value, bool) or value < 1 for value in values):
            raise ValueError("k-profile thresholds must be positive integers")
        if len(values) != len(set(values)):
            raise ValueError("k-profile thresholds must be unique")
        return values


class RowRiskPolicy(BaseModel):
    """Profile-controlled thresholds for internal composite row bands."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    medium_min_flags: int = Field(ge=1)
    high_min_flags: int = Field(ge=1)

    @model_validator(mode="after")
    def high_band_requires_more_flags(self) -> "RowRiskPolicy":
        if self.medium_min_flags > 4 or self.high_min_flags > 4:
            raise ValueError("row-risk flag thresholds cannot exceed four available flags")
        if self.high_min_flags <= self.medium_min_flags:
            raise ValueError("high_min_flags must exceed medium_min_flags")
        return self


class Threshold(BaseModel):
    """Review and fail thresholds for a metric."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    review: float = Field(ge=0, allow_inf_nan=False)
    fail: float = Field(ge=0, allow_inf_nan=False)

    @model_validator(mode="after")
    def review_does_not_exceed_fail(self) -> "Threshold":
        if self.review > self.fail:
            raise ValueError("review threshold must not exceed fail threshold")
        return self


class PolicyOptions(BaseModel):
    """Additional release-policy decisions."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    require_test_for_public_tier: bool


class PolicyProfile(BaseModel):
    """Validated analysis and disclosure-control policy."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    name: str = Field(min_length=1)
    release_tier: Literal["in_lab", "licensed", "public"]
    output: OutputPolicy
    variables: VariablePolicy
    identity: IdentityPolicy
    row_risk: RowRiskPolicy | None = None
    distance: DistancePolicy
    clusters: ClusterPolicy
    thresholds: dict[str, Threshold] = Field(min_length=1)
    policy: PolicyOptions


def load_policy_profile(path: str | Path) -> PolicyProfile:
    """Load and validate a YAML profile without evaluating YAML tags."""
    profile_path = Path(path)
    with profile_path.open("r", encoding="utf-8") as stream:
        try:
            data = yaml.safe_load(stream)
        except yaml.YAMLError as error:
            raise ValueError(f"Invalid policy YAML in {profile_path}: {error}") from error
    return PolicyProfile.model_validate(data)
