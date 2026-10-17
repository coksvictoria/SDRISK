"""Metric result models and internal row-profile storage."""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Literal

if TYPE_CHECKING:
    from sdrisk.decision import AssessmentDecision
    from sdrisk.output_filter import FilteredCount, OutputFilter
    from sdrisk.policy import PolicyProfile
    from sdrisk.provenance import RunProvenance

Status = Literal["pass", "review", "fail", "suppressed", "not_assessed"]
Level = Literal["table", "subset", "cluster"]
RiskBand = Literal["low", "medium", "high"]


@dataclass(frozen=True)
class MetricResult:
    """Represent a metric outcome; group labels must be generic, never observed keys."""

    name: str
    level: Level
    value: float | None
    baseline: float | None
    ci: tuple[float, float] | None
    n_records: int | None
    n_flagged: int | None
    status: Status
    group: str | None = None
    reason: str = ""
    experimental: bool = False

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("metric name must not be empty")
        if self.n_records is not None and self.n_records < 0:
            raise ValueError("n_records must be non-negative")
        if self.n_flagged is not None and self.n_flagged < 0:
            raise ValueError("n_flagged must be non-negative")
        if (
            self.n_records is not None
            and self.n_flagged is not None
            and self.n_flagged > self.n_records
        ):
            raise ValueError("n_flagged must not exceed n_records")
        if self.status in ("suppressed", "not_assessed") and not self.reason:
            raise ValueError(f"{self.status} results require a reason")
        for label, value in (("value", self.value), ("baseline", self.baseline)):
            if value is not None and not math.isfinite(value):
                raise ValueError(f"{label} must be finite")
        if self.ci is not None:
            lower, upper = self.ci
            if not math.isfinite(lower) or not math.isfinite(upper) or lower > upper:
                raise ValueError("ci must contain finite values in ascending order")


@dataclass(frozen=True)
class RowStore:
    """Internal-only storage of synthetic-row profiles.

    No method exposes a row profile. Only aggregate band counts are returned,
    and those counts are routed through the output filter.
    """

    _profiles: list[Mapping[str, object]] = field(default_factory=list, repr=False)

    def add_profile(self, profile: Mapping[str, object]) -> None:
        band = profile.get("composite_band")
        if band not in ("low", "medium", "high"):
            raise ValueError("row profile requires a valid composite_band")
        self._profiles.append(dict(profile))

    def aggregate_band_counts(
        self, output_filter: OutputFilter
    ) -> dict[str, FilteredCount]:
        counts = {band: 0 for band in ("low", "medium", "high")}
        for profile in self._profiles:
            band = profile["composite_band"]
            if isinstance(band, str):
                counts[band] += 1
        counts["total"] = sum(counts.values())
        return output_filter.filter_count_table(counts, total_key="total")


@dataclass(frozen=True)
class RunResult:
    """Hold completed aggregate results and restricted internal row profiles."""

    table_results: tuple[MetricResult, ...]
    subset_results: tuple[MetricResult, ...]
    cluster_results: tuple[MetricResult, ...]
    row_store: RowStore | None = field(repr=False)
    row_status: Literal["assessed", "not_assessed"]
    provenance: RunProvenance
    decision: AssessmentDecision
    profile: PolicyProfile = field(repr=False)
    row_reason: str = ""
