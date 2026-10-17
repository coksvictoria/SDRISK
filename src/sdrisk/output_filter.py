"""Disclosure-control filter for values leaving the package."""

from collections.abc import Iterable, Mapping
from dataclasses import dataclass, replace
from typing import Literal

from sdrisk.results import MetricResult

CountStatus = Literal["reported", "suppressed"]


@dataclass(frozen=True)
class FilteredCount:
    """A count cleared for export or explicitly suppressed."""

    value: int | None
    status: CountStatus
    reason: str = ""


class OutputFilter:
    """Apply configured cell suppression, rounding, and secondary suppression."""

    def __init__(self, *, min_cell_size: int, rounding_base: int) -> None:
        if (
            isinstance(min_cell_size, bool)
            or not isinstance(min_cell_size, int)
            or min_cell_size < 1
        ):
            raise ValueError("min_cell_size must be a positive integer")
        if (
            isinstance(rounding_base, bool)
            or not isinstance(rounding_base, int)
            or rounding_base < 1
        ):
            raise ValueError("rounding_base must be a positive integer")
        self.min_cell_size = min_cell_size
        self.rounding_base = rounding_base

    def filter_count(self, count: int) -> FilteredCount:
        """Suppress a small count or round it to the configured base."""
        if isinstance(count, bool) or not isinstance(count, int) or count < 0:
            raise ValueError("counts must be non-negative integers")
        if count < self.min_cell_size:
            return FilteredCount(
                value=None,
                status="suppressed",
                reason="count is below the configured minimum cell size",
            )
        rounded = ((count + self.rounding_base // 2) // self.rounding_base) * (
            self.rounding_base
        )
        if count > 0:
            rounded = max(rounded, self.rounding_base)
        return FilteredCount(value=rounded, status="reported")

    def filter_count_table(
        self, counts: Mapping[str, int], *, total_key: str | None = None
    ) -> dict[str, FilteredCount]:
        """Filter grouped counts and apply complementary secondary suppression."""
        if total_key is not None and total_key not in counts:
            raise ValueError("total_key must name a count in the table")
        if total_key is not None:
            component_total = sum(
                count for key, count in counts.items() if key != total_key
            )
            if counts[total_key] != component_total:
                raise ValueError("total must equal the sum of its component counts")

        filtered = {key: self.filter_count(count) for key, count in counts.items()}
        if total_key is None or filtered[total_key].status == "suppressed":
            return filtered

        children = [(key, count) for key, count in counts.items() if key != total_key]
        hidden_children = [
            key for key, _ in children if filtered[key].status == "suppressed"
        ]
        if len(hidden_children) == 1:
            visible_children = [
                (key, count)
                for key, count in children
                if filtered[key].status == "reported"
            ]
            if visible_children:
                key, _ = min(visible_children, key=lambda item: item[1])
                filtered[key] = FilteredCount(
                    value=None,
                    status="suppressed",
                    reason="secondary suppression prevents recovery from the total",
                )
            else:
                filtered[total_key] = FilteredCount(
                    value=None,
                    status="suppressed",
                    reason="secondary suppression prevents recovery from components",
                )
        return filtered

    def filter_metric_result(
        self,
        result: MetricResult,
        *,
        contributing_counts: Iterable[int] = (),
    ) -> MetricResult:
        """Filter values while retaining only caller-supplied generic group labels."""
        if result.status in ("suppressed", "not_assessed"):
            return replace(
                result,
                value=None,
                baseline=None,
                ci=None,
                n_records=None,
                n_flagged=None,
            )
        counts = list(contributing_counts)
        if result.n_records is not None:
            counts.append(result.n_records)
        if result.n_flagged is not None:
            counts.append(result.n_flagged)
        if any(count < 0 for count in counts):
            raise ValueError("contributing counts must be non-negative")
        if result.n_records is None or result.n_records < self.min_cell_size:
            return self._suppress_metric(result, "record count is below the configured minimum")
        if any(count < self.min_cell_size for count in counts):
            return self._suppress_metric(
                result, "a contributing count is below the configured minimum"
            )

        n_flagged = (
            self.filter_count(result.n_flagged).value
            if result.n_flagged is not None
            else None
        )
        return replace(
            result,
            n_records=self.filter_count(result.n_records).value,
            n_flagged=n_flagged,
        )

    @staticmethod
    def _suppress_metric(result: MetricResult, reason: str) -> MetricResult:
        return replace(
            result,
            value=None,
            baseline=None,
            ci=None,
            n_records=None,
            n_flagged=None,
            status="suppressed",
            reason=reason,
        )
