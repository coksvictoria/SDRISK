"""Shared metric protocol and registry."""

from collections.abc import Collection
from typing import Protocol


class Metric(Protocol):
    """Describe the registration metadata required for a metric."""

    name: str
    requires: frozenset[str]


class MetricRegistry:
    """Register metrics and retain their declared input requirements."""

    def __init__(self) -> None:
        self._metrics: dict[str, object] = {}
        self._requirements: dict[str, frozenset[str]] = {}

    def register(
        self, name: str, metric: object, *, requires: Collection[str]
    ) -> None:
        """Register a metric under a unique name."""
        if not name:
            raise ValueError("metric name must not be empty")
        if name in self._metrics:
            raise ValueError(f"metric {name!r} is already registered")
        required = frozenset(requires)
        allowed = {"real_train", "synthetic", "real_test"}
        if not required <= allowed:
            unknown = ", ".join(sorted(required - allowed))
            raise ValueError(f"unknown required inputs: {unknown}")
        self._metrics[name] = metric
        self._requirements[name] = required

    def get(self, name: str) -> object:
        """Return a registered metric or raise for an unknown name."""
        try:
            return self._metrics[name]
        except KeyError as error:
            raise KeyError(f"unknown metric {name!r}") from error

    def requires(self, name: str) -> frozenset[str]:
        """Return the required bundle inputs for a registered metric."""
        try:
            return self._requirements[name]
        except KeyError as error:
            raise KeyError(f"unknown metric {name!r}") from error
