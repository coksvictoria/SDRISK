"""Tests for metric applicability and registry behavior."""

import pandas as pd
import pytest

from sdrisk.applicability import check_applicability
from sdrisk.bundle import DatasetBundle
from sdrisk.metrics.base import MetricRegistry


def test_missing_holdout_returns_not_assessed_result() -> None:
    bundle = DatasetBundle(
        real_train=pd.DataFrame({"x": [1]}),
        synthetic=pd.DataFrame({"x": [2]}),
    )

    result = check_applicability("mia", {"real_train", "synthetic", "real_test"}, bundle)

    assert result.status == "not_assessed"
    assert result.reason == "missing required input: real_test"


def test_registered_metric_requirements_are_queryable() -> None:
    registry = MetricRegistry()
    metric = object()
    registry.register("exact_match", metric, requires={"real_train", "synthetic"})

    assert registry.get("exact_match") is metric
    assert registry.requires("exact_match") == frozenset({"real_train", "synthetic"})


def test_registry_rejects_duplicate_and_unknown_metrics() -> None:
    registry = MetricRegistry()
    registry.register("exact_match", object(), requires={"real_train"})

    with pytest.raises(ValueError, match="already registered"):
        registry.register("exact_match", object(), requires={"synthetic"})
    with pytest.raises(KeyError, match="unknown"):
        registry.get("unknown")
