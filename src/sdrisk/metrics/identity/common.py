"""Shared helpers for table-level identity metrics."""

import math
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence

import numpy as np
import pandas as pd

from sdrisk.bundle import DatasetBundle
from sdrisk.output_filter import OutputFilter
from sdrisk.policy import PolicyProfile, QISet
from sdrisk.prep.values import is_missing, numeric_values
from sdrisk.results import Level, MetricResult
from sdrisk.schema import Schema, Variable

_MISSING = object()
_WILSON_Z = 1.959963984540054
_DECISION_PENDING = "Decision classification is deferred to the M7 decision layer."


def qi_sets(schema: Schema, profile: PolicyProfile) -> tuple[QISet, ...]:
    """Validate profile QI sets against schema roles and return them in order."""
    variables = {variable.name: variable for variable in schema.variables}
    for qi_set in profile.variables.qi_sets:
        for column in qi_set.columns:
            variable = variables.get(column)
            if variable is None:
                raise ValueError(f"QI column {column!r} is missing from the schema")
            if {"id", "ignore"} & set(variable.role):
                raise ValueError(f"QI column {column!r} cannot be an identifier or ignored")
            if "qi" not in variable.role:
                raise ValueError(f"QI column {column!r} must have the qi schema role")
    return tuple(profile.variables.qi_sets)


def output_filter(profile: PolicyProfile) -> OutputFilter:
    """Build the disclosure filter from policy configuration."""
    return OutputFilter(
        min_cell_size=profile.output.min_cell_size,
        rounding_base=profile.output.rounding_base,
    )


def not_assessed(
    name: str,
    reason: str,
    *,
    group: str | None = None,
    level: Level = "table",
) -> MetricResult:
    """Create an explicit not-assessed metric result."""
    return MetricResult(
        name=name,
        level=level,
        group=group,
        value=None,
        baseline=None,
        ci=None,
        n_records=None,
        n_flagged=None,
        status="not_assessed",
        reason=reason,
    )


def assessed_result(
    name: str,
    value: float,
    baseline: float,
    ci: tuple[float, float] | None,
    n_records: int,
    n_flagged: int | None,
    *,
    group: str | None = None,
    experimental: bool = False,
    reason: str = _DECISION_PENDING,
) -> MetricResult:
    """Create an assessed but not-yet-adjudicated result."""
    return MetricResult(
        name=name,
        level="table",
        group=group,
        value=value,
        baseline=baseline,
        ci=ci,
        n_records=n_records,
        n_flagged=n_flagged,
        status="review",
        reason=reason,
        experimental=experimental,
    )


def filter_result(
    result: MetricResult,
    profile: PolicyProfile,
    *,
    contributing_counts: Sequence[int] = (),
) -> MetricResult:
    """Pass a metric result through the configured output filter."""
    return output_filter(profile).filter_metric_result(
        result, contributing_counts=contributing_counts
    )


def row_keys(data: pd.DataFrame, columns: Sequence[str]) -> list[tuple[object, ...]]:
    """Return hashable quasi-identifier keys with missing values normalized."""
    selected = data.loc[:, list(columns)]
    keys: list[tuple[object, ...]] = []
    for values in selected.itertuples(index=False, name=None):
        key: list[object] = []
        for value in values:
            if is_missing(value):
                key.append(_MISSING)
            else:
                try:
                    hash(value)
                except TypeError as error:
                    raise ValueError("QI values must be scalar and hashable") from error
                key.append(value)
        keys.append(tuple(key))
    return keys


def sensitive_variables(
    schema: Schema, profile: PolicyProfile, qi_set: QISet
) -> tuple[Variable, ...]:
    """Validate configured sensitive targets against the schema and selected QIs."""
    variables = {variable.name: variable for variable in schema.variables}
    targets: list[Variable] = []
    for name in profile.variables.sensitive:
        variable = variables.get(name)
        if variable is None:
            raise ValueError(f"sensitive column {name!r} is missing from the schema")
        if {"id", "ignore"} & set(variable.role):
            raise ValueError(f"sensitive column {name!r} cannot be an identifier or ignored")
        if "sensitive" not in variable.role:
            raise ValueError(f"sensitive column {name!r} must have the sensitive schema role")
        if name in qi_set.columns:
            raise ValueError(
                f"sensitive column {name!r} cannot also be a QI in {qi_set.name!r}"
            )
        targets.append(variable)
    return tuple(targets)


def validate_bundle_columns(bundle: DatasetBundle, columns: Sequence[str]) -> None:
    """Require selected schema columns to exist in the aligned input frames."""
    missing = [name for name in columns if name not in bundle.real_train.columns]
    if missing:
        raise ValueError(f"bundle is missing selected columns: {', '.join(missing)}")


def grouped_target_counts(
    data: pd.DataFrame, qi_columns: Sequence[str], target: str
) -> dict[tuple[object, ...], Counter[tuple[object, ...]]]:
    """Count target values within exact QI classes, including missing values."""
    groups: dict[tuple[object, ...], Counter[tuple[object, ...]]] = defaultdict(Counter)
    for qi_key, target_key in zip(
        row_keys(data, qi_columns), row_keys(data, (target,)), strict=True
    ):
        groups[qi_key][target_key] += 1
    return dict(groups)


def contributing_cell_counts(
    original: Mapping[tuple[object, ...], Counter[tuple[object, ...]]],
    synthetic: Mapping[tuple[object, ...], Counter[tuple[object, ...]]],
    groups: Sequence[tuple[object, ...]],
) -> list[int]:
    """Return nonzero class and target-cell counts used by an aggregate metric."""
    counts: list[int] = []
    for mapping in (original, synthetic):
        for key in groups:
            cells = mapping.get(key)
            if cells:
                counts.append(sum(cells.values()))
                counts.extend(cells.values())
    return counts


def target_keys(data: pd.DataFrame, target: str) -> list[tuple[object, ...]]:
    """Return normalized, hashable target labels."""
    return row_keys(data, (target,))


def key_counts(data: pd.DataFrame, columns: Sequence[str]) -> Counter[tuple[object, ...]]:
    """Count exact quasi-identifier equivalence classes."""
    return Counter(row_keys(data, columns))


def matching_rows(
    query: pd.DataFrame,
    candidates: pd.DataFrame,
    columns: Sequence[str],
    schema: Schema,
) -> np.ndarray:
    """Flag query rows with an exact or configured-tolerance candidate match."""
    selected_columns = tuple(columns)
    variables = {variable.name: variable for variable in schema.variables}
    tolerance_variables = []
    for column in selected_columns:
        variable = variables[column]
        tolerance = variable.tolerance
        if tolerance is not None and tolerance > 0:
            tolerance_variables.append(variable)
    exact_columns = tuple(
        column for column in selected_columns if column not in {v.name for v in tolerance_variables}
    )
    candidate_buckets: dict[tuple[object, ...], list[int]] = defaultdict(list)
    for position, key in enumerate(row_keys(candidates, exact_columns)):
        candidate_buckets[key].append(position)

    if not tolerance_variables:
        candidate_keys = set(candidate_buckets)
        return np.fromiter(
            (key in candidate_keys for key in row_keys(query, exact_columns)),
            dtype=np.bool_,
            count=len(query),
        )

    candidate_values = {
        variable.name: numeric_values(candidates[variable.name], variable)
        for variable in tolerance_variables
    }
    query_values = {
        variable.name: numeric_values(query[variable.name], variable)
        for variable in tolerance_variables
    }
    matched = np.zeros(len(query), dtype=np.bool_)
    for query_position, key in enumerate(row_keys(query, exact_columns)):
        positions = candidate_buckets.get(key)
        if not positions:
            continue
        candidates_match = np.ones(len(positions), dtype=np.bool_)
        for variable in tolerance_variables:
            tolerance = variable.tolerance
            if tolerance is None or tolerance <= 0:
                continue
            query_value = query_values[variable.name][query_position]
            candidate_values_for_key = candidate_values[variable.name][positions]
            query_missing = not math.isfinite(float(query_value))
            candidate_missing = ~np.isfinite(candidate_values_for_key)
            if query_missing:
                within_tolerance = candidate_missing
            else:
                within_tolerance = (
                    ~candidate_missing
                    & (np.abs(candidate_values_for_key - query_value) <= tolerance)
                )
            candidates_match &= within_tolerance
            if not candidates_match.any():
                break
        matched[query_position] = bool(candidates_match.any())
    return matched


def wilson_interval(
    successes: int, total: int, *, scale: float = 1.0
) -> tuple[float, float] | None:
    """Return a 95% Wilson score interval for a proportion."""
    if total <= 0:
        return None
    proportion = successes / total
    z_squared = _WILSON_Z**2
    denominator = 1 + z_squared / total
    center = (proportion + z_squared / (2 * total)) / denominator
    half_width = (
        _WILSON_Z
        * math.sqrt(
            proportion * (1 - proportion) / total + z_squared / (4 * total**2)
        )
        / denominator
    )
    return ((center - half_width) * scale, (center + half_width) * scale)
