"""User-defined and automatic subset-level metric evaluation."""

from collections.abc import Sequence
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from sdrisk.bundle import DatasetBundle
from sdrisk.levels.common import lift_results, tier1_results
from sdrisk.metrics.identity.common import (
    filter_result,
    not_assessed,
    output_filter,
    qi_sets,
    row_keys,
    validate_bundle_columns,
)
from sdrisk.output_filter import FilteredCount
from sdrisk.policy import PolicyProfile
from sdrisk.results import MetricResult
from sdrisk.schema import Schema


@dataclass(frozen=True)
class _Subset:
    """Internal subset frames; labels never contain observed category values."""

    label: str
    train: pd.DataFrame = field(repr=False)
    synthetic: pd.DataFrame = field(repr=False)
    test: pd.DataFrame | None = field(repr=False)


@dataclass(frozen=True)
class SubsetAnalysis:
    """Filtered subset results and multiple-comparison metadata."""

    metrics: tuple[MetricResult, ...]
    n_subsets_tested: FilteredCount
    n_subsets_suppressed: FilteredCount
    correction_method: str
    multiple_comparison_caution: str


def analyze_subsets(
    bundle: DatasetBundle,
    schema: Schema,
    profile: PolicyProfile,
    rng: np.random.Generator,
) -> SubsetAnalysis:
    """Evaluate table-level metrics for configured strata and exact QI classes.

    Subset group labels contain only the stratum or QI-set name, never observed
    category values or QI keys. Small partitions return an explicit suppressed
    guard result. No multiple-comparison correction is applied.
    """
    if not isinstance(rng, np.random.Generator):
        raise TypeError("rng must be a numpy.random.Generator")
    if bundle.real_train.empty or bundle.synthetic.empty:
        return _analysis_not_assessed(profile, "real_train and synthetic must both contain records")

    subsets = _build_subsets(bundle, schema, profile)
    results: list[MetricResult] = []
    n_subsets_tested = 0
    n_subsets_suppressed = 0
    count_filter = output_filter(profile)
    minimum = profile.output.min_cell_size
    for subset in subsets:
        partition_counts = [len(subset.train), len(subset.synthetic)]
        if subset.test is not None:
            partition_counts.append(len(subset.test))
        if any(count < minimum for count in partition_counts):
            n_subsets_suppressed += 1
            results.append(
                filter_result(
                    MetricResult(
                        name="subset_min_n_guard",
                        level="subset",
                        group=subset.label,
                        value=None,
                        baseline=None,
                        ci=None,
                        n_records=None,
                        n_flagged=None,
                        status="suppressed",
                        reason="subset partition is below the configured minimum cell size",
                    ),
                    profile,
                )
            )
            continue

        n_subsets_tested += 1
        selected_bundle = DatasetBundle(
            subset.train,
            subset.synthetic,
            subset.test,
        )
        results.extend(
            lift_results(
                tier1_results(selected_bundle, schema, profile, rng),
                profile,
                level="subset",
                group=subset.label,
                contributing_counts=tuple(partition_counts),
            )
        )

    return SubsetAnalysis(
        metrics=tuple(results),
        n_subsets_tested=count_filter.filter_count(n_subsets_tested),
        n_subsets_suppressed=count_filter.filter_count(n_subsets_suppressed),
        correction_method="none",
        multiple_comparison_caution=(
            "Subset results are not adjusted for multiple comparisons; "
            "interpret isolated high-risk results cautiously."
        ),
    )


def _build_subsets(bundle: DatasetBundle, schema: Schema, profile: PolicyProfile) -> list[_Subset]:
    """Construct partitions from the union of observed input keys."""
    schema_variables = {variable.name: variable for variable in schema.variables}
    for name in profile.variables.strata:
        variable = schema_variables.get(name)
        if variable is None:
            raise ValueError(f"stratum column {name!r} is missing from the schema")
        if "stratum" not in variable.role:
            raise ValueError(f"stratum column {name!r} must have the stratum schema role")
        if {"id", "ignore"} & set(variable.role):
            raise ValueError(f"stratum column {name!r} cannot be an identifier or ignored")
        if name not in bundle.real_train.columns:
            raise ValueError(f"bundle is missing stratum column {name!r}")

    subsets: list[_Subset] = []
    for name in profile.variables.strata:
        subsets.extend(
            _partitions_for_values(
                bundle,
                (name,),
                label=f"stratum:{name}",
            )
        )
    for qi_set in qi_sets(schema, profile):
        validate_bundle_columns(bundle, qi_set.columns)
        subsets.extend(
            _partitions_for_values(
                bundle,
                tuple(qi_set.columns),
                label=f"qi:{qi_set.name}",
            )
        )
    return subsets


def _partitions_for_values(
    bundle: DatasetBundle,
    columns: Sequence[str],
    *,
    label: str,
) -> list[_Subset]:
    """Split all supplied frames by each key observed in any input."""
    datasets: tuple[pd.DataFrame, ...] = (
        (bundle.real_train, bundle.synthetic)
        if bundle.real_test is None
        else (bundle.real_train, bundle.synthetic, bundle.real_test)
    )
    distinct_keys = tuple(
        dict.fromkeys(key for data in datasets for key in row_keys(data, columns))
    )
    partitions: list[_Subset] = []
    for key in distinct_keys:
        selected: list[pd.DataFrame] = []
        for data in datasets:
            keys = row_keys(data, columns)
            mask = np.fromiter(
                (candidate == key for candidate in keys),
                dtype=np.bool_,
                count=len(keys),
            )
            selected.append(data.iloc[np.flatnonzero(mask)].reset_index(drop=True))
        test = selected[2] if len(selected) == 3 else None
        partitions.append(
            _Subset(
                label=label,
                train=selected[0],
                synthetic=selected[1],
                test=test,
            )
        )
    return partitions


def _analysis_not_assessed(profile: PolicyProfile, reason: str) -> SubsetAnalysis:
    result = filter_result(
        not_assessed("subset_analysis", reason, level="subset"),
        profile,
    )
    return SubsetAnalysis(
        metrics=(result,),
        n_subsets_tested=output_filter(profile).filter_count(0),
        n_subsets_suppressed=output_filter(profile).filter_count(0),
        correction_method="none",
        multiple_comparison_caution=reason,
    )
