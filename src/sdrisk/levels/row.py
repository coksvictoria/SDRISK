"""Internal per-synthetic-row risk profiles."""

from collections import defaultdict
from dataclasses import asdict, dataclass

import numpy as np

from sdrisk.bundle import DatasetBundle
from sdrisk.distance.gower import GowerDistance
from sdrisk.distance.nn_search import nearest_neighbor_distances
from sdrisk.levels.cluster import ClusterAnalysis
from sdrisk.metrics.identity.common import (
    key_counts,
    matching_rows,
    qi_sets,
    row_keys,
    sensitive_variables,
    validate_bundle_columns,
)
from sdrisk.policy import PolicyProfile
from sdrisk.results import RiskBand, RowStore
from sdrisk.schema import Schema


@dataclass(frozen=True)
class RowProfile:
    """Internal-only risk indicators for one synthetic row.

    `dcr_percentile` is the empirical lower-is-closer percentile among
    leave-one-out nearest-training distances. `local_density_ratio` is the
    nearest-holdout distance divided by nearest-training distance; values above
    one indicate the synthetic row is closer to training data. These values
    remain internal and never leave the RowStore.
    """

    dcr_train: float | None
    dcr_percentile: float | None
    exact_match: bool
    near_match: bool
    replicated_unique: bool
    k_syn: tuple[tuple[str, int], ...]
    k_real: tuple[tuple[str, int], ...]
    attribute_disclosure: bool
    local_density_ratio: float | None
    cluster_id: str | None
    cluster_exposure_ratio: float | None
    composite_band: RiskBand


def build_row_store(
    bundle: DatasetBundle,
    schema: Schema,
    profile: PolicyProfile,
    *,
    cluster_analysis: ClusterAnalysis | None = None,
) -> RowStore:
    """Build internal synthetic-row profiles without exposing row-level data.

    Requires at least two training rows for leave-one-out reference distances.
    Exact-match, replicated-unique, k-profile, and attribute flags are computed
    over configured QI sets. Composite bands count four flags: exact match,
    near match, replicated unique, and correct sensitive-attribute inference.
    Their cutoffs come exclusively from `profile.row_risk`.

    The optional holdout proxy and cluster fields are populated only when
    available. The function returns only the restricted RowStore abstraction;
    callers must use its output-filtered aggregate method for export.
    """
    if profile.row_risk is None:
        raise ValueError("row-risk policy must be configured to build row profiles")
    if len(bundle.real_train) < 2:
        raise ValueError("row profiles require at least two real training rows")
    if bundle.synthetic.empty:
        raise ValueError("row profiles require at least one synthetic row")
    if bundle.real_test is not None and bundle.real_test.empty:
        raise ValueError("row profiles require a non-empty supplied real_test")

    configured_qis = qi_sets(schema, profile)
    columns = tuple(dict.fromkeys(column for qi_set in configured_qis for column in qi_set.columns))
    targets = tuple(
        dict.fromkeys(
            target.name
            for qi_set in configured_qis
            for target in sensitive_variables(schema, profile, qi_set)
        )
    )
    validate_bundle_columns(bundle, (*columns, *targets))

    distance = GowerDistance(schema, columns=columns).fit(bundle.real_train)
    train_distances = nearest_neighbor_distances(
        bundle.real_train,
        bundle.real_train,
        distance,
        exclude_self=True,
    )
    synthetic_distances = nearest_neighbor_distances(
        bundle.synthetic,
        bundle.real_train,
        distance,
    )
    test_distances = (
        nearest_neighbor_distances(bundle.synthetic, bundle.real_test, distance)
        if bundle.real_test is not None
        else None
    )
    reference_distribution = np.sort(train_distances[np.isfinite(train_distances)])

    exact_matches = np.zeros(len(bundle.synthetic), dtype=np.bool_)
    replicated_uniques = np.zeros(len(bundle.synthetic), dtype=np.bool_)
    k_syn: list[list[int]] = [[] for _ in range(len(bundle.synthetic))]
    k_real: list[list[int]] = [[] for _ in range(len(bundle.synthetic))]
    attribute_disclosures = np.zeros(len(bundle.synthetic), dtype=np.bool_)

    for qi_set in configured_qis:
        exact_matches |= matching_rows(bundle.synthetic, bundle.real_train, qi_set.columns, schema)
        train_key_counts = key_counts(bundle.real_train, qi_set.columns)
        synthetic_key_counts = key_counts(bundle.synthetic, qi_set.columns)
        training_keys = row_keys(bundle.real_train, qi_set.columns)
        synthetic_keys = row_keys(bundle.synthetic, qi_set.columns)

        for position, key in enumerate(synthetic_keys):
            k_syn[position].append(synthetic_key_counts[key])
            k_real[position].append(train_key_counts[key])
            if train_key_counts[key] == 1 and synthetic_key_counts[key] == 1:
                replicated_uniques[position] = True

        for target in sensitive_variables(schema, profile, qi_set):
            real_target_keys = row_keys(bundle.real_train, (target.name,))
            synthetic_target_keys = row_keys(bundle.synthetic, (target.name,))
            class_targets: dict[tuple[object, ...], set[tuple[object, ...]]] = defaultdict(set)
            for qi_key, target_key in zip(training_keys, real_target_keys, strict=True):
                class_targets[qi_key].add(target_key)
            for position, (qi_key, target_key) in enumerate(
                zip(synthetic_keys, synthetic_target_keys, strict=True)
            ):
                known_targets = class_targets.get(qi_key)
                if known_targets is not None and len(known_targets) == 1:
                    attribute_disclosures[position] |= target_key in known_targets

    if cluster_analysis is None:
        cluster_labels: tuple[str, ...] | None = None
        exposure_ratios: dict[str, float] = {}
    else:
        cluster_labels = cluster_analysis._synthetic_labels
        exposure_ratios = cluster_analysis._exposure_ratios
        if cluster_labels is not None and len(cluster_labels) != len(bundle.synthetic):
            raise ValueError("cluster assignments do not match the synthetic row count")

    risk_policy = profile.row_risk
    store = RowStore()
    for position, distance_value in enumerate(synthetic_distances):
        dcr_train = _finite_float(distance_value)
        percentile = _empirical_percentile(reference_distribution, dcr_train)
        local_density_ratio = None
        if test_distances is not None:
            test_distance = _finite_float(test_distances[position])
            if dcr_train is not None and test_distance is not None and dcr_train > 0:
                local_density_ratio = test_distance / dcr_train

        exact_match = bool(exact_matches[position])
        near_match = dcr_train is not None and dcr_train <= profile.distance.near_match_threshold
        replicated_unique = bool(replicated_uniques[position])
        attribute_disclosure = bool(attribute_disclosures[position])
        flag_count = sum((exact_match, near_match, replicated_unique, attribute_disclosure))
        band: RiskBand
        if flag_count >= risk_policy.high_min_flags:
            band = "high"
        elif flag_count >= risk_policy.medium_min_flags:
            band = "medium"
        else:
            band = "low"
        cluster_id = cluster_labels[position] if cluster_labels is not None else None
        cluster_exposure = exposure_ratios.get(cluster_id) if cluster_id is not None else None

        row_profile = RowProfile(
            dcr_train=dcr_train,
            dcr_percentile=percentile,
            exact_match=exact_match,
            near_match=near_match,
            replicated_unique=replicated_unique,
            k_syn=tuple(
                (qi_set.name, count)
                for qi_set, count in zip(configured_qis, k_syn[position], strict=True)
            ),
            k_real=tuple(
                (qi_set.name, count)
                for qi_set, count in zip(configured_qis, k_real[position], strict=True)
            ),
            attribute_disclosure=attribute_disclosure,
            local_density_ratio=local_density_ratio,
            cluster_id=cluster_id,
            cluster_exposure_ratio=cluster_exposure,
            composite_band=band,
        )
        store.add_profile(asdict(row_profile))
    return store


def _finite_float(value: float) -> float | None:
    """Return a finite scalar as a Python float, or no value if unavailable."""
    return float(value) if np.isfinite(value) else None


def _empirical_percentile(reference_distribution: np.ndarray, value: float | None) -> float | None:
    """Return the inclusive empirical percentile in the [0, 1] interval."""
    if value is None or reference_distribution.size == 0:
        return None
    rank = np.searchsorted(reference_distribution, value, side="right")
    return float(rank / reference_distribution.size)
