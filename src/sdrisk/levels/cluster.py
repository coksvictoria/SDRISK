"""Seeded cluster assignments, stability, cluster metrics, and CER."""

from collections import Counter
from dataclasses import dataclass, field
from itertools import combinations

import numpy as np
import pandas as pd
from sklearn.cluster import MiniBatchKMeans
from sklearn.metrics import adjusted_rand_score

from sdrisk.bundle import DatasetBundle
from sdrisk.levels.common import lift_results, tier1_results
from sdrisk.metrics.identity.common import (
    assessed_result,
    filter_result,
    not_assessed,
    qi_sets,
    validate_bundle_columns,
)
from sdrisk.policy import PolicyProfile
from sdrisk.prep.encoder import MixedTypeEncoder
from sdrisk.results import MetricResult
from sdrisk.schema import Schema

_MINIBATCH_N_INIT = 3
_MINIBATCH_MAX_SIZE = 1024


class ClusterModel:
    """Internal mixed-type cluster model fitted only on real training rows."""

    def __init__(
        self,
        encoder: MixedTypeEncoder,
        estimator: MiniBatchKMeans,
        columns: tuple[str, ...],
    ) -> None:
        self._encoder = encoder
        self._estimator = estimator
        self.columns = columns

    def predict(self, data: pd.DataFrame) -> np.ndarray:
        """Assign rows using the fitted training encoder and clustering model."""
        if data.empty:
            return np.empty(0, dtype=np.intp)
        features = data.loc[:, list(self.columns)]
        encoded = self._encoder.transform(features)
        return np.asarray(self._estimator.predict(encoded), dtype=np.intp)


@dataclass(frozen=True)
class ClusterAnalysis:
    """Aggregate cluster results with assignments retained as internal state."""

    results: tuple[MetricResult, ...]
    _model: ClusterModel | None = field(repr=False)
    _train_labels: tuple[str, ...] | None = field(repr=False)
    _synthetic_labels: tuple[str, ...] | None = field(repr=False)
    _test_labels: tuple[str, ...] | None = field(repr=False)
    _exposure_ratios: dict[str, float] = field(repr=False)


def analyze_clusters(
    bundle: DatasetBundle,
    schema: Schema,
    profile: PolicyProfile,
    rng: np.random.Generator,
) -> ClusterAnalysis:
    """Fit stable clusters and evaluate applicable metrics within each cluster.

    MiniBatchKMeans is fitted on the union of configured QIs from
    `real_train`. Small clusters are merged into an `other` group when any
    supplied partition is below `clusters.min_cluster_size`. Cluster exposure
    ratio is experimental and requires `real_test` for its baseline.
    """
    if not isinstance(rng, np.random.Generator):
        raise TypeError("rng must be a numpy.random.Generator")
    if bundle.real_train.empty or bundle.synthetic.empty:
        return _not_assessed_analysis(profile, "real_train and synthetic must both contain records")

    selected_qis = qi_sets(schema, profile)
    columns = tuple(dict.fromkeys(column for qi_set in selected_qis for column in qi_set.columns))
    validate_bundle_columns(bundle, columns)
    n_clusters = profile.clusters.k
    if n_clusters > len(bundle.real_train):
        return _not_assessed_analysis(
            profile,
            "configured cluster count exceeds the number of real training rows",
        )

    encoder = MixedTypeEncoder(schema, columns=columns).fit(bundle.real_train.loc[:, list(columns)])
    training_features = encoder.transform(bundle.real_train.loc[:, list(columns)])
    estimators: list[MiniBatchKMeans] = []
    training_assignments: list[np.ndarray] = []
    batch_size = max(1, min(_MINIBATCH_MAX_SIZE, len(bundle.real_train)))
    for _ in range(profile.clusters.seeds):
        estimator = MiniBatchKMeans(
            n_clusters=n_clusters,
            random_state=int(rng.integers(0, np.iinfo(np.int32).max)),
            n_init=_MINIBATCH_N_INIT,
            batch_size=batch_size,
        )
        estimator.fit(training_features)
        estimators.append(estimator)
        training_assignments.append(np.asarray(estimator.labels_, dtype=np.intp))

    pairwise_ari = [
        adjusted_rand_score(left, right) for left, right in combinations(training_assignments, 2)
    ]
    stability = float(np.mean(pairwise_ari)) if pairwise_ari else None
    model = ClusterModel(encoder, estimators[0], columns)
    train_raw = training_assignments[0]
    synthetic_raw = model.predict(bundle.synthetic)
    test_raw = None if bundle.real_test is None else model.predict(bundle.real_test)
    train_labels, synthetic_labels, test_labels = _merge_small_clusters(
        train_raw,
        synthetic_raw,
        test_raw,
        minimum=profile.clusters.min_cluster_size,
    )

    results: list[MetricResult] = []
    if stability is None:
        results.append(
            filter_result(
                not_assessed(
                    "cluster_stability_ari",
                    "stability requires at least two configured cluster seeds",
                ),
                profile,
            )
        )
    else:
        stability_reason = (
            "Identical assignments were found across seeded fits."
            if stability == 1.0
            else "Seeded fits produced different assignments; interpret cluster results cautiously."
        )
        results.append(
            filter_result(
                assessed_result(
                    "cluster_stability_ari",
                    stability,
                    1.0,
                    None,
                    len(bundle.real_train),
                    None,
                    reason=stability_reason,
                ),
                profile,
            )
        )

    exposure_ratios: dict[str, float] = {}
    labels = sorted(
        set(train_labels)
        | set(synthetic_labels)
        | (set(test_labels) if test_labels is not None else set())
    )
    for label in labels:
        train_count = train_labels.count(label)
        synthetic_count = synthetic_labels.count(label)
        test_count = test_labels.count(label) if test_labels is not None else None
        partition_counts = [train_count, synthetic_count]
        if test_count is not None:
            partition_counts.append(test_count)
        if any(count < profile.clusters.min_cluster_size for count in partition_counts):
            guard = filter_result(
                MetricResult(
                    name="cluster_min_size_guard",
                    level="cluster",
                    group=label,
                    value=None,
                    baseline=None,
                    ci=None,
                    n_records=None,
                    n_flagged=None,
                    status="suppressed",
                    reason="merged cluster remains below the configured minimum size",
                ),
                profile,
            )
            results.append(guard)
            continue

        if test_count is None:
            exposure = filter_result(
                not_assessed(
                    "cluster_exposure_ratio",
                    "no holdout supplied for the CER baseline",
                    group=label,
                    level="cluster",
                ),
                profile,
            )
        else:
            train_share = train_count / len(bundle.real_train)
            synthetic_share = synthetic_count / len(bundle.synthetic)
            test_share = test_count / len(bundle.real_test) if bundle.real_test is not None else 0.0
            value = synthetic_share / train_share
            baseline = test_share / train_share
            exposure_ratios[label] = value
            exposure = filter_result(
                MetricResult(
                    name="cluster_exposure_ratio",
                    level="cluster",
                    group=label,
                    value=value,
                    baseline=baseline,
                    ci=None,
                    n_records=synthetic_count,
                    n_flagged=None,
                    status="review",
                    reason="Experimental cluster exposure diagnostic; not a validated risk metric.",
                    experimental=True,
                ),
                profile,
                contributing_counts=tuple(partition_counts),
            )
        results.append(exposure)

        selected_bundle = _cluster_bundle(
            bundle,
            train_labels,
            synthetic_labels,
            test_labels,
            label,
        )
        results.extend(
            lift_results(
                tier1_results(selected_bundle, schema, profile, rng),
                profile,
                level="cluster",
                group=label,
                contributing_counts=tuple(partition_counts),
            )
        )

    return ClusterAnalysis(
        results=tuple(results),
        _model=model,
        _train_labels=train_labels,
        _synthetic_labels=synthetic_labels,
        _test_labels=test_labels,
        _exposure_ratios=exposure_ratios,
    )


def _merge_small_clusters(
    train: np.ndarray,
    synthetic: np.ndarray,
    test: np.ndarray | None,
    *,
    minimum: int,
) -> tuple[tuple[str, ...], tuple[str, ...], tuple[str, ...] | None]:
    """Map undersized raw labels to a shared `other` bucket."""
    train_counts = Counter(train.tolist())
    synthetic_counts = Counter(synthetic.tolist())
    test_counts = Counter(test.tolist()) if test is not None else None
    raw_labels = set(train_counts) | set(synthetic_counts)
    if test_counts is not None:
        raw_labels |= set(test_counts)
    small_labels = {
        label
        for label in raw_labels
        if train_counts.get(label, 0) < minimum
        or synthetic_counts.get(label, 0) < minimum
        or (test_counts is not None and test_counts.get(label, 0) < minimum)
    }

    def names(values: np.ndarray) -> tuple[str, ...]:
        return tuple(
            "other" if value in small_labels else f"cluster_{value}" for value in values.tolist()
        )

    return names(train), names(synthetic), names(test) if test is not None else None


def _cluster_bundle(
    bundle: DatasetBundle,
    train_labels: tuple[str, ...],
    synthetic_labels: tuple[str, ...],
    test_labels: tuple[str, ...] | None,
    label: str,
) -> DatasetBundle:
    """Select rows for one merged cluster without exposing assignments."""
    train = bundle.real_train.iloc[
        np.flatnonzero(np.asarray(train_labels, dtype=object) == label)
    ].reset_index(drop=True)
    synthetic = bundle.synthetic.iloc[
        np.flatnonzero(np.asarray(synthetic_labels, dtype=object) == label)
    ].reset_index(drop=True)
    test = None
    if bundle.real_test is not None and test_labels is not None:
        test = bundle.real_test.iloc[
            np.flatnonzero(np.asarray(test_labels, dtype=object) == label)
        ].reset_index(drop=True)
    return DatasetBundle(train, synthetic, test)


def _not_assessed_analysis(profile: PolicyProfile, reason: str) -> ClusterAnalysis:
    """Return an explicit failed-applicability cluster outcome."""
    result = filter_result(
        not_assessed("cluster_analysis", reason),
        profile,
    )
    return ClusterAnalysis(
        results=(result,),
        _model=None,
        _train_labels=None,
        _synthetic_labels=None,
        _test_labels=None,
        _exposure_ratios={},
    )
