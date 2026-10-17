"""Nearest-synthetic-distance membership inference."""

import numpy as np

from sdrisk.bundle import DatasetBundle
from sdrisk.distance.gower import GowerDistance
from sdrisk.distance.nn_search import nearest_neighbor_distances
from sdrisk.metrics.identity.common import (
    qi_sets,
    validate_bundle_columns,
)
from sdrisk.metrics.membership.common import (
    membership_not_assessed,
    membership_results,
)
from sdrisk.policy import PolicyProfile
from sdrisk.results import MetricResult
from sdrisk.schema import Schema


def distance_mia(
    bundle: DatasetBundle,
    schema: Schema,
    profile: PolicyProfile,
    rng: np.random.Generator,
) -> list[MetricResult]:
    """Evaluate a nearest-synthetic-distance membership attacker.

    Definition
        Score each real record by the negative nearest Gower distance to the
        released synthetic rows. Compare training members with holdout
        non-members using ROC AUC, attacker advantage, and TPR at 1% and 0.1%
        false-positive rates.
    Inputs
        `real_train`, `synthetic`, `real_test`, and the configured QI sets.
    Baseline
        AUC 0.5, advantage 0, and TPR equal to the operating false-positive
        rate for a random attacker.
    Interpretation
        Higher member scores than holdout scores indicate that training records
        tend to lie closer to synthetic records under the configured Gower
        distance; this is not proof of memorization.
    Weaknesses
        Results depend on the QIs, distance weights, finite synthetic sample,
        and holdout representativeness. The attacker uses only exact nearest
        neighbors and does not observe generator internals.
    Citation
        Project-defined distance attacker in AGENTS.md §7.3; related
        overfitting-based membership analysis is discussed by van Breugel et
        al. (2023), https://arxiv.org/abs/2302.12580.
    """
    if not isinstance(rng, np.random.Generator):
        raise TypeError("rng must be a numpy.random.Generator")
    selected_qis = qi_sets(schema, profile)
    if bundle.real_test is None:
        return [
            result
            for qi_set in selected_qis
            for result in membership_not_assessed(
                "mia_distance", "no holdout supplied", qi_set.name, profile
            )
        ]
    if bundle.real_train.empty or bundle.synthetic.empty or bundle.real_test.empty:
        return [
            result
            for qi_set in selected_qis
            for result in membership_not_assessed(
                "mia_distance",
                "real_train, synthetic, and real_test must all contain records",
                qi_set.name,
                profile,
            )
        ]

    results: list[MetricResult] = []
    for qi_set in selected_qis:
        validate_bundle_columns(bundle, qi_set.columns)
        distance = GowerDistance(schema, columns=qi_set.columns).fit(bundle.synthetic)
        member_distances = nearest_neighbor_distances(
            bundle.real_train,
            bundle.synthetic,
            distance,
        )
        nonmember_distances = nearest_neighbor_distances(
            bundle.real_test,
            bundle.synthetic,
            distance,
        )
        member_scores = -member_distances
        nonmember_scores = -nonmember_distances
        finite_members = np.isfinite(member_scores)
        finite_nonmembers = np.isfinite(nonmember_scores)
        if not finite_members.any() or not finite_nonmembers.any():
            results.extend(
                membership_not_assessed(
                    "mia_distance",
                    "nearest synthetic distances are unavailable for one comparison group",
                    qi_set.name,
                    profile,
                )
            )
            continue
        results.extend(
            membership_results(
                member_scores[finite_members],
                nonmember_scores[finite_nonmembers],
                name_prefix="mia_distance",
                group=qi_set.name,
                profile=profile,
                rng=rng,
            )
        )
    return results
