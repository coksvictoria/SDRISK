"""Exact-block synthetic-data linkability attack."""

import numpy as np
import pandas as pd

from sdrisk.bundle import DatasetBundle
from sdrisk.metrics.attacks.common import (
    not_assessed_attack,
    positions_by_key,
    success_result,
)
from sdrisk.metrics.identity.common import qi_sets, row_keys, validate_bundle_columns
from sdrisk.policy import PolicyProfile
from sdrisk.results import MetricResult
from sdrisk.schema import Schema

_LINKABILITY_REASON = (
    "Simplified exact-block linkability attack; QI columns are split in "
    "configured order and this does not implement Anonymeter's attacker."
)


def linkability(
    bundle: DatasetBundle,
    schema: Schema,
    profile: PolicyProfile,
    rng: np.random.Generator,
) -> list[MetricResult]:
    """Evaluate a simplified exact-linkage attack using two QI blocks.

    Definition
        Split each configured QI set at its midpoint. Each synthetic row
        proposes linking its left- and right-block keys. A guess succeeds when
        each key matches exactly one target row and both keys identify the
        same row.
    Inputs
        `real_train`, `synthetic`, `real_test`, and QI sets with at least two
        columns.
    Baseline
        Success rate for the same synthetic links against disjoint
        `real_test` control records.
    Interpretation
        Higher training than control success indicates the proposed links
        identify training rows more often than holdout rows.
    Weaknesses
        This exact-key split is a simplified candidate attack, not
        Anonymeter's linkability algorithm. It does not model external tables,
        probabilistic linkage, or attacker-chosen auxiliary data.
    Citation
        Giomi et al. (2023), *A Unified Framework for Quantifying Privacy Risk
        in Synthetic Data*, https://arxiv.org/abs/2211.10459; the paper defines
        linkability as linking records belonging to the same individual.
    """
    if not isinstance(rng, np.random.Generator):
        raise TypeError("rng must be a numpy.random.Generator")
    selected_qis = qi_sets(schema, profile)
    if bundle.real_test is None:
        return [
            not_assessed_attack("linkability_success", "no holdout supplied", qi_set.name, profile)
            for qi_set in selected_qis
        ]
    if bundle.real_train.empty or bundle.synthetic.empty or bundle.real_test.empty:
        return [
            not_assessed_attack(
                "linkability_success",
                "real_train, synthetic, and real_test must all contain records",
                qi_set.name,
                profile,
            )
            for qi_set in selected_qis
        ]

    results: list[MetricResult] = []
    for qi_set in selected_qis:
        if len(qi_set.columns) < 2:
            results.append(
                not_assessed_attack(
                    "linkability_success",
                    "linkability requires at least two configured QI columns",
                    qi_set.name,
                    profile,
                )
            )
            continue
        validate_bundle_columns(bundle, qi_set.columns)
        split_at = len(qi_set.columns) // 2
        left_columns = tuple(qi_set.columns[:split_at])
        right_columns = tuple(qi_set.columns[split_at:])
        synthetic_left = row_keys(bundle.synthetic, left_columns)
        synthetic_right = row_keys(bundle.synthetic, right_columns)
        guesses = tuple(zip(synthetic_left, synthetic_right, strict=True))
        train_successes = _link_successes(
            bundle.real_train,
            left_columns,
            right_columns,
            guesses,
        )
        control_successes = _link_successes(
            bundle.real_test,
            left_columns,
            right_columns,
            guesses,
        )
        results.append(
            success_result(
                "linkability_success",
                qi_set.name,
                bundle,
                profile,
                successes=train_successes,
                control_successes=control_successes,
                reason=_LINKABILITY_REASON,
            )
        )
    return results


def _link_successes(
    target: pd.DataFrame,
    left_columns: tuple[str, ...],
    right_columns: tuple[str, ...],
    guesses: tuple[tuple[tuple[object, ...], tuple[object, ...]], ...],
) -> int:
    """Count synthetic links uniquely identifying the same target row."""
    left_positions = positions_by_key(row_keys(target, left_columns))
    right_positions = positions_by_key(row_keys(target, right_columns))
    successes = 0
    for left_key, right_key in guesses:
        matching_left = left_positions.get(left_key, [])
        matching_right = right_positions.get(right_key, [])
        if (
            len(matching_left) == 1
            and len(matching_right) == 1
            and matching_left[0] == matching_right[0]
        ):
            successes += 1
    return successes
