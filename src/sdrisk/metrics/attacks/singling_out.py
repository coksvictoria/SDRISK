"""Exact-QI singling-out attack."""

from collections import Counter

import numpy as np

from sdrisk.bundle import DatasetBundle
from sdrisk.metrics.attacks.common import not_assessed_attack, success_result
from sdrisk.metrics.identity.common import qi_sets, row_keys, validate_bundle_columns
from sdrisk.policy import PolicyProfile
from sdrisk.results import MetricResult
from sdrisk.schema import Schema

_SINGLING_OUT_REASON = (
    "Simplified exact-QI singling-out attack; this does not implement "
    "Anonymeter's randomized predicate-generation algorithm."
)


def singling_out(
    bundle: DatasetBundle,
    schema: Schema,
    profile: PolicyProfile,
    rng: np.random.Generator,
) -> list[MetricResult]:
    """Evaluate a simplified exact-key singling-out attack.

    Definition
        Treat each synthetic QI tuple as an attack guess. A guess succeeds
        against a target partition when exactly one target row has that tuple.
    Inputs
        `real_train`, `synthetic`, `real_test`, and configured QI sets.
    Baseline
        Success rate for the same synthetic guesses against disjoint
        `real_test` control records.
    Interpretation
        Higher training uniqueness than holdout uniqueness indicates more
        synthetic guesses single out training records than control records.
    Weaknesses
        This exact-key variant does not generate randomized conjunctions,
        sample targets as Anonymeter does, or include its naive random attack.
        It is experimental and sensitive to QI selection.
    Citation
        Giomi et al. (2023), *A Unified Framework for Quantifying Privacy Risk
        in Synthetic Data*, https://arxiv.org/abs/2211.10459; this package uses
        the paper's exactly-one success criterion but not its attack sampler.
    """
    if not isinstance(rng, np.random.Generator):
        raise TypeError("rng must be a numpy.random.Generator")
    selected_qis = qi_sets(schema, profile)
    if bundle.real_test is None:
        return [
            not_assessed_attack("singling_out_success", "no holdout supplied", qi_set.name, profile)
            for qi_set in selected_qis
        ]
    if bundle.real_train.empty or bundle.synthetic.empty or bundle.real_test.empty:
        return [
            not_assessed_attack(
                "singling_out_success",
                "real_train, synthetic, and real_test must all contain records",
                qi_set.name,
                profile,
            )
            for qi_set in selected_qis
        ]

    results: list[MetricResult] = []
    for qi_set in selected_qis:
        validate_bundle_columns(bundle, qi_set.columns)
        guesses = row_keys(bundle.synthetic, qi_set.columns)
        train_counts = Counter(row_keys(bundle.real_train, qi_set.columns))
        test_counts = Counter(row_keys(bundle.real_test, qi_set.columns))
        successes = sum(train_counts[key] == 1 for key in guesses)
        control_successes = sum(test_counts[key] == 1 for key in guesses)
        results.append(
            success_result(
                "singling_out_success",
                qi_set.name,
                bundle,
                profile,
                successes=successes,
                control_successes=control_successes,
                reason=_SINGLING_OUT_REASON,
            )
        )
    return results
