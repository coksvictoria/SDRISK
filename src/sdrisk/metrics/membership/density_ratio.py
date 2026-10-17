"""KDE-based density-ratio membership inference."""

import numpy as np
import pandas as pd
from sklearn.neighbors import KernelDensity

from sdrisk.bundle import DatasetBundle
from sdrisk.metrics.identity.common import qi_sets, validate_bundle_columns
from sdrisk.metrics.membership.common import (
    membership_not_assessed,
    membership_results,
)
from sdrisk.policy import PolicyProfile
from sdrisk.prep.encoder import MixedTypeEncoder
from sdrisk.results import MetricResult
from sdrisk.schema import Schema


def density_ratio_mia(
    bundle: DatasetBundle,
    schema: Schema,
    profile: PolicyProfile,
    rng: np.random.Generator,
) -> list[MetricResult]:
    """Evaluate a KDE approximation to the DOMIAS density-ratio attacker.

    Definition
        Estimate synthetic and reference densities with Gaussian KDEs in the
        package's mixed-type encoded QI space. The attack score is
        `log(p_synthetic(x) / p_reference(x))`; a higher score indicates
        stronger membership evidence. Split the holdout into disjoint,
        seeded reference-density and control-query partitions.
    Inputs
        `real_train`, `synthetic`, `real_test`, and configured QI sets. The
        holdout is partitioned into reference and control samples.
    Baseline
        AUC 0.5, attacker advantage 0, and TPR equal to the operating
        false-positive rate for a random attacker.
    Interpretation
        Higher density ratios for training members than for holdout records
        indicate local synthetic over-representation relative to the
        holdout-estimated population density.
    Weaknesses
        This experimental approximation uses one isotropic bandwidth selected
        by Scott's rule in a one-hot encoded mixed-type space. The run RNG
        divides the holdout into disjoint reference and control partitions.
        KDE is sensitive to encoding, bandwidth, dimensionality, and holdout
        size; it is not a faithful reproduction of every DOMIAS implementation.
    Citation
        van Breugel, Sun, Qian, and van der Schaar (2023), *Membership
        Inference Attacks against Synthetic Data through Overfitting
        Detection*, Section 3.2, https://arxiv.org/abs/2302.12580.
    """
    if not isinstance(rng, np.random.Generator):
        raise TypeError("rng must be a numpy.random.Generator")
    selected_qis = qi_sets(schema, profile)
    if bundle.real_test is None:
        return [
            result
            for qi_set in selected_qis
            for result in membership_not_assessed(
                "mia_density", "no holdout supplied", qi_set.name, profile
            )
        ]
    if bundle.real_train.empty or bundle.synthetic.empty or bundle.real_test.empty:
        return [
            result
            for qi_set in selected_qis
            for result in membership_not_assessed(
                "mia_density",
                "real_train, synthetic, and real_test must all contain records",
                qi_set.name,
                profile,
            )
        ]
    if len(bundle.real_test) < 4:
        return [
            result
            for qi_set in selected_qis
            for result in membership_not_assessed(
                "mia_density",
                "real_test requires four rows for independent reference and control samples",
                qi_set.name,
                profile,
            )
        ]

    results: list[MetricResult] = []
    for qi_set in selected_qis:
        validate_bundle_columns(bundle, qi_set.columns)
        reference_data, control_data = _split_reference_and_control(bundle.real_test, rng)
        encoder = MixedTypeEncoder(schema, columns=qi_set.columns).fit(
            bundle.real_train.loc[:, list(qi_set.columns)]
        )
        synthetic_encoded = encoder.transform(bundle.synthetic.loc[:, list(qi_set.columns)])
        member_encoded = encoder.transform(bundle.real_train.loc[:, list(qi_set.columns)])
        reference_encoded = encoder.transform(reference_data.loc[:, list(qi_set.columns)])
        control_encoded = encoder.transform(control_data.loc[:, list(qi_set.columns)])
        bandwidth = _scott_bandwidth(
            min(len(synthetic_encoded), len(reference_encoded)),
            synthetic_encoded.shape[1],
        )
        synthetic_density = KernelDensity(kernel="gaussian", bandwidth=bandwidth).fit(
            synthetic_encoded
        )
        reference_density = KernelDensity(kernel="gaussian", bandwidth=bandwidth).fit(
            reference_encoded
        )
        member_scores = synthetic_density.score_samples(
            member_encoded
        ) - reference_density.score_samples(member_encoded)
        nonmember_scores = synthetic_density.score_samples(
            control_encoded
        ) - reference_density.score_samples(control_encoded)
        finite_members = np.isfinite(member_scores)
        finite_nonmembers = np.isfinite(nonmember_scores)
        if not finite_members.any() or not finite_nonmembers.any():
            results.extend(
                membership_not_assessed(
                    "mia_density",
                    "KDE density-ratio scores are unavailable for one comparison group",
                    qi_set.name,
                    profile,
                )
            )
            continue
        results.extend(
            membership_results(
                member_scores[finite_members],
                nonmember_scores[finite_nonmembers],
                name_prefix="mia_density",
                group=qi_set.name,
                profile=profile,
                rng=rng,
                experimental=True,
                reason=(
                    "Experimental Gaussian KDE density-ratio approximation; "
                    "real_test was split into disjoint reference and control samples; "
                    "interpret with its encoding and dimensionality limitations."
                ),
            )
        )
    return results


def _split_reference_and_control(
    real_test: pd.DataFrame, rng: np.random.Generator
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split holdout records into disjoint KDE-reference and control samples."""
    if len(real_test) < 4:
        raise ValueError("at least four holdout rows are required for the split")
    positions = rng.permutation(len(real_test))
    split_position = len(real_test) // 2
    reference_positions = np.sort(positions[:split_position])
    control_positions = np.sort(positions[split_position:])
    return (
        real_test.iloc[reference_positions].reset_index(drop=True),
        real_test.iloc[control_positions].reset_index(drop=True),
    )


def _scott_bandwidth(n_samples: int, n_features: int) -> float:
    """Return Scott's rule-of-thumb bandwidth for the common KDE scale."""
    if n_samples < 1 or n_features < 1:
        raise ValueError("KDE requires at least one sample and one feature")
    return float(n_samples ** (-1.0 / (n_features + 4)))
