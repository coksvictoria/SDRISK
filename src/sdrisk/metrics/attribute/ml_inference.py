"""Synthetic-trained and experimental cross-validated attribute inference."""

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold

from sdrisk.bundle import DatasetBundle
from sdrisk.metrics.attribute.common import (
    assessed_or_not,
    get_qi_sets,
    missing_sensitive_result,
    result_group,
    validate_metric_inputs,
)
from sdrisk.metrics.identity.common import (
    assessed_result,
    filter_result,
    target_keys,
)
from sdrisk.policy import PolicyProfile
from sdrisk.prep.encoder import MixedTypeEncoder
from sdrisk.prep.values import is_missing
from sdrisk.results import MetricResult
from sdrisk.schema import Schema, Variable

_CV_MAX_SPLITS = 5
_LOGISTIC_MAX_ITERATIONS = 1000


def ml_inference_gain(
    bundle: DatasetBundle,
    schema: Schema,
    profile: PolicyProfile,
    rng: np.random.Generator,
) -> list[MetricResult]:
    """Assess QI-to-target classifier accuracy against a marginal-only control.

    Definition
        With a holdout, fit logistic regression on synthetic data and report
        the train-versus-holdout accuracy gap. Compare it with the same gap for
        an independent empirical-marginal target guesser. Without a holdout,
        report an experimental stratified cross-validation accuracy and its
        fold-local marginal-guess baseline using `real_train` only.
    Inputs
        `real_train`, `synthetic`, optional `real_test`, categorical or ordinal
        sensitive targets, and an explicit seeded NumPy generator.
    Baseline
        Marginal-only train/holdout accuracy gap, or fold-local marginal
        accuracy for the experimental no-holdout path.
    Interpretation
        A larger train/holdout gap than the marginal-only gap may indicate
        target predictability concentrated in training records. The
        cross-validation alternative is not a synthetic-data leakage test.
    Weaknesses
        Logistic regression is only one attacker; accuracy ignores probability
        calibration and class-imbalance costs. The no-holdout path is
        experimental and does not provide an independent holdout assessment.
    Citation
        Project-defined diagnostic, AGENTS.md §7.2; the no-holdout behavior
        follows the project's explicitly experimental cross-validation variant.
    """
    if not isinstance(rng, np.random.Generator):
        raise TypeError("rng must be a numpy.random.Generator")
    if not profile.variables.sensitive:
        return missing_sensitive_result("ml_inference_gain", profile)
    if bundle.real_train.empty or bundle.synthetic.empty:
        return [
            assessed_or_not(
                "ml_inference_gain",
                "real_train and synthetic must both contain records",
                profile,
            )
        ]

    results: list[MetricResult] = []
    for qi_set in get_qi_sets(schema, profile):
        targets = validate_metric_inputs(bundle, schema, profile, qi_set)
        for target in targets:
            group = result_group(qi_set, target)
            if target.vtype not in ("categorical", "ordinal"):
                results.append(
                    assessed_or_not(
                        "ml_inference_gain",
                        "classifier inference requires a categorical or ordinal target",
                        profile,
                        group=group,
                    )
                )
                continue
            if bundle.real_test is None:
                result = _cross_validated_real_check(
                    bundle, schema, profile, tuple(qi_set.columns), target, group, rng
                )
            else:
                result = _synthetic_train_holdout_check(
                    bundle, schema, profile, tuple(qi_set.columns), target, group
                )
            results.append(result)
    return results


def _synthetic_train_holdout_check(
    bundle: DatasetBundle,
    schema: Schema,
    profile: PolicyProfile,
    qi_columns: tuple[str, ...],
    target: Variable,
    group: str,
) -> MetricResult:
    """Measure real train/holdout gaps for a model fitted on synthetic records."""
    if bundle.real_test is None:
        raise RuntimeError("holdout check requires real_test")
    synthetic_valid = _nonmissing_target_mask(bundle.synthetic, target.name)
    synthetic = bundle.synthetic.loc[synthetic_valid]
    synthetic_labels, label_map = _labels_for(synthetic, target.name)
    if len(set(synthetic_labels.tolist())) < 2:
        return assessed_or_not(
            "ml_inference_gain",
            "synthetic target has fewer than two observed classes",
            profile,
            group=group,
        )
    encoder = MixedTypeEncoder(schema, columns=qi_columns).fit(
        synthetic.loc[:, list(qi_columns)]
    )
    model = LogisticRegression(max_iter=_LOGISTIC_MAX_ITERATIONS)
    model.fit(
        encoder.transform(synthetic.loc[:, list(qi_columns)]),
        synthetic_labels,
    )
    synthetic_prior = np.bincount(synthetic_labels) / len(synthetic_labels)

    train_values = _evaluation_values(bundle.real_train, target.name, label_map)
    test_values = _evaluation_values(bundle.real_test, target.name, label_map)
    if not train_values or not test_values:
        return assessed_or_not(
            "ml_inference_gain",
            "real_train and real_test must each contain non-missing target values",
            profile,
            group=group,
        )
    train_features = encoder.transform(bundle.real_train.loc[:, list(qi_columns)])
    test_features = encoder.transform(bundle.real_test.loc[:, list(qi_columns)])
    train_accuracy = float(
        np.mean(model.predict(train_features[train_values[0]]) == train_values[1])
    )
    test_accuracy = float(
        np.mean(model.predict(test_features[test_values[0]]) == test_values[1])
    )
    train_marginal = _marginal_accuracy(train_values[1], synthetic_prior)
    test_marginal = _marginal_accuracy(test_values[1], synthetic_prior)
    result = assessed_result(
        "ml_inference_gain",
        train_accuracy - test_accuracy,
        train_marginal - test_marginal,
        None,
        len(train_values[1]),
        None,
        group=group,
    )
    return filter_result(
        result,
        profile,
        contributing_counts=[
            len(train_values[1]),
            len(test_values[1]),
            *np.bincount(synthetic_labels).tolist(),
        ],
    )


def _cross_validated_real_check(
    bundle: DatasetBundle,
    schema: Schema,
    profile: PolicyProfile,
    qi_columns: tuple[str, ...],
    target: Variable,
    group: str,
    rng: np.random.Generator,
) -> MetricResult:
    """Run the explicitly experimental real-data-only cross-validation check."""
    valid = _nonmissing_target_mask(bundle.real_train, target.name)
    data = bundle.real_train.loc[valid]
    labels, _ = _labels_for(data, target.name)
    class_counts = np.bincount(labels)
    if len(class_counts) < 2 or int(class_counts.min()) < 2:
        return assessed_or_not(
            "ml_inference_gain",
            "cross-validation requires at least two target classes with two records each",
            profile,
            group=group,
        )
    n_splits = min(_CV_MAX_SPLITS, int(class_counts.min()))
    seed = int(rng.integers(0, np.iinfo(np.int32).max))
    folds = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    features = data.loc[:, list(qi_columns)]
    correct = 0
    baseline_total = 0.0
    n_evaluated = 0
    supporting_counts: list[int] = []
    for train_positions, validation_positions in folds.split(features, labels):
        train_frame = features.iloc[train_positions]
        validation_frame = features.iloc[validation_positions]
        encoder = MixedTypeEncoder(schema, columns=qi_columns).fit(train_frame)
        train_x = encoder.transform(train_frame)
        validation_x = encoder.transform(validation_frame)
        train_y = labels[train_positions]
        validation_y = labels[validation_positions]
        supporting_counts.extend(
            np.bincount(train_y, minlength=len(class_counts)).tolist()
        )
        supporting_counts.extend(
            np.bincount(validation_y, minlength=len(class_counts)).tolist()
        )
        model = LogisticRegression(max_iter=_LOGISTIC_MAX_ITERATIONS)
        model.fit(train_x, train_y)
        predictions = model.predict(validation_x)
        correct += int(np.sum(predictions == validation_y))
        fold_prior = np.bincount(train_y, minlength=len(class_counts)) / len(train_y)
        baseline_total += float(np.sum(fold_prior[validation_y]))
        n_evaluated += len(validation_y)
    result = assessed_result(
        "ml_inference_gain",
        correct / n_evaluated,
        baseline_total / n_evaluated,
        None,
        n_evaluated,
        correct,
        group=group,
        experimental=True,
        reason=(
            "Experimental cross-validation on real_train only; no independent "
            "real_test was supplied, so this is not a synthetic-data leakage estimate."
        ),
    )
    return filter_result(
        result,
        profile,
        contributing_counts=supporting_counts,
    )


def _nonmissing_target_mask(data: pd.DataFrame, target: str) -> np.ndarray:
    """Return a boolean selector for rows with observed target values."""
    return np.fromiter(
        (not is_missing(value) for value in data[target].tolist()),
        dtype=np.bool_,
        count=len(data),
    )


def _labels_for(
    data: pd.DataFrame, target: str
) -> tuple[np.ndarray, dict[tuple[object, ...], int]]:
    """Encode hashable scalar target keys as stable integer classes."""
    keys = target_keys(data, target)
    mapping: dict[tuple[object, ...], int] = {}
    labels = np.empty(len(keys), dtype=np.int64)
    for index, key in enumerate(keys):
        if key not in mapping:
            mapping[key] = len(mapping)
        labels[index] = mapping[key]
    return labels, mapping


def _evaluation_values(
    data: pd.DataFrame,
    target: str,
    label_map: dict[tuple[object, ...], int],
) -> tuple[np.ndarray, np.ndarray] | tuple[()]:
    """Return valid row positions and encoded labels; unknown labels map to -1."""
    positions = np.flatnonzero(_nonmissing_target_mask(data, target))
    if len(positions) == 0:
        return ()
    all_keys = target_keys(data, target)
    labels = np.fromiter(
        (label_map.get(all_keys[position], -1) for position in positions),
        dtype=np.int64,
        count=len(positions),
    )
    return positions, labels


def _marginal_accuracy(labels: np.ndarray, prior: np.ndarray) -> float:
    """Expected accuracy from independent guesses following a target prior."""
    supported = labels >= 0
    probabilities = np.zeros(len(labels), dtype=np.float64)
    probabilities[supported] = prior[labels[supported]]
    return float(np.mean(probabilities))
