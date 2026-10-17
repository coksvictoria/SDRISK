"""Run reproducible disclosure-risk benchmarks on a bundled public dataset."""

from __future__ import annotations

import argparse
import json
import time
import tracemalloc
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Literal, cast

import numpy as np
import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split

from sdrisk import __version__
from sdrisk.bundle import DatasetBundle
from sdrisk.policy import PolicyProfile
from sdrisk.provenance import make_rng
from sdrisk.results import RunResult
from sdrisk.runner import run_assessment
from sdrisk.schema import Schema, Variable

Scenario = Literal[
    "independent",
    "bootstrap",
    "noisy_copy",
    "overfit",
    "attribute_leak",
    "localized_leak",
    "heavy_noise",
]

SCENARIO_REPLICATES: dict[str, int] = {
    "independent": 3,
    "bootstrap": 2,
    "noisy_copy": 2,
    "overfit": 2,
    "attribute_leak": 2,
    "localized_leak": 3,
    "heavy_noise": 1,
}


def load_public_data(seed: int = 42) -> tuple[DatasetBundle, Schema]:
    """Return a stratified, disjoint split of scikit-learn's bundled UCI data."""
    dataset = load_breast_cancer()
    frame = pd.DataFrame(
        {
            "radius": dataset.data[:, 0],
            "texture": dataset.data[:, 1],
            "smoothness": dataset.data[:, 4],
            "target": [f"class_{value}" for value in dataset.target],
            "record_id": [f"public-{position}" for position in range(len(dataset.data))],
        }
    )
    for column in ("radius", "texture", "smoothness"):
        frame[f"{column}_band"] = pd.qcut(
            frame[column],
            q=3,
            labels=False,
            duplicates="drop",
        ).astype(str)
    frame = frame[
        [
            "radius",
            "texture",
            "smoothness",
            "radius_band",
            "texture_band",
            "smoothness_band",
            "target",
            "record_id",
        ]
    ]
    train, test = train_test_split(
        frame,
        train_size=250,
        random_state=seed,
        stratify=frame["target"],
    )
    train = train.reset_index(drop=True)
    test = test.reset_index(drop=True)
    synthetic = _with_synthetic_ids(train)
    bundle = DatasetBundle(train, synthetic, test)
    schema = _make_schema()
    return bundle, schema


def make_benchmark_profile(
    *,
    qi_choice: Literal["both", "core", "extended"] = "both",
    numeric_weight: float = 1.0,
) -> PolicyProfile:
    """Build a benchmark-only profile with permissive aggregate filtering."""
    if not np.isfinite(numeric_weight) or numeric_weight <= 0:
        raise ValueError("numeric_weight must be a positive finite number")
    q_sets = {
        "core": {"name": "core", "columns": ["radius_band", "texture_band"]},
        "extended": {
            "name": "extended",
            "columns": [
                "radius_band",
                "texture_band",
                "smoothness_band",
                "radius",
                "texture",
                "smoothness",
            ],
        },
    }
    selected_sets = (
        list(q_sets.values())
        if qi_choice == "both"
        else [q_sets[qi_choice]]
    )
    return PolicyProfile.model_validate(
        {
            "name": f"benchmark-{qi_choice}-weight-{numeric_weight:g}",
            "release_tier": "in_lab",
            "output": {"min_cell_size": 1, "rounding_base": 1},
            "variables": {
                "qi_sets": selected_sets,
                "sensitive": ["target"],
                "strata": ["target"],
            },
            "identity": {"k_profile_thresholds": [2, 5, 10]},
            "row_risk": {"medium_min_flags": 1, "high_min_flags": 2},
            "distance": {
                "max_rows_exact": 100_000,
                "subsample_rows": 50_000,
                "near_match_threshold": 0.05,
            },
            "clusters": {"k": 4, "min_cluster_size": 10, "seeds": 3},
            "thresholds": {
                "exact_match_rate": {"review": 0.01, "fail": 0.05},
                "dcr_overfit_share": {"review": 0.55, "fail": 0.65},
                "mia_auc": {"review": 0.55, "fail": 0.65},
            },
            "policy": {"require_test_for_public_tier": False},
        }
    )


def generate_synthetic(
    real_train: pd.DataFrame,
    n_rows: int,
    *,
    scenario: Scenario | str,
    rng: np.random.Generator,
) -> pd.DataFrame:
    """Create a seeded synthetic fixture without modifying the training frame."""
    allowed = set(SCENARIO_REPLICATES)
    if scenario not in allowed:
        raise ValueError(f"unknown benchmark scenario {scenario!r}")
    if n_rows < 1:
        raise ValueError("n_rows must be positive")
    if not isinstance(rng, np.random.Generator):
        raise TypeError("rng must be a numpy.random.Generator")

    qi_columns = (
        "radius",
        "texture",
        "smoothness",
        "radius_band",
        "texture_band",
        "smoothness_band",
    )
    if not set((*qi_columns, "target", "record_id")) <= set(real_train.columns):
        raise ValueError("training data do not contain the benchmark schema columns")

    if scenario == "independent":
        generated = _sample_marginals(real_train, n_rows, rng)
    elif scenario == "bootstrap":
        generated = _bootstrap_rows(real_train, n_rows, rng)
    elif scenario == "noisy_copy":
        generated = _bootstrap_rows(real_train, n_rows, rng)
        generated = _mutate_features(
            generated,
            real_train,
            rng,
            numeric_scale=0.01,
            category_probability=0.03,
        )
    elif scenario == "overfit":
        generated = _knn_resample(real_train, n_rows, rng)
    elif scenario == "attribute_leak":
        generated = _sample_marginals(real_train, n_rows, rng)
        generated["target"] = _majority_targets(real_train, generated)
    elif scenario == "localized_leak":
        generated = _sample_marginals(real_train, n_rows, rng)
        band = real_train["radius_band"].value_counts().idxmin()
        members = real_train.loc[real_train["radius_band"] == band]
        n_copied = max(1, int(n_rows * 0.4))
        copied = _bootstrap_rows(members, n_copied, rng)
        generated.iloc[:n_copied] = copied.to_numpy()
    else:
        generated = _bootstrap_rows(real_train, n_rows, rng)
        generated = _mutate_features(
            generated,
            real_train,
            rng,
            numeric_scale=0.25,
            category_probability=0.8,
        )
        generated["target"] = _sample_column(real_train["target"], n_rows, rng)

    return _with_synthetic_ids(generated.reset_index(drop=True))


def summarize_detection_rates(
    observations: list[dict[str, Any]],
) -> dict[str, dict[str, dict[str, float | int]]]:
    """Summarize boolean aggregate responses by scenario and diagnostic."""
    counts: dict[str, dict[str, list[int]]] = defaultdict(dict)
    for observation in observations:
        scenario = str(observation["scenario"])
        responses = observation["responses"]
        if not isinstance(responses, dict):
            raise ValueError("benchmark responses must be a mapping")
        for metric_name, response in responses.items():
            metric_counts = counts[scenario].setdefault(str(metric_name), [0, 0])
            metric_counts[1] += 1
            metric_counts[0] += int(bool(response))
    return {
        scenario: {
            metric_name: {
                "detections": positive,
                "runs": runs,
                "rate": positive / runs,
            }
            for metric_name, (positive, runs) in sorted(metrics.items())
        }
        for scenario, metrics in sorted(counts.items())
    }


def run_public_benchmark(*, seed: int = 42) -> dict[str, Any]:
    """Run full assessments for planted scenarios and QI/distance sensitivities."""
    base_bundle, schema = load_public_data(seed)
    train = base_bundle.real_train
    test = base_bundle.real_test
    if test is None:
        raise RuntimeError("public benchmark loader must supply a holdout")
    records: list[dict[str, Any]] = []

    for scenario, repetitions in SCENARIO_REPLICATES.items():
        for replicate in range(repetitions):
            case_seed = seed + replicate
            synthetic = generate_synthetic(
                train,
                len(train),
                scenario=scenario,
                rng=make_rng(case_seed),
            )
            records.append(
                _run_case(
                    train,
                    synthetic,
                    test,
                    schema,
                    make_benchmark_profile(),
                    scenario=scenario,
                    replicate=replicate,
                    seed=case_seed,
                    configuration="both-uniform",
                )
            )

    sensitivity_rows: list[dict[str, Any]] = []
    sensitivity_configurations: tuple[
        tuple[str, Literal["core", "extended"], float], ...
    ] = (
        ("core-uniform", "core", 1.0),
        ("extended-weighted", "extended", 3.0),
    )
    for configuration, qi_choice, weight in sensitivity_configurations:
        config_schema = _make_schema(numeric_weight=weight)
        profile = make_benchmark_profile(qi_choice=qi_choice, numeric_weight=weight)
        for scenario in ("independent", "localized_leak"):
            case_seed = seed + 1000 + len(sensitivity_rows)
            synthetic = generate_synthetic(
                train,
                len(train),
                scenario=scenario,
                rng=make_rng(case_seed),
            )
            observation = _run_case(
                train,
                synthetic,
                test,
                config_schema,
                profile,
                scenario=scenario,
                replicate=0,
                seed=case_seed,
                configuration=configuration,
            )
            sensitivity_rows.append(observation)

    rates = summarize_detection_rates(records)
    return {
        "dataset": {
            "name": "Wisconsin Diagnostic Breast Cancer",
            "source": "UCI Machine Learning Repository, bundled with scikit-learn",
            "n_rows": len(base_bundle.real_train) + len(test),
            "n_train": len(train),
            "n_holdout": len(test),
            "features": [
                "mean radius",
                "mean texture",
                "mean smoothness",
                "three train-derived quantile bands",
            ],
        },
        "package_version": __version__,
        "seed": seed,
        "scenario_replicates": SCENARIO_REPLICATES,
        "response_rules": {
            "exact_match_rate": (
                "Extended-QI table exact-match value exceeds its holdout baseline."
            ),
            "dcr_overfit_share": "Table value exceeds its 0.5 no-overfit baseline.",
            "mia_distance_auc": "Table AUC exceeds the 0.5 random-attacker baseline.",
            "attribute_disclosure": (
                "Core-QI homogeneity or CAP/TCAP value exceeds its reported baseline."
            ),
            "localized_exposure": (
                "At least one cluster CER exceeds its own holdout baseline; "
                "CER remains experimental."
            ),
        },
        "observations": records,
        "detection_rates": rates,
        "independent_false_alarm_rates": rates.get("independent", {}),
        "localized_leak_detection_rates": rates.get("localized_leak", {}),
        "sensitivity_observations": sensitivity_rows,
        "memory_measurement": (
            "peak_traced_bytes uses tracemalloc and excludes some native NumPy, "
            "scikit-learn, and BLAS allocations."
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(__file__).parents[2]
        / "benchmarks"
        / "public_validation_results.json",
    )
    parser.add_argument("--seed", type=int, default=42)
    arguments = parser.parse_args()
    if arguments.seed < 0:
        parser.error("--seed must be non-negative")
    result = run_public_benchmark(seed=arguments.seed)
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    arguments.output.write_text(
        json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(f"Aggregate benchmark results written to {arguments.output.resolve()}")


def _make_schema(*, numeric_weight: float = 1.0) -> Schema:
    return Schema(
        variables=[
            Variable(name="radius", vtype="numeric", role=["qi"], weight=numeric_weight),
            Variable(name="texture", vtype="numeric", role=["qi"], weight=numeric_weight),
            Variable(name="smoothness", vtype="numeric", role=["qi"], weight=numeric_weight),
            Variable(name="radius_band", vtype="categorical", role=["qi"]),
            Variable(name="texture_band", vtype="categorical", role=["qi"]),
            Variable(name="smoothness_band", vtype="categorical", role=["qi"]),
            Variable(name="target", vtype="categorical", role=["sensitive", "stratum"]),
            Variable(name="record_id", vtype="categorical", role=["id"]),
        ]
    )


def _sample_marginals(
    reference: pd.DataFrame,
    n_rows: int,
    rng: np.random.Generator,
) -> pd.DataFrame:
    result = {
        column: _sample_column(reference[column], n_rows, rng)
        for column in reference.columns
        if column != "record_id"
    }
    result["record_id"] = [f"pending-{position}" for position in range(n_rows)]
    return cast(pd.DataFrame, pd.DataFrame(result).loc[:, list(reference.columns)])


def _sample_column(
    values: pd.Series,
    n_rows: int,
    rng: np.random.Generator,
) -> list[Any]:
    positions = rng.integers(0, len(values), size=n_rows)
    return values.iloc[positions].tolist()


def _bootstrap_rows(
    reference: pd.DataFrame,
    n_rows: int,
    rng: np.random.Generator,
) -> pd.DataFrame:
    positions = rng.integers(0, len(reference), size=n_rows)
    result = reference.iloc[positions].reset_index(drop=True).copy()
    result["record_id"] = [f"pending-{position}" for position in range(n_rows)]
    return cast(pd.DataFrame, result)


def _mutate_features(
    generated: pd.DataFrame,
    reference: pd.DataFrame,
    rng: np.random.Generator,
    *,
    numeric_scale: float,
    category_probability: float,
) -> pd.DataFrame:
    result = generated.copy()
    for column in ("radius", "texture", "smoothness"):
        standard_deviation = float(reference[column].std())
        result[column] = result[column] + rng.normal(
            0.0,
            numeric_scale * standard_deviation,
            size=len(result),
        )
    for column in ("radius_band", "texture_band", "smoothness_band"):
        mutate = rng.random(len(result)) < category_probability
        categories = reference[column].unique()
        if len(categories) > 1:
            result.loc[mutate, column] = rng.choice(categories, size=int(mutate.sum()))
    return cast(pd.DataFrame, result)


def _knn_resample(
    reference: pd.DataFrame,
    n_rows: int,
    rng: np.random.Generator,
) -> pd.DataFrame:
    """Sample from a local five-neighbor pool around marginally drawn queries."""
    features = reference.loc[:, ["radius", "texture", "smoothness"]].to_numpy()
    scale = np.std(features, axis=0)
    scale[scale == 0] = 1.0
    queries = np.column_stack(
        [
            rng.choice(reference[column].to_numpy(), size=n_rows)
            for column in ("radius", "texture", "smoothness")
        ]
    )
    distances = np.sum(
        ((queries[:, None, :] - features[None, :, :]) / scale) ** 2,
        axis=2,
    )
    n_neighbors = min(5, len(reference))
    nearest = np.argpartition(distances, n_neighbors - 1, axis=1)[:, :n_neighbors]
    choices = np.asarray(
        [rng.choice(neighbors) for neighbors in nearest],
        dtype=np.intp,
    )
    result = reference.iloc[choices].reset_index(drop=True).copy()
    result["record_id"] = [f"pending-{position}" for position in range(n_rows)]
    return cast(pd.DataFrame, result)


def _majority_targets(
    reference: pd.DataFrame,
    generated: pd.DataFrame,
) -> list[str]:
    class_counts: dict[tuple[str, str], Counter[str]] = defaultdict(Counter)
    class_columns = reference.loc[:, ["radius_band", "texture_band", "target"]]
    for row in class_columns.itertuples(index=False, name=None):
        class_counts[(str(row[0]), str(row[1]))][str(row[2])] += 1
    global_majority = str(reference["target"].mode().iloc[0])
    return [
        str(
            class_counts[(radius_band, texture_band)].most_common(1)[0][0]
            if class_counts[(radius_band, texture_band)]
            else global_majority
        )
        for radius_band, texture_band in zip(
            generated["radius_band"],
            generated["texture_band"],
            strict=True,
        )
    ]


def _with_synthetic_ids(data: pd.DataFrame) -> pd.DataFrame:
    result = data.copy()
    result["record_id"] = [f"synthetic-{position}" for position in range(len(result))]
    return cast(pd.DataFrame, result)


def _run_case(
    train: pd.DataFrame,
    synthetic: pd.DataFrame,
    test: pd.DataFrame,
    schema: Schema,
    profile: PolicyProfile,
    *,
    scenario: str,
    replicate: int,
    seed: int,
    configuration: str,
) -> dict[str, Any]:
    bundle = DatasetBundle(train, synthetic, test)
    tracemalloc.start()
    started = time.perf_counter()
    result = run_assessment(bundle, schema, profile, seed=seed)
    runtime_seconds = time.perf_counter() - started
    _, peak_traced_bytes = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    metrics = _aggregate_results(result)
    responses = _responses(metrics)
    return {
        "scenario": scenario,
        "replicate": replicate,
        "configuration": configuration,
        "seed": seed,
        "runtime_seconds": runtime_seconds,
        "peak_traced_bytes": peak_traced_bytes,
        "responses": responses,
        "metrics": metrics,
    }


def _aggregate_results(
    result: RunResult,
) -> dict[str, dict[str, list[dict[str, Any]]]]:
    metrics: dict[str, dict[str, list[dict[str, Any]]]] = {
        "table": defaultdict(list),
        "subset": defaultdict(list),
        "cluster": defaultdict(list),
    }
    result_groups = {
        "table": result.table_results,
        "subset": result.subset_results,
        "cluster": result.cluster_results,
    }
    for level, level_results in result_groups.items():
        for item in level_results:
            if item.value is None or item.baseline is None:
                continue
            metrics[level][item.name].append(
                {
                    "group": item.group,
                    "value": item.value,
                    "baseline": item.baseline,
                    "experimental": item.experimental,
                }
            )
    return metrics


def _responses(metrics: dict[str, dict[str, list[dict[str, Any]]]]) -> dict[str, bool]:
    table = metrics["table"]
    extended_exact = [
        item for item in table.get("exact_match_rate", []) if item["group"] == "extended"
    ]
    dcr_results = table.get("dcr_overfit_share", [])
    mia_results = table.get("mia_distance_auc", [])
    core_attribute = [
        item
        for name in ("homogeneity_share", "cap_accuracy", "tcap_accuracy")
        for item in table.get(name, [])
        if item["group"] == "core"
    ]
    cluster_results = metrics["cluster"].get("cluster_exposure_ratio", [])
    return {
        "exact_match_rate": any(item["value"] > item["baseline"] for item in extended_exact),
        "dcr_overfit_share": any(item["value"] > item["baseline"] for item in dcr_results),
        "mia_distance_auc": any(item["value"] > 0.5 for item in mia_results),
        "attribute_disclosure": any(
            item["value"] > item["baseline"] for item in core_attribute
        ),
        "localized_exposure": any(
            item["value"] > item["baseline"] for item in cluster_results
        ),
    }


if __name__ == "__main__":
    main()
