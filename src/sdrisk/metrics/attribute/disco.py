"""DiSCO attribute-disclosure measure."""

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
    contributing_cell_counts,
    filter_result,
    grouped_target_counts,
    wilson_interval,
)
from sdrisk.policy import PolicyProfile
from sdrisk.results import MetricResult
from sdrisk.schema import Schema


def disco(
    bundle: DatasetBundle, schema: Schema, profile: PolicyProfile
) -> list[MetricResult]:
    """Calculate certainty-based disclosure correctness and its original baseline.

    Definition
        Percentage of original records whose QI class has one synthetic target
        value and whose true target equals that prediction, over all original
        records. The baseline is the percentage of original records in
        deterministic original QI/target classes (`Dorig`).
    Inputs
        `real_train` and `synthetic`, one categorical or ordinal target and
        QI set at a time.
    Baseline
        Original-data equivalent: target is determined with certainty by the
        original QI class.
    Interpretation
        A value above baseline indicates that synthetic certainty predicts
        correct original targets more often than original-data certainty alone.
    Weaknesses
        Results depend on exact QI classes and can be unstable for small cells;
        numeric targets require prior policy-defined discretization.
    Citation
        Raab, Nowok, and Dibben (2024), Appendix 1, equations (5) and (8),
        https://arxiv.org/abs/2406.16826.
    """
    if not profile.variables.sensitive:
        return missing_sensitive_result("disco_pct", profile)
    if bundle.real_train.empty or bundle.synthetic.empty:
        return [
            assessed_or_not(
                "disco_pct",
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
                        "disco_pct",
                        "DiSCO requires a categorical or ordinal target; "
                        "discretize numeric targets by policy",
                        profile,
                        group=group,
                    )
                )
                continue

            original_counts = grouped_target_counts(
                bundle.real_train, qi_set.columns, target.name
            )
            synthetic_counts = grouped_target_counts(
                bundle.synthetic, qi_set.columns, target.name
            )
            disco_records = 0
            original_disclosive_records = 0
            contributing_groups: set[tuple[object, ...]] = set()
            for qi_key, original_cells in original_counts.items():
                original_size = sum(original_cells.values())
                synthetic_cells = synthetic_counts.get(qi_key)
                if len(original_cells) == 1:
                    original_disclosive_records += original_size
                    contributing_groups.add(qi_key)
                if synthetic_cells and len(synthetic_cells) == 1:
                    prediction = next(iter(synthetic_cells))
                    disco_records += original_cells[prediction]
                    contributing_groups.add(qi_key)

            n_original = len(bundle.real_train)
            if n_original == 0:
                results.append(
                    assessed_or_not(
                        "disco_pct",
                        "real_train has no records",
                        profile,
                        group=group,
                    )
                )
                continue
            result = assessed_result(
                "disco_pct",
                100.0 * disco_records / n_original,
                100.0 * original_disclosive_records / n_original,
                wilson_interval(disco_records, n_original, scale=100.0),
                n_original,
                disco_records,
                group=group,
            )
            results.append(
                filter_result(
                    result,
                    profile,
                    contributing_counts=[
                        *contributing_cell_counts(
                            original_counts,
                            synthetic_counts,
                            tuple(contributing_groups),
                        ),
                        original_disclosive_records,
                    ],
                )
            )
    return results
