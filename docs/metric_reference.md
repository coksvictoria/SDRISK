# Metric reference

This reference covers the table-level identity metrics implemented in M3 and
attribute metrics implemented in M4. Results are filtered before they are
returned. The M7 decision layer applies only configured thresholds; results
without configured thresholds remain in `review`.

## Exact match rate

For a configured QI set, the value is the share of `real_train` rows having at
least one synthetic row with the same QI values. Numeric columns use their
schema-declared tolerance when positive; all other columns require exact
agreement. Missing values match other missing values. The baseline is the share
of training rows matching `real_test` under the same comparison. Without a
holdout baseline, the metric is `not_assessed`.

This is a record-level identity indicator, not a probability that a person has
been identified. Results depend on QI selection and tolerance.

## Unique exact match rate

The value is the share of exact-QI-unique `real_train` rows that match at least
one synthetic row. The baseline is the equivalent share for exact-QI-unique
`real_test` rows matched against the synthetic data. Uniqueness is determined
by exact QI equivalence classes; numeric tolerance applies only to the match,
not to uniqueness. The metric is `not_assessed` if a holdout or either unique
population is absent.

## UiO and RepU

The implementation follows the definitions in Raab, Nowok, and Dibben (2024),
Appendix 1, equations (1) and (4):

- `UiO = 100 * count(d.q == 1) / N_d`, the percentage of original records whose
  QI combination is unique in `real_train`.
- `RepU = 100 * count(d.q == 1 and s.q == 1) / N_d`, the percentage of original
  records that are unique in both the original and synthetic QI tables.

RepU is returned with UiO as its baseline. QI equivalence is exact; the
match-tolerance setting is not used for these published uniqueness measures.
This does not measure attribute or membership disclosure.

Citation: Raab, Nowok, and Dibben, “Practical privacy metrics for synthetic
data,” arXiv:2406.16826, Appendix 1, equations (1)–(4),
[arXiv](https://arxiv.org/abs/2406.16826).

Exact and unique exact-match metrics are also related to the [NIST SDNist
benchmark and evaluation tools](https://github.com/usnistgov/SDNist).

## k-profile

For each profile-configured cutoff `k`, the value is the share of synthetic rows
whose exact QI class has fewer than `k` records. The baseline is the same share
in `real_train`. Cutoffs are configured under `identity.k_profile_thresholds`;
`[2, 5, 10]` are illustrative values only, not release-policy approvals.

## DCR overfitting share

For each synthetic row, compare its nearest Gower distance to `real_train` and
`real_test`. The value is the share of finite, non-tied comparisons closer to
training data; ties (within a small floating-point comparison tolerance) are
excluded. The baseline is 0.5. The metric requires a holdout, and its meaning
depends on the QI set, reference-fitted ranges, weights, and holdout
representativeness.

This is the raw closer-to-training share specified by the project brief. It is
not the transformed protection score used by [SDMetrics
DCROverfittingProtection](https://docs.sdv.dev/sdmetrics/data-metrics/privacy/dcroverfittingprotection).

## DCR versus a marginal-only control

The value is the median synthetic-to-training nearest distance divided by the
median nearest distance for an independent sample drawn from each training
column's empirical marginal. The baseline is 1.0. This control follows the
project brief's empirical-marginal specification rather than the uniform-range
random sampling described by [SDMetrics
DCRBaselineProtection](https://docs.sdv.dev/sdmetrics/data-metrics/privacy/dcrbaselineprotection).
Values below one indicate closer synthetic proximity than the marginal-only
control under that distance. If either sample has no comparable distances, or
the control median is zero, the result is `not_assessed` rather than an
infinite or non-finite ratio. The independent-marginal control discards
dependencies and may generate implausible records. Exact median components
are not exposed; when the control median is zero, the not-assessed reason
identifies the zero component and whether the synthetic median is zero or
positive without revealing an exact continuous value.

## CAP and TCAP

For categorical or ordinal targets, CAP is the expected probability that a
synthetic target sampled from the synthetic conditional distribution for a
real row's exact QI class equals that row's true target. Unsupported original
QI classes contribute zero; the denominator is the full original size
(`DCAP_d`-style). Its baseline is the expected accuracy of an independent
target guess sampled from the original empirical marginal. CAP is
`not_assessed` for continuous targets; use the configured numeric-tolerance
metric instead.

TCAP is the fraction of original records correctly predicted among records
whose QI class is present in synthetic data and has exactly one synthetic target
value. Its baseline is marginal-only expected accuracy on those same records.
Unlike DiSCO, TCAP uses this conditional denominator, so it may look high when
few original rows qualify. These denominators follow the variants distinguished
in Raab, Nowok, and Dibben (2024), Section 3.3.2,
[arXiv:2406.16826](https://arxiv.org/abs/2406.16826).

## DiSCO

DiSCO is the percentage of original records whose QI class has one synthetic
target value and for which that value matches the original target, divided by
the full original size. Its baseline is `Dorig`, the corresponding percentage
of original records whose target is determined by their original QI class.
This follows Appendix 1, equations (5) and (8), in
[Raab, Nowok, and Dibben (2024)](https://arxiv.org/abs/2406.16826). Numeric
targets are not discretized implicitly; any discretization must be policy
defined before assessment.

## Numeric-tolerance recovery

A finite real numeric target is counted as recovered when at least one
synthetic target with the same exact QI values is within its schema-declared
tolerance. Unmatched QI classes count as not recovered. The baseline is the
expected recovery rate if a target is independently sampled from the real
training marginal. Missing tolerance or non-numeric target variables produce
`not_assessed`.

## Homogeneity

Homogeneity is the share of synthetic QI equivalence classes meeting the
profile's `output.min_cell_size` for which all records share one exact
sensitive value. The baseline is the equivalent share of eligible original
classes. Classes are equally weighted, not weighted by record count. Derived
class and target counts are passed through output suppression.

## ML inference gain

With `real_test`, a logistic-regression classifier is fitted to synthetic QIs
and targets. The value is its accuracy gap between real training members and
holdout non-members; the baseline is the corresponding gap for an independent
guess from the synthetic target marginal. Without `real_test`, a stratified
cross-validated model on `real_train` is reported against fold-local marginal
guessing and marked `experimental=True`. This fallback measures predictive
structure in the real data and is not a synthetic-data leakage estimate.
Only classification accuracy is used; more powerful attackers and calibration
are not assessed.

## Output status and suppression

Computed identity values always carry a baseline and pass through
`output_filter`. A contributing count below the profile minimum suppresses the
value, baseline, interval, and counts. Metrics not yet assigned policy
thresholds retain `review` and require manual adjudication.

## Subset analysis

Subset analyses lift the implemented table-level metrics to each configured
stratum value and each exact QI equivalence class. Candidate keys are collected
from all supplied datasets, so synthetic-only and holdout-only groups are not
silently omitted. Result group labels contain only the stratum or QI-set name;
observed values and exact keys are not exposed. A group is suppressed when a
required train, synthetic, or supplied holdout partition is below
`output.min_cell_size`. The JSON and HTML reports summarize metrics by generic
QI-set or stratum label, with unweighted medians of reportable values and
baselines plus output-filtered status counts. They report minimum-n suppressed
group counts by label; individual equivalence-class keys are never exposed.
No multiple-comparison correction is applied, and isolated results require
cautious interpretation.

## Cluster analysis and cluster exposure ratio

MiniBatchKMeans is fit on `real_train` using the mixed-type encoding of the
union of configured QIs. The first seeded fit supplies assignments; mean
pairwise adjusted Rand index (ARI) across seeded fits reports stability. A
reason explicitly calls out any departure from perfect agreement; no
policy-approved stability cutoff is currently configured. Clusters that fall
below `clusters.min_cluster_size` in any supplied partition are merged into
`other`, and a remaining undersized bucket is suppressed. Applicable Tier 1
metrics are recalculated within each reportable cluster.

The experimental cluster exposure ratio for cluster `c` is
`(share_synthetic(c) / share_train(c))`. Its baseline is
`(share_test(c) / share_train(c))`; without a holdout it is `not_assessed`.
CER is a candidate diagnostic, not an established privacy metric, and remains
experimental and unvalidated. Stability and CER are both aggregate diagnostics;
no cluster's raw record assignments are exportable.

## Internal row profiles

`RowStore` holds per-synthetic-row nearest-training distance and its empirical
percentile against leave-one-out training distances, exact and near-match
flags, replicated-unique status, QI class sizes, attribute-correctness flags,
and optional holdout-distance and cluster diagnostics. The optional
`local_density_ratio` is a nearest-holdout distance divided by
nearest-training distance, not a density estimator. Composite bands count
exact match, near match, replicated uniqueness, and correct sensitive-attribute
inference flags using `row_risk.medium_min_flags` and
`row_risk.high_min_flags`.

Row-level profiles remain internal. The only supported summary API returns
low/medium/high counts after `output_filter` suppression and rounding; profiles,
nearest-record matches, cluster assignments, and exact distance values must
not be logged, serialized, or returned from report APIs.

## Distance-based membership inference

For every configured QI set, each real query is scored by the negative distance
to its nearest synthetic row using training-fitted Gower ranges. Training rows
are members and holdout rows are non-members. Results include ROC AUC,
attacker advantage (`max(TPR - FPR)`), and TPR at 1% and 0.1% FPR; each metric
has a stratified 200-replicate percentile bootstrap 95% interval. Random
attacker baselines are respectively 0.5, 0, 0.01, and 0.001. All four
statistics are `not_assessed` without `real_test`.

This is a black-box distance attacker over the released synthetic sample. It
does not query or inspect the generator and is sensitive to QI selection,
distance weights, synthetic sample size, and holdout representativeness.

## KDE density-ratio membership inference (experimental)

The DOMIAS-style score is `log(p_synthetic(x) / p_reference(x))`, with both
densities estimated by Gaussian KDE in the training-fitted mixed-type encoded
QI space. A shared Scott-rule bandwidth is selected from the smaller
synthetic/reference sample size and encoded dimension. `real_test` is split
with the explicit run RNG into disjoint reference-density and control-query
samples, so a control row is never scored against a KDE trained on itself. AUC,
advantage, and low-FPR TPR results and their baselines match the distance-based
MIA. Every result is marked experimental.

This implements the density-ratio premise in Section 3.2 of van Breugel et al.
(2023), not a canonical DOMIAS configuration. KDE behavior can be unstable in
high dimensions and depends on the encoding and bandwidth. No result is
reported without a holdout.

## Simplified singling-out and linkability attacks (experimental)

Singling-out uses each synthetic QI tuple as a guess; a guess succeeds when
exactly one training or holdout row matches. Linkability splits each configured
QI set at its midpoint in declared column order. A synthetic row proposes a
link between those two blocks; success requires each block to match exactly
one target row and both matches to identify the same row. The reported value
is the training success rate and the baseline is the same guesses' success
rate against disjoint holdout records. Wilson intervals are reported for the
training success rate. Both variants are marked experimental and are
`not_assessed` without `real_test`.

These are deliberately simplified exact-key attacks, not Anonymeter's full
algorithms. In particular, they do not generate randomized attribute
predicates, use external auxiliary tables, or run Anonymeter's naive random
attack. See Giomi et al. (2023),
[arXiv:2211.10459](https://arxiv.org/abs/2211.10459) for the main/naive/control
framework and attack definitions.
