# Design decisions

## 2026-10-17: Use uv to manage the project environment and lock

- **Decision:** Declare package metadata and dependency groups in `pyproject.toml`, pin the
  resolved environment in `uv.lock`, and use uv for local setup and CI.
- **Reason:** The requested workflow should be repeatable without mixing direct pip installs
  with project management commands.
- **Alternatives considered:** Keep pip-based editable installation and add uv only as an
  optional local tool.
- **Status:** Accepted.

## 2026-10-17: Keep setuptools as the build backend while adopting uv

- **Decision:** Use uv for dependency resolution, locking, and environment execution while
  retaining the existing setuptools backend.
- **Reason:** uv does not require a backend change, and keeping the existing backend avoids
  unrelated packaging migration risk.
- **Alternatives considered:** Replace setuptools with Hatchling or another backend.
- **Status:** Accepted.

## 2026-10-17: Leave package licence metadata unset until approval

- **Decision:** Do not declare a package licence identifier or licence-file expression
  while `LICENSE` remains an explicit placeholder.
- **Reason:** The project brief leaves ownership and distribution rights unresolved; package
  metadata must not imply permission that has not been granted.
- **Alternatives considered:** Publish a guessed SPDX identifier or mark the placeholder as
  the package's actual licence.
- **Status:** Accepted pending licence approval.

## 2026-10-17: Add M1 scientific and validation dependencies

- **Decision:** Add NumPy, pandas, Pydantic v2, and PyYAML as runtime dependencies; add
  Hypothesis and pandas/PyYAML type stubs to the development dependency group.
- **Reason:** These dependencies are required by the M1 data model, policy loader, seeded RNG,
  property-based output-safety checks, and strict mypy validation.
- **Alternatives considered:** Defer dependencies and implement temporary substitutes.
- **Status:** Accepted; additional metric-specific dependencies remain deferred to later
  milestones.

## 2026-10-17: Use training-reference ranges for Gower distance

- **Decision:** Fit numeric, ordinal, and date ranges on the reference dataframe; omit
  pairwise-missing variables from the weighted denominator, cap normalized numeric
  differences at one, and use a binary mismatch for constant reference columns.
- **Reason:** Reference-fitted ranges keep the distance scale fixed when querying synthetic
  or holdout rows, and explicit missing handling avoids treating missing values as matches.
- **Alternatives considered:** Recalculate ranges on every pair of input frames or impute
  missing values before distance calculation.
- **Status:** Accepted for the M2 distance engine; metric-specific interpretations must be
  documented when used.

## 2026-10-17: Treat blocking as a local candidate restriction

- **Decision:** Exact-value blocking returns the nearest candidate within the matching block;
  if no block exists, the default is a full-reference fallback.
- **Reason:** This makes the local search scope explicit and avoids silently reporting no
  distance merely because a block has no candidate.
- **Alternatives considered:** Always return `NaN` on an empty block or search globally even
  when a matching block exists.
- **Status:** Accepted; callers must not interpret a block-restricted result as a global
  nearest neighbor.

## 2026-10-17: Keep categorical encoding levels fit-scoped

- **Decision:** Learn categorical levels from the training frame in first-seen order and
  reserve separate encoded positions for missing and unseen levels.
- **Reason:** This keeps transform behavior fixed after fit while retaining information
  about missing and previously unobserved synthetic values.
- **Alternatives considered:** Recompute levels per transformed frame or discard unseen
  values as all-zero vectors.
- **Status:** Accepted.

## 2026-10-17: Keep M2 preprocessing dependency-free

- **Decision:** Implement preprocessing, exact Gower distance, blocking, and sampling with
  the existing NumPy and pandas dependencies; defer approximate-neighbor libraries.
- **Reason:** The exact reference implementation is testable against brute force and avoids
  committing to an optional backend before workload requirements are confirmed.
- **Alternatives considered:** Add scikit-learn or an approximate-neighbor package during M2.
- **Status:** Accepted.

## 2026-10-17: Implement RepU from the cited definition

- **Decision:** Calculate UiO as the percentage of original records unique on exact QI
  combinations and RepU as the percentage of original records unique in both the original
  and synthetic QI tables; use UiO as the RepU baseline.
- **Reason:** The formula was verified in Raab, Nowok, and Dibben (2024), Appendix 1,
  equations (1) and (4), rather than inferred from the name of the metric.
- **Alternatives considered:** Use a generic synthetic uniqueness ratio or apply match
  tolerance to the published QI equivalence definitions.
- **Status:** Accepted; implementation citation is in the metric docstring.

## 2026-10-17: Require holdout controls for match-rate baselines

- **Decision:** Return exact match rate and UEM as `not_assessed` without `real_test`, because
  their configured control baselines cannot be calculated.
- **Reason:** The project-wide rule requires every reported risk value to have a baseline.
- **Alternatives considered:** Report the risk value alone or substitute an unverified
  constant baseline.
- **Status:** Accepted.

## 2026-10-17: Configure k-profile cutoffs in policy

- **Decision:** Add `identity.k_profile_thresholds` to each policy profile; the example uses
  the project-brief values 2, 5, and 10.
- **Reason:** The class-size cutoffs are configuration, and the M2 result model did not yet
  expose a policy field for them.
- **Alternatives considered:** Hard-code the three cutoffs in the metric implementation.
- **Status:** Accepted; example values are not release-policy approvals.

## 2026-10-17: Defer metric adjudication to the decision milestone

- **Decision:** Return measured M3 results as provisional `review` with an explicit reason;
  map configured risk thresholds to final status in M7.
- **Reason:** M3 implements measurements and baselines, while the shared decision and
  confidence behavior is scoped to M7.
- **Alternatives considered:** Treat unadjudicated values as `pass` or add metric-specific
  decision logic to each implementation.
- **Status:** Accepted.

## 2026-10-17: Use setuptools as the build backend

- **Decision:** Use setuptools for the initial package scaffold.
- **Reason:** It is widely available in target Python environments and supports a standards-based
  `pyproject.toml` configuration for a `src/` layout.
- **Alternatives considered:** Hatchling, flit-core, Poetry.
- **Status:** Provisional; may be changed if the target environment requires another backend.

## 2026-10-17: Use explicit output-filter configuration

- **Decision:** Require minimum cell size and rounding base to be supplied to the output
  filter rather than embedding agency policy defaults in code.
- **Reason:** The applicable disclosure-control rules are unresolved and must remain policy
  configuration.
- **Alternatives considered:** Use the suggested placeholders from the project brief as
  implicit defaults.
- **Status:** Accepted; profile example values remain marked `TODO(confirm)`.

## 2026-10-17: Round exportable counts to the nearest configured base

- **Decision:** Round reported counts to the nearest multiple of the configured base, with
  halfway values rounded upward.
- **Reason:** This provides deterministic rounding without embedding a particular base in
  code; the profile supplies the still-unconfirmed base.
- **Alternatives considered:** Always round down or always round up.
- **Status:** Provisional; confirm the target agency's required rounding convention.

## 2026-10-17: Use Python 3.10 as the provisional minimum

- **Decision:** Configure tooling for Python 3.10 and test CI on 3.10, 3.11, and 3.12.
- **Reason:** The project brief proposes Python 3.10+ but marks target-environment support
  as `TODO(confirm)`.
- **Alternatives considered:** Python 3.11+ only.
- **Status:** Provisional, `TODO(confirm)`.

## 2026-10-17: Define table-level attribute metric variants

- **Decision:** Calculate CAP as expected correct-target probability under the
  synthetic conditional target distribution, with the original-data denominator
  (`DCAP_d`-style); calculate TCAP only among original rows whose QI class is
  present and deterministic in synthetic data; calculate DiSCO as the share of
  original rows correctly predicted with certainty by synthetic data, against
  the original-data deterministic-class share (`Dorig`). Homogeneity eligibility
  uses the profile's configured `output.min_cell_size` for both synthetic and
  original QI classes.
- **Reason:** Raab, Nowok, and Dibben (2024), Section 3.3.2 and Appendix 1,
  equations (5) and (8), distinguish CAP/TCAP denominator variants and define
  DiSCO. The original-data denominator avoids inflating disclosure scores when
  the synthetic release is smaller. For CAP and TCAP, the marginal-only
  baseline is an independent target guess sampled from the empirical marginal.
- **Alternatives considered:** Use a single unqualified CAP/TCAP definition;
  divide only by supported records for CAP; compare against a modal-value
  guesser.
- **Status:** Accepted; these denominator and baseline choices are stated in
  metric documentation.

## 2026-10-17: Specify experimental ML attribute inference

- **Decision:** With a holdout, train a logistic-regression classifier on
  synthetic QIs and targets, and report its train-versus-holdout accuracy gap
  beside the equivalent gap for an empirical-marginal random guesser. Without
  a holdout, report a stratified cross-validated real-data predictive check
  against a fold-local marginal guesser and mark it experimental.
- **Reason:** This follows the project brief's training-member/holdout comparison
  while making the no-holdout alternative explicitly exploratory. The
  cross-validated check is not presented as evidence of synthetic-data leakage.
- **Alternatives considered:** Report raw model accuracy without a marginal
  baseline; treat the no-holdout cross-validation result as a validated privacy
  metric.
- **Status:** Provisional diagnostic; model and fold choices are implementation
  conventions, not agency policy thresholds.

## 2026-10-17: Add scikit-learn for attribute inference

- **Decision:** Add scikit-learn as a required runtime dependency for the
  documented logistic-regression and cross-validation implementation.
- **Reason:** Scikit-learn is part of the project's core stack and provides
  maintained, deterministic classification primitives; mixed-type feature
  preparation reuses the package's existing encoder.
- **Alternatives considered:** Implement a classifier and stratified folds
  directly, or omit the ML inference metric.
- **Status:** Accepted.

## 2026-10-17: Keep subset labels aggregate and generic

- **Decision:** Generate subset tests from the union of configured strata
  values and exact QI equivalence classes observed in any supplied dataset,
  but never include the category or QI values in result labels. Return a
  suppressed guard result when any required subset partition falls below the
  configured minimum; report tested and suppressed counts separately and make
  no multiple-comparison correction.
- **Reason:** Group labels must not become a side channel for real QI values.
  A correction method is not specified by the current brief, so the analysis
  reports its absence rather than selecting one implicitly.
- **Alternatives considered:** Include raw subset keys in result groups; omit
  values occurring only in synthetic or holdout inputs; apply an unapproved
  statistical correction.
- **Status:** Accepted; intersections are not generated without an explicit
  profile definition. The result distinguishes evaluated subsets from
  suppressed candidates.

## 2026-10-17: Fit stable clusters and mark CER experimental

- **Decision:** Fit MiniBatchKMeans on the mixed-type encoding of the union of
  configured QIs, use the first seeded fit for assignments, and report mean
  pairwise adjusted Rand agreement across configured seed fits. Any departure
  from perfect ARI is explicitly called out in the result reason because no
  policy cutoff is specified. Merge a raw cluster into `other` if it misses
  the profile minimum in any supplied partition. Compare synthetic/train
  exposure ratio with the matching holdout/train ratio; mark CER experimental
  and not assessed without holdout.
- **Reason:** These choices follow the M5 cluster-model and CER specification
  while making training instability and small partitions explicit.
- **Alternatives considered:** Select the best-performing seed; report raw
  labels for undersized clusters; present CER as established.
- **Status:** Provisional diagnostic; stability and CER validation remains
  outstanding; CER remains experimental and must not be presented as validated.

## 2026-10-17: Restrict row profiles and configure composite bands

- **Decision:** Store per-synthetic-row profiles only in RowStore; compute
  training-distance percentiles against leave-one-out nearest-training
  distances; treat exact match, near match, replicated uniqueness, and
  attribute-correctness as the composite-band flags. Require explicit
  `row_risk.medium_min_flags` and `row_risk.high_min_flags` settings; only
  filtered band aggregates are returned. The internal local-density field is
  a train/holdout nearest-distance ratio proxy, not a density estimator.
- **Reason:** The profile thresholds remain policy-controlled, leave-one-out
  references avoid self-distance zeros, and RowStore must not expose row-level
  details through its public summary method.
- **Alternatives considered:** Hard-code band thresholds; expose row profiles
  for debugging; label a distance ratio as a validated density estimate.
- **Status:** Provisional; example band thresholds are `TODO(confirm)`.

## 2026-10-17: Add holdout-controlled membership metrics

- **Decision:** Score members and holdout non-members by negative nearest
  synthetic Gower distance; report AUC, maximum attacker advantage, and TPR at
  1% and 0.1% FPR, each with a stratified 200-replicate percentile bootstrap
  interval. Use random-attacker baselines of 0.5, 0, 0.01, and 0.001.
- **Reason:** These are the outcomes explicitly requested by the M6 brief. A
  stratified bootstrap preserves the member and non-member sample sizes and
  uses the run's explicit generator.
- **Alternatives considered:** Report only AUC; bootstrap combined labels
  without preserving group sizes.
- **Status:** Accepted as a distance-based diagnostic; results depend on
  configured QIs, Gower weights, and holdout representativeness.

## 2026-10-17: Approximate DOMIAS with a mixed-type KDE density ratio

- **Decision:** Fit Gaussian KDEs for synthetic and holdout densities in the
  training-fitted mixed-type encoded QI space, split `real_test` at random into
  disjoint reference and control samples, use a common Scott-rule bandwidth
  based on the smaller synthetic/reference sample, and score with
  `log(p_synthetic / p_reference)`. Mark all results experimental.
- **Reason:** DOMIAS defines a membership score as a function of
  `p_G(x) / p_R(x)` (van Breugel et al. 2023, Section 3.2). The paper notes any
  density estimator may be used; the package needs an explicit practical KDE
  variant for mixed tables and must avoid a control row inflating its own
  reference density; scoring control rows against the same data used to fit
  that density otherwise creates a systematic self-density difference.
- **Alternatives considered:** Use synthetic density alone; leave the holdout
  control rows in their own reference KDE; present one encoding/bandwidth as
  canonical DOMIAS.
- **Status:** Experimental approximation; high dimensionality, encoding, and
  bandwidth sensitivity remain validation limitations.

## 2026-10-17: Keep singling-out and linkability attacks explicitly simplified

- **Decision:** Use every synthetic QI tuple as a singling-out guess and count
  success when exactly one row in train or holdout matches. For linkability,
  split the configured QI columns in declaration order and count a guess only
  when each block uniquely matches the same target row. Use the holdout rate
  as the control baseline; mark both attacks experimental.
- **Reason:** Giomi et al. (2023) define the main, naive, and control attack
  framework, with singling-out requiring exactly one matching record and
  linkability concerning links between records of the same individual. These
  exact-key attacks retain the core success checks but are not the paper's
  randomized predicate and auxiliary-record attack algorithms.
- **Alternatives considered:** Add Anonymeter as a runtime dependency; report a
  training success rate without a disjoint control; imply these variants are
  faithful reproductions.
- **Status:** Provisional simplified diagnostics. No dependency was added;
  external-table, randomized-guess, and naive-baseline attacks remain outside
  this implementation.

## 2026-10-17: Centralize M7 decisions and report exports

- **Decision:** Classify metrics only from configured policy thresholds, retain
  unconfigured metrics as review, and cap public-tier recommendations when the
  required holdout is absent. Use a single filtered aggregate payload for JSON
  and HTML; keep row profiles in memory and recompute them for `clean` rather
  than persisting row-level flags.
- **Reason:** The package is decision support, its configured threshold
  semantics must remain explicit, and reports/remediation must not expose
  row-level risk indicators or removed-row identities.
- **Alternatives considered:** Hard-code missing thresholds; serialize row
  flags to make remediation faster; write separate independently assembled
  JSON and HTML report data.
- **Status:** Accepted for M7. Decision confidence and recommendation are
  operational heuristics, not a guarantee of release safety.

## 2026-10-17: Add Typer, Jinja2, and PyArrow for M7

- **Decision:** Add Typer for the `sdrisk` CLI, Jinja2 for escaped HTML reports,
  and PyArrow to read and write the documented Parquet inputs.
- **Reason:** These are required by the public M7 command and output formats;
  using uv-managed runtime dependencies keeps the CLI reproducible.
- **Alternatives considered:** Hand-roll argument parsing and HTML escaping, or
  leave Parquet support as an undeclared optional environment assumption.
- **Status:** Accepted; dependencies are declared in `pyproject.toml` and
  locked by uv.

## 2026-10-17: Accept schema-normalized CSV inputs

- **Decision:** Allow `sdrisk run` and `clean` to read CSV as well as Parquet;
  normalize CSV columns according to the declared schema before aligning input
  dtypes.
- **Reason:** Real and generated files commonly infer numeric columns as
  different integer or floating-point dtypes even when they represent the same
  variable. Schema-driven normalization permits those files to be assessed
  without silently changing their analytic roles.
- **Alternatives considered:** Require callers to convert every file to Parquet
  or relax bundle dtype validation for all formats.
- **Status:** Accepted. Textual ordinal columns still require an ordered
  categorical representation and cannot be inferred from CSV alone.

## 2026-10-17: Warn on identical full-row values across real partitions

- **Decision:** Emit a warning rather than reject `real_train` and `real_test`
  when their full-row values overlap.
- **Reason:** Without stable source identifiers, identical observed attributes
  can belong to distinct people; a row-value collision alone cannot establish
  that the holdout contains a training record. Callers must confirm disjointness
  from the source split before interpreting holdout-controlled metrics.
- **Alternatives considered:** Reject every identical row pattern as proof of
  record overlap or silently accept the collision.
- **Status:** Accepted; the warning is explicit, but source-level disjointness
  remains the user's responsibility when identifiers are unavailable.

## 2026-10-17: Index subset equivalence classes in one pass

- **Decision:** Build row-position indexes for each input once when constructing
  subset partitions, then select each class from those indexes.
- **Reason:** Recomputing all QI keys for every candidate class caused repeated
  full-table scans and made the Adult-data pilot impractically slow.
- **Alternatives considered:** Retain repeated scans or cap the number of
  reported subset candidates, which could silently omit groups.
- **Status:** Accepted; this changes the indexing strategy, not the candidate
  subsets or their disclosure controls.

## 2026-10-17: Remove the public-data M8 benchmark from project scope

- **Decision:** Remove the bundled public benchmark runner and its dedicated
  fixture tests; retain the local data folders used to assess the user's
  approved datasets.
- **Reason:** The project will focus on assessments of the user's generator
  outputs rather than maintaining a separate general-purpose public-data
  validation suite.
- **Alternatives considered:** Keep the benchmark harness as an independent
  validation workstream or require its completion before documentation and
  release work.
- **Status:** Accepted. Metrics that depended on that validation, including
  CER, remain explicitly experimental and unvalidated.

## 2026-10-17: Use manual quality checks instead of GitHub Actions CI

- **Decision:** Remove the GitHub Actions workflow and leave code-quality
  checks to be run manually with the documented local commands.
- **Reason:** The project owner prefers manual commits and does not need an
  automated CI/CD pipeline.
- **Alternatives considered:** Keep checks running automatically on pushes and
  pull requests.
- **Status:** Accepted. The Python 3.10–3.12 test matrix is no longer run
  automatically; local lint, type, and test tools remain available.

## 2026-10-17: Do not ship an example release-policy profile

- **Decision:** Remove the placeholder public-tier YAML profile; require users
  to supply a policy profile appropriate to their release context.
- **Reason:** The example's thresholds, disclosure controls, and variable
  choices were unapproved and could be mistaken for a recommended default.
- **Alternatives considered:** Keep the example with warnings or replace it
  with another unconfirmed policy.
- **Status:** Accepted. YAML loading and validation remain supported and are
  covered by tests using temporary policy files.

## 2026-10-17: Aggregate subset report results

- **Decision:** Keep table and cluster metrics quantitative and per result, but
  summarize subset metrics by generic QI-set or stratum label. Report
  unweighted medians of values and baselines, output-filtered status counts,
  and the minimum-n suppressed-group count; do not export subset keys.
- **Reason:** Per-equivalence-class rows repeated generic labels and produced
  an unwieldy report without safely identifying actionable groups. Aggregated
  medians and status counts retain quantitative detail without exporting QI
  values; guard counts remain grouped by their safe configured label.
- **Alternatives considered:** Export every subset row with repeated metric
  explanations, or remove subset results entirely.
- **Status:** Accepted. Metric definitions and interpretation remain in the
  metric reference; the report displays subset medians to four decimal places.
