# AGENTS.md: build `sdrisk`, a disclosure-risk evaluation package for synthetic tabular data

> Project brief and working rules for AI coding assistants (Claude Code, Copilot agent, Cursor, etc.).
> Items marked `TODO(confirm)` are policy decisions that a human must confirm. Never guess them.

---

## 0. How to work with this file (read first)

1. Read this whole file before writing any code.
2. Work **one milestone at a time** (section 14). Before coding a milestone, write a short plan (files to create, functions, tests) and wait for approval if a human is present.
3. **Tests first.** For every metric, write a hand-computable unit test before the implementation.
4. After each milestone run `ruff check`, `mypy`, and `pytest`. All must pass. Never edit a test just to make it pass; if a test is wrong, explain why.
5. When a definition is ambiguous (for example the exact synthpop definition of RepU or DiSCO), **read the cited paper or source, implement it faithfully, and cite it in the docstring.** If you cannot access the source, leave a `TODO(confirm)` and implement the simplest documented variant behind a clearly named function.
6. Record every non-trivial design decision in `docs/decisions.md` (date, decision, reason, alternatives).
7. Make small commits with clear messages (`feat(metrics): add exact match rate`, `test: add planted leak fixtures`).
8. Do not add dependencies without listing them in `docs/decisions.md` with a reason. Keep the dependency set small and optional where possible.

---

## 1. Project summary

`sdrisk` is a Python package that gives output checkers (for example a national statistics office release-check team) **decision support** on whether a synthetic dataset can be released. It runs a comprehensive privacy and disclosure risk check at four levels:

1. **Overall table**
2. **Subsets / groups** (user-defined and automatic)
3. **Machine-learning-based clusters**
4. **Each synthetic row** (internal only; see rule 3.3)

It is **not** a generator benchmark and it **cannot prove** a file is safe. Reports must say what the attacks tested, what they found, and what was not assessed.

Context: the tool is designed to run inside a restricted data environment (the real data cannot leave it), so its own outputs are a potential disclosure channel and must be treated as such.

---

## 2. Inputs and outputs

### Inputs

| Input | Required | Notes |
|---|---|---|
| `real_train` | **Yes** | The data the generator was trained on |
| `synthetic` | **Yes** | The file under review |
| `real_test` | Optional | Real records from the same population **not** used to train the generator. Acts as the holdout. Must be disjoint from `real_train` (validate; warn if suspicious) |
| `schema` | Yes | Variable types and roles (see 6) |
| `policy profile` | Yes | YAML: thresholds, QI sets, sensitive variables, strata, minimum cell sizes (see 10) |

### Outputs

- A `RunResult` containing `MetricResult` objects at table, subset and cluster level, plus an **internal** `RowStore`.
- A **disclosure-safe** JSON report and HTML report (aggregates only, after the output filter).
- An audit log (config hash, input file hashes, package version, seeds, sampling fractions, decision, rationale).
- Optionally, a **cleaned synthetic file** (rows above a chosen risk band removed) with **no** list of which rows were removed.

---

## 3. Hard rules (non-negotiable)

1. **Baseline everywhere.** Every risk number is reported next to a baseline (control attack, holdout comparison, or marginal-only guess). No raw number is ever reported alone.
2. **Single choke point for exports.** Every result passes through `output_filter` before any report or export. Metrics must not be able to bypass it.
3. **Row-level results are internal.** Row-level data lives in memory or a restricted internal directory. It is never written to a report, never logged, never printed, and never returned by the public report API. Reports may only contain **suppressed aggregates** (counts per risk band, banded distributions).
4. **No nearest-real-record displays.** Never output a real record, an index of a real record, or exact minimum/maximum values of continuous quantities.
5. **Policy is configuration, not code.** Thresholds, QI sets, sensitive variables, strata and minimum cell sizes come from YAML profiles. Do not hard-code them.
6. **Reproducible.** All randomness uses an explicit `numpy.random.Generator` seeded from config. Every run records the config hash, input hashes, package version and sampling fractions.
7. **Honest wording.** Use phrases like "these attacks found limited risk". Never use "safe", "anonymous", or "privacy-preserving" in reports.
8. **No real or restricted data in the repo.** Develop and test with public datasets only (for example the Adult census dataset) and synthetic fixtures generated in tests.
9. **Not-assessed is explicit.** If a metric cannot run (missing input, too few records), return `status='not_assessed'` with a reason. Never skip silently.

---

## 4. Tech stack and conventions

- Python 3.10+ (`TODO(confirm)` against the target environment). `src/` layout, `pyproject.toml`, installable with `pip install -e .`.
- Core: `numpy`, `pandas`, `scipy`, `scikit-learn`, `pyarrow`.
- Config/validation: `pydantic` (v2), `pyyaml`.
- CLI: `typer`. Reports: `jinja2`.
- Dev: `pytest`, `pytest-cov`, `hypothesis`, `ruff`, `mypy` (strict on `src/`), `pre-commit`.
- Optional extras (only behind `pip install sdrisk[extra]`; must degrade gracefully if absent): `anonymeter`, `domias`, `hdbscan`/sklearn HDBSCAN, an approximate nearest-neighbour library.
- Type hints everywhere; docstrings in NumPy style; each metric docstring must state: definition, inputs, baseline, interpretation, known weaknesses, citation.
- Pure functions where possible; no global state; no network access at runtime.

---

## 5. Repository layout

```
sdrisk/
  AGENTS.md
  pyproject.toml
  README.md
  CHANGELOG.md
  docs/
    decisions.md
    metric_reference.md
    limitations.md
    checker_guide.md
  src/sdrisk/
    schema.py               # variable types and roles
    bundle.py               # DatasetBundle: real_train, synthetic, real_test (optional)
    policy.py               # load and validate YAML profiles
    results.py              # MetricResult, RowStore, RunResult
    output_filter.py        # suppression, rounding, secondary suppression
    provenance.py           # hashes, seeds, versions, run log
    applicability.py        # which metrics can run given the inputs
    prep/                   # encoders, scalers, discretisation, missing values
    distance/               # gower.py, nn_search.py, blocking.py, subsample.py
    metrics/
      base.py               # Metric protocol and registry
      identity/             # exact_match, uem, repu, kprofile, dcr
      attribute/            # cap, disco, ml_inference, numeric_tolerance, homogeneity
      membership/           # mia_distance, mia_density
      attacks/              # singling_out, linkability
      derived_tables.py
    levels/
      table.py
      subset.py             # strata + QI equivalence classes
      cluster.py            # ClusterModel, cluster exposure ratio, per-cluster metrics
      row.py                # RowProfile construction (internal)
    decision.py             # thresholds -> pass/review/fail, confidence, recommendation
    remediation.py          # drop rows above a band without exporting flags
    report/
      json_report.py
      html_report.py
      templates/
    cli.py
  tests/
    unit/
    property/
    planted_leaks/
    fixtures/
  benchmarks/               # validation on public data
```

---

## 6. Core data model

```python
# schema.py
Role = Literal['qi', 'sensitive', 'stratum', 'id', 'ignore']
VarType = Literal['categorical', 'ordinal', 'numeric', 'date']

class Variable(BaseModel):
    name: str
    vtype: VarType
    role: list[Role] = []          # a variable may be both qi and stratum
    weight: float = 1.0            # distance weight
    tolerance: float | None = None # numeric tolerance for match/disclosure

class Schema(BaseModel):
    variables: list[Variable]
```

```python
# bundle.py
@dataclass(frozen=True)
class DatasetBundle:
    real_train: pd.DataFrame
    synthetic: pd.DataFrame
    real_test: pd.DataFrame | None = None
    # validate: same columns/dtypes; no id columns used as features;
    # train/test disjoint; minimum sizes; warn if test is suspiciously close to train
```

```python
# results.py
Status = Literal['pass', 'review', 'fail', 'suppressed', 'not_assessed']
Level = Literal['table', 'subset', 'cluster']

@dataclass(frozen=True)
class MetricResult:
    name: str
    level: Level
    group: str | None            # stratum or cluster label
    value: float | None
    baseline: float | None
    ci: tuple[float, float] | None
    n_records: int
    n_flagged: int | None
    status: Status
    reason: str = ''             # required when not_assessed or suppressed
    experimental: bool = False   # True for candidate/unvalidated metrics

class RowStore:
    """INTERNAL ONLY. Holds per-synthetic-row profiles.
    No public method may serialise row-level values. Only aggregated,
    filter-compliant summaries are exposed (counts per band, banded distributions)."""
```

```python
# metrics/base.py
class Metric(Protocol):
    name: str
    requires: frozenset[str]   # subset of {'real_train','synthetic','real_test'}
    def compute(self, bundle, schema, profile, rng, row_store) -> list[MetricResult]: ...
```

The runner: validate -> check applicability -> compute metrics -> (optionally) strata and clusters -> build row profiles -> **output filter** -> decision -> report.

---

## 7. Metrics catalogue

For every metric: implement the table-level version first, then subset and cluster versions via the shared hooks in `levels/`. State the baseline. Add confidence intervals where possible (Wilson for proportions, bootstrap otherwise).

### 7.1 Identity disclosure

| Metric | Requires | Computation | Baseline |
|---|---|---|---|
| Exact match rate | train | Share of `real_train` records with an identical synthetic record on a variable set (hash join on discretised values, tolerance configurable) | Same measure between `real_train` and `real_test` if available |
| Unique exact match (UEM) | train | Among real records unique on the variable set, share with an exact synthetic match | Test-set equivalent |
| UiO / RepU | train | UiO: percent of real records unique on the QI set. RepU: replicated uniques per the synthpop definition (**read Raab, Nowok, Dibben, "Practical privacy metrics for synthetic data" before implementing**) | UiO |
| k-profile | train | Equivalence-class sizes on QI sets in real and synthetic; minimum k; share of records with k below thresholds (2, 5, 10) | Real-data profile |
| DCR overfitting | train, **test** | For each synthetic row, distance to nearest train row and nearest test row; share closer to train | 0.5 expected with no overfitting |
| DCR vs random | train | Median nearest-train distance of synthetic divided by that of random data drawn from marginals; guard against zero median (return NaN and report components) | Random data |
| Singling out / linkability | train (control: test) | Anonymeter-style attacks; wrap `anonymeter` if installed, otherwise implement a simplified version and document differences | Control set |

### 7.2 Attribute disclosure

| Metric | Requires | Computation | Baseline |
|---|---|---|---|
| CAP / TCAP | train | For each real record, match synthetic rows on key (QI) values and check how often they carry the true target value; TCAP restricts to classes where all matching synthetic rows share one target value | Guess from the target's marginal (or within-class real) distribution |
| DiSCO | train | Per synthpop definition (verify from source) | Original-data equivalent |
| ML inference gain | train (**test** preferred) | Train a classifier on synthetic (keys to target); compare accuracy/AUC on train members versus test non-members | Gap near zero; marginal-only baseline. Without test: cross-validated model on real data (`experimental=True`) |
| Numeric tolerance | train | Share of real numeric targets recovered within tolerance | Marginal guess |
| Homogeneity | synthetic | Share of synthetic QI classes (above a minimum size) where every record has the same sensitive value | Real-data equivalent |

### 7.3 Membership disclosure (needs `real_test`)

| Metric | Computation |
|---|---|
| Distance-based MIA | Score = negative distance to nearest synthetic record; compare train members and test non-members; report AUC, attacker advantage, TPR at 1% and 0.1% FPR with bootstrap CIs |
| Density-ratio MIA (DOMIAS-style) | KDE estimate of synthetic density divided by a reference density from `real_test`; report as above; document dimensionality limits |

Without `real_test`, return `not_assessed` with reason `'no holdout supplied'`.

### 7.4 Derived-table checks

Apply the output rules to standard cross-tabs built from the synthetic file, and compare with the real tables.

---

## 8. The four levels

### 8.1 Table level
Headline rates for every applicable metric with baselines and CIs.

### 8.2 Subset level
- **User-defined strata** from the profile (for example ethnicity, region, age band) and optional intersections.
- **Automatic subsets:** QI equivalence classes and rare categories.
- Every subset has a **minimum-n guard**: below the profile's minimum size, return `suppressed`.
- Record the number of subsets tested; report a multiple-comparison caution and, where used, a correction method.

### 8.3 Cluster level (ML-based)
`ClusterModel`:
- Fit on `real_train` in an encoded space (default: scaled numeric plus one-hot or ordinal categorical, `MiniBatchKMeans` with configurable k; optional HDBSCAN on a subsample; optional k-prototypes if allowed).
- Assign `synthetic` and (if present) `real_test` rows to the fitted clusters.
- **Stability check:** repeat with several seeds and report agreement (adjusted Rand index). Unstable clusterings are flagged, not hidden.
- **Minimum cluster size** from the profile; smaller clusters are merged into an `other` bucket or suppressed.

Per cluster, report: size, each applicable Tier 1 metric, local distance-to-closest-record distribution (banded), and the **cluster exposure ratio**.

> **Cluster exposure ratio (CER), `experimental=True`.**
> `CER_c = share_synthetic(c) / share_train(c)`.
> With `real_test`, compare against the same ratio for the test set, which should be near 1 for an independent sample. Over-representation of a cluster in the synthetic data beyond that expectation may indicate copying or local overfitting.
> This is a candidate diagnostic and not an established metric. Always set `experimental=True` and describe it as such in reports; no validation benchmark is currently in scope.

### 8.4 Row level (internal)
For each synthetic row, build a `RowProfile` stored only in `RowStore`:

- `dcr_train`, and its percentile against a reference distribution
- `exact_match`, `near_match` (distance below a profile threshold)
- `replicated_unique`
- `k_syn`, `k_real` on each QI set
- per-sensitive-variable attribute-disclosure flag
- `local_density_ratio` (only if `real_test` is available)
- `cluster_id`, `cluster_exposure_ratio`
- `composite_band` in {low, medium, high}, computed from the profile's rules (not hard-coded)

Rules:
- Only aggregates leave `RowStore`: counts per band, banded distributions, all passed through `output_filter`.
- `remediation.py` may drop rows above a chosen band and write a cleaned file, but must **not** write or return which rows were dropped.
- Add a test that scans every exported object (JSON, HTML, logs) for row-level fields.

---

## 9. Output filter specification

`output_filter.py` is the only module that decides what can leave the package. Requirements:

- Suppress any count below `min_cell_size` (`TODO(confirm)`; placeholder 6).
- Round counts to `rounding_base` (`TODO(confirm)`; placeholder 3).
- Suppress derived percentages, ratios and rates whenever a contributing count is suppressed.
- Apply secondary suppression where a suppressed value could be recovered by subtraction from totals.
- Never report exact minimum or maximum of continuous values; report banded percentiles only.
- Enforce minimum-n for subsets and clusters.
- Return `MetricResult` objects with `status='suppressed'` and a `reason`; never silently drop them.
- **Property-based tests (hypothesis):** random inputs must never produce an unsuppressed small count, an exact extreme value, or a row-level field in any exported object.

---

## 10. Policy profile example (public-safe placeholders)

```yaml
# example_policy.yaml
name: example_policy
release_tier: public            # in_lab | licensed | public
output:
  min_cell_size: 6              # TODO(confirm)
  rounding_base: 3              # TODO(confirm)
variables:
  qi_sets:
    - name: demo_core
      columns: [age_band, sex, region]
    - name: demo_extended
      columns: [age_band, sex, region, ethnicity]
  sensitive: [income_band, health_flag]
  strata: [ethnicity, region, age_band]
distance:
  max_rows_exact: 200000
  subsample_rows: 50000
  near_match_threshold: 0.05    # TODO(confirm)
clusters:
  k: 20
  min_cluster_size: 50
  seeds: 5
thresholds:                     # all TODO(confirm); placeholders only
  exact_match_rate: {review: 0.001, fail: 0.01}
  repu_pct:         {review: 0.5,   fail: 2.0}
  disco_gain:       {review: 0.02,  fail: 0.05}
  mia_auc:          {review: 0.55,  fail: 0.65}
  dcr_overfit_share:{review: 0.55,  fail: 0.65}
policy:
  require_test_for_public_tier: true   # caps decision at 'review' without real_test
```

---

## 11. Decision layer and report

- Map each metric result to `pass`, `review` or `fail` using the profile thresholds, and compute an overall recommendation: **Release**, **Release with modifications**, or **Refuse**. The checker decides; the tool recommends.
- Compute a **confidence level** (high/medium/low) from what was assessed. Missing `real_test`, heavy subsampling, unstable clusters, or many `not_assessed` metrics lower confidence. Say so in the report.
- Each metric has a plain-language note: what it measures, what a fail means, what to ask the submitter.
- Report sections: inputs and provenance, applicability (what was and was not assessed), table-level results, subset results, cluster results, row-band summary (aggregates only), recommendation and confidence, limitations, decision/rationale fields for the checker.
- Required limitations text: "Passing these checks lowers concern but does not prove the data cannot be disclosive. Thresholds are policy choices. Attacks requiring generator access were not run."

---

## 12. CLI

```
sdrisk run \
  --profile policy.yaml \
  --schema schema.yaml \
  --real-train train.parquet \
  --synthetic synthetic.parquet \
  [--real-test test.parquet] \
  --out-dir reports/run_001

sdrisk clean --run reports/run_001 --max-band medium --out cleaned_synthetic.parquet
```

`sdrisk clean` must not output the list of removed rows.

---

## 13. Testing strategy

### 13.1 Unit tests
Tiny hand-built datasets with known exact-match rate, UEM, DCR, CAP, k-profile values. Include edge cases: all-unique data, constant columns, missing values, a zero-median random baseline, empty strata.

### 13.2 Planted-leak fixtures (generated in tests; public data only)
| Fixture | How built | Expected |
|---|---|---|
| `independent` | Independent sample from the same population | Metrics near baseline; low false alarms |
| `exact_copy_x` | Copy x% of real rows | Exact match, UEM, RepU, DCR, MIA all respond monotonically in x |
| `noisy_copy` | Copy rows with small noise | DCR and MIA respond; exact match does not |
| `overfit` | A simple resampling or kNN generator | DCR overfitting and MIA respond |
| `attribute_leak` | Deterministic sensitive value within QI classes | CAP/TCAP, DiSCO, homogeneity respond |
| `localized_leak` | Copy rows from only one small cluster or subgroup | Table-level metrics change little; **subset and cluster level must catch it** |
| `heavy_noise` | Strongly noised file | Risk low; utility not our concern |

### 13.3 Other tests
- Property-based tests for the output filter.
- Determinism: same inputs, config and seed give identical results.
- Export scan: no row-level fields in any JSON, HTML or log.
- Performance tests at 10^4, 10^5 and 10^6 rows; record runtime and memory.
- Applicability tests: each metric returns `not_assessed` when its inputs are missing.

---

## 14. Milestones (give each to the agent as a separate task)

### M0. Setup
Create repo skeleton, `pyproject.toml`, ruff, mypy (strict), pytest, pre-commit, licence placeholder, README, CHANGELOG.
**Done when:** a clean install runs an empty test suite and local quality checks pass.

### M1. Core scaffolding
`schema.py`, `bundle.py` (with `real_test` optional and disjointness validation), `policy.py` (pydantic plus YAML), `results.py`, `metrics/base.py` registry, `applicability.py`, `provenance.py`, `output_filter.py`.
**Done when:** invalid inputs fail with clear messages; property-based tests show nothing unfiltered reaches an export; `not_assessed` works end to end.

### M2. Preprocessing and distance engine
Encoders and scalers for mixed types, missing-value handling, Gower distance, nearest-neighbour search with chunking and blocking, optional approximate search, subsampling with recorded fraction.
**Done when:** results match a brute-force reference on small data; runtime and memory documented for 10^5 and 10^6 rows.

### M3. Identity metrics
Exact match, UEM, UiO/RepU, k-profile, DCR overfitting, DCR vs random.
**Done when:** each flags the planted-leak fixtures and stays quiet on `independent`.

### M4. Attribute metrics
CAP/TCAP, DiSCO, ML inference gain, numeric tolerance, homogeneity.
**Done when:** results are reproducible, match published examples where available, and respond to `attribute_leak`.

### M5. Subset, cluster and row levels
`levels/subset.py`, `levels/cluster.py` (ClusterModel with stability, cluster exposure ratio), `levels/row.py`, `RowStore`, composite band, aggregated row-band summaries.
**Done when:** `localized_leak` is detected at subset or cluster level; the export-scan test proves no row-level leakage.

### M6. Membership and attack metrics
Distance-based MIA, density-ratio MIA, singling out, linkability (wrapper or simplified implementation).
**Done when:** attacks respond to planted leaks and sit near chance on `independent`; `not_assessed` without `real_test`.

### M7. Decision, remediation, reporting, CLI
`decision.py` (status, confidence, recommendation), `remediation.py`, JSON and HTML reports, CLI.
**Done when:** a non-expert can read a report and say what to do next; `sdrisk clean` works without exporting flags.

### M9. Documentation and release
`metric_reference.md`, `limitations.md`, `checker_guide.md`, developer guide, release v0.1.0.
The full public-data validation benchmark formerly planned as M8 is out of
scope. Experimental diagnostics, including CER, remain marked as unvalidated.

---

## 15. Definition of done (v0.1)

- All identity and attribute metrics implemented at table and subset level with baselines and CIs.
- Membership metrics implemented and correctly `not_assessed` without `real_test`.
- Cluster-level results with stability checks; CER marked experimental unless validated.
- Row-level profiles computed internally, with a passing export-scan test.
- Output filter covers every export path, with property-based tests.
- User-supplied policy profiles for supported release tiers load from YAML.
- JSON and HTML reports with confidence level, applicability section and limitations.
- Lint, types, and tests pass locally; docs complete.

---

## 16. Things the agent must NOT do

- Do not invent definitions of published metrics; read the source or mark `TODO(confirm)`.
- Do not hard-code thresholds, QI sets, minimum cell sizes or rounding rules.
- Do not print, log or serialise row-level data, nearest real records, or exact extremes.
- Do not bypass `output_filter`.
- Do not describe results as proving safety or anonymity.
- Do not present experimental metrics as established.
- Do not use real or restricted data; use public datasets and generated fixtures only.
- Do not wrap a third-party library as the only implementation of a core metric; wrappers are optional extras.
- Do not modify tests to make failing code pass.

---

## 17. Open questions (leave `TODO(confirm)`, do not guess)

1. Exact Stats NZ (or target agency) output rules: minimum cell size, rounding base, percentage and secondary-suppression rules.
2. Default QI sets, sensitive variables and strata for the first profile.
3. Thresholds for pass/review/fail per release tier.
4. Allowed Python version and packages in the target environment.
5. Whether a holdout is always available, and the policy when it is not.
6. Typical file sizes, to set performance targets.
7. Licence, ownership, and the approval route for open-sourcing and publication.

---

## 18. References (read before implementing the relevant metric)

- Raab, Nowok, Dibben. *Practical privacy metrics for synthetic data* (synthpop disclosure functions: RepU, DiSCO), arXiv 2406.16826.
- Raab. *Privacy risk from synthetic data: practical proposals*, arXiv 2409.04257.
- Giomi et al. *A Unified Framework for Quantifying Privacy Risk in Synthetic Data* (Anonymeter), arXiv 2211.10459.
- van Breugel et al. *Membership Inference Attacks against Synthetic Data through Overfitting Detection* (DOMIAS), arXiv 2302.12580.
- TAPAS documentation: tapas-privacy.readthedocs.io.
- NIST Collaborative Research Cycle and SDNist (unique exact match and related metrics).
- SDMetrics privacy metrics: docs.sdv.dev/sdmetrics.
- Yu. *Synthetic Data Disclosure Risk Assessment: A Literature Review*, Statistics Canada, 2025.
