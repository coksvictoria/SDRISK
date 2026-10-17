# Changelog

All notable changes to this project will be documented here.

The format is based on Keep a Changelog, and the project intends to follow Semantic
Versioning once a release policy is confirmed.

## [Unreleased]

### Added

- Initial M0 repository scaffold.
- Packaging, linting, strict type checking, tests, pre-commit, and CI configuration.
- uv-managed project environment and lockfile.
- M1 core schemas, validated dataset bundles and policy profiles, metric applicability,
  provenance helpers, result models, and disclosure-safe output filtering.
- M2 mixed-type preprocessing, weighted Gower dissimilarity, chunked and blocked
  nearest-neighbor search, and reproducible subsampling.
- M3 table-level exact match, unique exact match, UiO/RepU, k-profile, and DCR identity
  metrics with baselines, confidence intervals where applicable, and output filtering.
- M4 table-level CAP/TCAP, DiSCO, ML inference gain, numeric-tolerance recovery, and
  homogeneity metrics with configured baselines and output filtering.
- M5 generic-label subset analysis, seeded MiniBatchKMeans cluster stability and
  experimental cluster exposure ratio, plus internal RowStore profiles with
  output-filtered risk-band aggregates.
- M6 holdout-controlled distance and experimental KDE density-ratio membership
  metrics, plus documented experimental exact-key singling-out and linkability
  attacks.
- M7 policy-threshold decision support, confidence and recommendation mapping,
  in-memory row-band remediation, output-filtered JSON/HTML reports, provenance
  audit logs, and the `sdrisk run` / `sdrisk clean` Typer CLI.
