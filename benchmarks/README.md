# Benchmarks

## M8 validation

The M8 harness in `src/sdrisk/benchmarking.py` uses scikit-learn's bundled
Wisconsin Diagnostic Breast Cancer data and generated planted-leak scenarios.
Its helper and fixture-generation tests are in
`tests/planted_leaks/test_m8_benchmark.py`. The full validation milestone is
not complete: detection and false-alarm rates, QI and distance-weight
sensitivity, and performance still need to be assessed and documented.

This is distinct from evaluating generator outputs on Adult data. The Adult
CSV files under [`local_data`](./local_data/README.md) support those
data-specific assessments; they do not replace the planted-leak validation.
For ordinary assessments, use the `sdrisk run` command documented in the
project [README](../README.md).

## Data handling

CSV inputs placed in `local_data` are Git-trackable, and committed files are
public because this repository is public. Commit only data approved for public
release and authorized for processing in this environment. Never commit
restricted data or unapproved generated data. Parquet inputs and generated
assessment outputs under `local_data/results/` are ignored by Git.
