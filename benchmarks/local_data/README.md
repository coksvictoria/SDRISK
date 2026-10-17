# Local benchmark data

CSV files in this directory are Git-trackable; committed files will be public
in this repository. Do not place data here unless it is approved for public
release and authorized for processing in this environment. Benchmark outputs
under `results/` and Parquet files are ignored by Git.

Place files in this structure:

```text
benchmarks/local_data/
  real/
    real.csv
    holdout.csv
  synthetic/
    <model-name>.csv
  results/                  # local profiles, samples, and reports; Git-ignored
```

CSV files should have matching analysis columns. Keep the real training and
holdout files disjoint; synthetic files should have the same analysis columns
as the real files. Add one named CSV per model or run under `synthetic/`.

Before benchmarking, provide the target column, any columns to exclude (such
as row identifiers), and a short mapping from each synthetic file's model/run
name to its generation settings. Confirm that the holdout was not used to train
the generators.

Git retains committed CSVs in repository history even if a file is later
deleted. Only place data here if it is approved for public sharing and you are
authorized to process it in this environment.
