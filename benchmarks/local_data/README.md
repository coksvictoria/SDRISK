# Local benchmark data

CSV files in this directory are tracked by Git and will be public in this
repository. Do not place data here unless it is approved for public release.
Benchmark outputs under `results/` and Parquet files are ignored by Git.

Place files in this structure:

```text
benchmarks/local_data/
  real/
    train.parquet
    holdout.parquet
  synthetic/
    <model-name>/
      <run-name>.parquet
```

Use CSV files with matching analysis columns. Keep the real training and
holdout files disjoint; synthetic files should have the same analysis columns
as the real files. You can add multiple named model or run files under
`synthetic/`.

Before benchmarking, provide the target column, any columns to exclude (such
as row identifiers), and a short mapping from each synthetic file's model/run
name to its generation settings. Confirm that the holdout was not used to train
the generators.

Git retains committed CSVs in repository history even if a file is later
deleted. Only place data here if it is approved for public sharing and you are
authorized to process it in this environment.
