# Local benchmark data

This directory is for local benchmark inputs only. Git ignores its contents;
do not force-add data files or commit datasets or benchmark outputs.

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

CSV files are also acceptable if they contain the same columns. Keep the real
training and holdout files disjoint; synthetic files should have the same
analysis columns as the real files. You can add multiple named model or run
files under `synthetic/`.

Before benchmarking, provide the target column, any columns to exclude (such
as row identifiers), and a short mapping from each synthetic file's model/run
name to its generation settings. Confirm that the holdout was not used to train
the generators.

Only place data here if you are authorized to process it in this environment.
Git ignores these files to prevent accidental commits, but that does not make
them private or otherwise control access to this machine.
