# sdrisk

`sdrisk` is a Python package providing decision support for disclosure-risk
assessment of synthetic tabular data.

> Passing checks lowers concern but does not prove that data cannot be disclosive.
> Thresholds are policy choices, and attacks requiring generator access may not be run.

## Project status

Milestones **M1: core scaffolding**, **M2: preprocessing and distance engine**,
**M3: identity metrics**, **M4: attribute metrics**, and **M5: subset, cluster,
and row levels**, **M6: membership and attack metrics**, and **M7: decision,
remediation, reporting, and CLI** are implemented.
The package provides table and subgroup metrics, seeded cluster diagnostics,
internal row profiles whose only exportable summaries pass through the output
filter, holdout-controlled membership and simplified attack diagnostics,
policy-based recommendations, and aggregate-only JSON/HTML reports. It is
decision support, not a release approval system. Thresholds and example profile
values remain policy placeholders; the candidate metrics and simplified attacks
are not fully validated. Do not interpret a recommendation as proof that data
are safe or anonymous.

## Development setup

```bash
uv sync --group dev
uv run pre-commit install
uv run ruff check .
uv run mypy src
uv run pytest
```

`pyproject.toml` is the source of package metadata and dependency declarations;
`uv.lock` pins the resolved environment. Use `uv sync --locked --group dev` to
recreate the development environment without changing the lock file. uv manages
the environment and lock; setuptools remains the package build backend.

## Run an assessment

Inputs may be CSV or Parquet dataframes, plus a YAML schema and policy profile.
CSV columns are normalized according to the schema types:

```powershell
uv run sdrisk run `
  --profile profiles/example_public_tier.yaml `
  --schema schema.yaml `
  --real-train train.parquet `
  --synthetic synthetic.parquet `
  --real-test test.parquet `
  --out-dir reports/run_001 `
  --seed 42
```

The run directory contains `report.json`, `report.html`, and `audit.json`.
Reports contain filtered aggregates only. `run_manifest.json` is operational
state for the `clean` command; it contains source paths and hashes, so keep the
run directory in the restricted environment.

To remove synthetic rows above a selected risk band, rerun the same assessment
inputs using the manifest and write only the cleaned dataset:

```powershell
uv run sdrisk clean --run reports/run_001 --max-band medium --out cleaned.parquet
```

Cleaning recomputes internal row profiles and refuses to run if source files or
configuration have changed. It does not report which rows were removed or how
many. The output dataset itself should be handled under the same controls as the
input synthetic file.

## Repository policy

Read [`AGENTS.md`](AGENTS.md) before making changes. Work one milestone at a time,
write hand-computable tests before metric implementations, and keep all report
results behind the output filter.

## Roadmap

The full public-data validation benchmark (M8) is out of scope. The next
planned milestone is **M9: documentation and release**. Experimental metrics
remain explicitly unvalidated; see [`docs/limitations.md`](docs/limitations.md).
See `AGENTS.md`, section 14, for the project milestones.
Distance-engine behavior is described in [`docs/distance_engine.md`](docs/distance_engine.md);
identity and attribute metric definitions are in
[`docs/metric_reference.md`](docs/metric_reference.md).
See [`docs/checker_guide.md`](docs/checker_guide.md) for report interpretation and
operational cautions.

## Licence

The licence is intentionally unresolved. See `LICENSE` and `docs/decisions.md`.
