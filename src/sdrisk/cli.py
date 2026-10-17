"""Command-line entry points for assessment and row-band remediation."""

import json
from pathlib import Path
from typing import Annotated, Any

import pandas as pd
import typer
import yaml

from sdrisk.bundle import DatasetBundle
from sdrisk.policy import load_policy_profile
from sdrisk.provenance import config_sha256, file_sha256
from sdrisk.remediation import remove_above_band
from sdrisk.report.html_report import write_html_report
from sdrisk.report.json_report import write_audit_log, write_json_report
from sdrisk.runner import run_assessment
from sdrisk.schema import Schema

app = typer.Typer(
    name="sdrisk",
    help="Assess disclosure risk in synthetic tabular data.",
    no_args_is_help=True,
    add_completion=False,
)


@app.command("run")
def run_command(
    profile: Annotated[Path, typer.Option("--profile", help="YAML policy profile.")],
    schema: Annotated[Path, typer.Option("--schema", help="YAML variable schema.")],
    real_train: Annotated[
        Path, typer.Option("--real-train", help="Training data in Parquet.")
    ],
    synthetic: Annotated[
        Path, typer.Option("--synthetic", help="Synthetic data in Parquet.")
    ],
    out_dir: Annotated[
        Path, typer.Option("--out-dir", help="Directory for aggregate reports.")
    ],
    real_test: Annotated[
        Path | None,
        typer.Option("--real-test", help="Optional disjoint holdout data in Parquet."),
    ] = None,
    seed: Annotated[
        int, typer.Option("--seed", min=0, help="Seed for all stochastic metrics.")
    ] = 0,
) -> None:
    """Run the configured assessment and write aggregate-only reports."""
    policy = load_policy_profile(profile)
    schema_model = _load_schema(schema)
    train_data = pd.read_parquet(real_train)
    synthetic_data = pd.read_parquet(synthetic)
    test_data = pd.read_parquet(real_test) if real_test is not None else None
    bundle = DatasetBundle(train_data, synthetic_data, test_data)
    input_files = {
        "real_train": real_train,
        "synthetic": synthetic,
    }
    if real_test is not None:
        input_files["real_test"] = real_test
    result = run_assessment(
        bundle,
        schema_model,
        policy,
        seed=seed,
        input_files=input_files,
    )

    out_dir.mkdir(parents=True, exist_ok=True)
    write_json_report(result, out_dir / "report.json")
    write_html_report(result, out_dir / "report.html")
    write_audit_log(result, out_dir / "audit.json")
    manifest: dict[str, Any] = {
        "profile_path": str(profile.resolve()),
        "schema_path": str(schema.resolve()),
        "input_paths": {name: str(path.resolve()) for name, path in input_files.items()},
        "input_hashes": dict(result.provenance.input_hashes),
        "config_hash": result.provenance.config_hash,
        "seed": seed,
    }
    (out_dir / "run_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    typer.echo(f"Reports written to {out_dir.resolve()}")


@app.command("clean")
def clean_command(
    run: Annotated[
        Path, typer.Option("--run", help="Directory created by `sdrisk run`.")
    ],
    out: Annotated[Path, typer.Option("--out", help="Output Parquet file.")],
    max_band: Annotated[
        str, typer.Option("--max-band", help="Keep rows in this band or a lower-risk band.")
    ] = "medium",
) -> None:
    """Write cleaned synthetic data without exporting removed-row details."""
    manifest_path = run / "run_manifest.json"
    manifest = _read_manifest(manifest_path)
    input_paths = {
        name: Path(path) for name, path in manifest["input_paths"].items()
    }
    expected_names = {"real_train", "synthetic"}
    if not expected_names <= set(input_paths):
        raise ValueError("run manifest is missing required input paths")
    for name, path in input_paths.items():
        if not path.is_file():
            raise FileNotFoundError(f"input file recorded for {name!r} no longer exists: {path}")
        expected_hash = manifest["input_hashes"].get(name)
        if expected_hash != file_sha256(path):
            raise ValueError(f"input file for {name!r} differs from the assessed run")

    policy_path = Path(manifest["profile_path"])
    schema_path = Path(manifest["schema_path"])
    policy = load_policy_profile(policy_path)
    schema_model = _load_schema(schema_path)
    config_hash = config_sha256(
        {
            "profile": policy.model_dump(mode="json"),
            "schema": schema_model.model_dump(mode="json"),
        }
    )
    if config_hash != manifest["config_hash"]:
        raise ValueError("policy profile or schema differs from the assessed run")

    train_data = pd.read_parquet(input_paths["real_train"])
    synthetic_data = pd.read_parquet(input_paths["synthetic"])
    test_path = input_paths.get("real_test")
    test_data = pd.read_parquet(test_path) if test_path is not None else None
    bundle = DatasetBundle(train_data, synthetic_data, test_data)
    result = run_assessment(
        bundle,
        schema_model,
        policy,
        seed=manifest["seed"],
        input_files=input_paths,
    )
    if result.row_store is None:
        raise ValueError(f"row-level remediation is unavailable: {result.row_reason}")
    cleaned = remove_above_band(
        synthetic_data,
        result.row_store,
        max_band=max_band,
    )
    out.parent.mkdir(parents=True, exist_ok=True)
    cleaned.to_parquet(out, index=False)
    typer.echo(f"Cleaned synthetic data written to {out.resolve()}")


def main() -> None:
    """Run the Typer application."""
    app()


def _load_schema(path: Path) -> Schema:
    with path.open("r", encoding="utf-8") as stream:
        try:
            raw: Any = yaml.safe_load(stream)
        except yaml.YAMLError as error:
            raise ValueError(f"Invalid schema YAML in {path}: {error}") from error
    return Schema.model_validate(raw)


def _read_manifest(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as stream:
        data: Any = json.load(stream)
    if not isinstance(data, dict):
        raise ValueError("run manifest must be a JSON object")
    required = {
        "profile_path",
        "schema_path",
        "input_paths",
        "input_hashes",
        "config_hash",
        "seed",
    }
    missing = required - set(data)
    if missing:
        raise ValueError(f"run manifest is missing fields: {', '.join(sorted(missing))}")
    if not isinstance(data["input_paths"], dict) or not isinstance(data["input_hashes"], dict):
        raise ValueError("run manifest input paths and hashes must be objects")
    return data


if __name__ == "__main__":
    main()
