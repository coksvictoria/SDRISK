"""End-to-end CLI tests for safe reports and remediation."""

import json
from pathlib import Path

import pandas as pd
import pytest
import yaml
from typer.testing import CliRunner

from sdrisk.cli import app


@pytest.mark.parametrize("extension", [".csv", ".parquet"])
def test_run_and_clean_write_reports_without_row_flags_or_removal_lists(
    tmp_path: Path, extension: str
) -> None:
    train = pd.DataFrame(
        {
            "key": [f"key-{index}" for index in range(12)],
            "value": list(range(12)),
            "secret": [f"secret-{index % 3}" for index in range(12)],
        }
    )
    test = train.assign(key=[f"holdout-{index}" for index in range(12)])
    synthetic = (
        train.astype({"value": "float64"}) if extension == ".csv" else train
    )
    train_path = tmp_path / f"train{extension}"
    synthetic_path = tmp_path / f"synthetic{extension}"
    test_path = tmp_path / f"test{extension}"
    _write_table(train, train_path)
    _write_table(synthetic, synthetic_path)
    _write_table(test, test_path)

    schema_path = tmp_path / "schema.yaml"
    schema_path.write_text(
        yaml.safe_dump(
            {
                "variables": [
                    {"name": "key", "vtype": "categorical", "role": ["qi"]},
                    {"name": "value", "vtype": "numeric", "role": ["qi"]},
                    {"name": "secret", "vtype": "categorical", "role": ["sensitive"]},
                ]
            }
        ),
        encoding="utf-8",
    )
    profile_path = tmp_path / "profile.yaml"
    profile_path.write_text(
        yaml.safe_dump(
            {
                "name": "cli-test",
                "release_tier": "in_lab",
                "output": {"min_cell_size": 2, "rounding_base": 1},
                "variables": {
                    "qi_sets": [{"name": "core", "columns": ["key", "value"]}],
                    "sensitive": ["secret"],
                    "strata": [],
                },
                "identity": {"k_profile_thresholds": [2]},
                "row_risk": {"medium_min_flags": 1, "high_min_flags": 2},
                "distance": {
                    "max_rows_exact": 100,
                    "subsample_rows": 100,
                    "near_match_threshold": 0.05,
                },
                "clusters": {"k": 2, "min_cluster_size": 2, "seeds": 2},
                "thresholds": {
                    "exact_match_rate": {"review": 0.01, "fail": 0.1}
                },
                "policy": {"require_test_for_public_tier": True},
            }
        ),
        encoding="utf-8",
    )
    run_dir = tmp_path / "run"
    runner = CliRunner()
    run_result = runner.invoke(
        app,
        [
            "run",
            "--profile",
            str(profile_path),
            "--schema",
            str(schema_path),
            "--real-train",
            str(train_path),
            "--synthetic",
            str(synthetic_path),
            "--real-test",
            str(test_path),
            "--out-dir",
            str(run_dir),
            "--seed",
            "7",
        ],
    )

    assert run_result.exit_code == 0, run_result.output
    exported_report_text = "\n".join(
        (run_dir / filename).read_text(encoding="utf-8").lower()
        for filename in (
            "report.json",
            "report.html",
            "audit.json",
            "run_manifest.json",
        )
    )
    assert "dcr_train" not in exported_report_text
    assert "row_profiles" not in exported_report_text
    assert "key-0" not in exported_report_text
    report_text = (run_dir / "report.json").read_text(encoding="utf-8").lower()
    assert json.loads(report_text)["provenance"]["seed"] == 7
    assert json.loads(report_text)["decision"]["checker_decision"] is None
    assert (run_dir / "report.html").exists()
    assert (run_dir / "audit.json").exists()

    output_path = tmp_path / "cleaned.parquet"
    clean_result = runner.invoke(
        app,
        [
            "clean",
            "--run",
            str(run_dir),
            "--max-band",
            "medium",
            "--out",
            str(output_path),
        ],
    )

    assert clean_result.exit_code == 0, clean_result.output
    assert output_path.exists()
    assert "key-0" not in clean_result.output
    assert "removed" not in clean_result.output.lower()
    assert pd.read_parquet(output_path).empty


def _write_table(frame: pd.DataFrame, path: Path) -> None:
    if path.suffix == ".csv":
        frame.to_csv(path, index=False)
    else:
        frame.to_parquet(path, index=False)
