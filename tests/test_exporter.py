"""Unit tests for benchmark.exporter — dashboard JSON export."""
from __future__ import annotations

import json
import math
from pathlib import Path

import pandas as pd
import pytest

from benchmark.db import BenchmarkDB, RunRecord
from benchmark.exporter import export_dashboard


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _make_record(
    run_id: str,
    dataset_name: str,
    config_hash: str,
    combiner: str,
    primary_metric_value: float | None,
    status: str = "done",
    learning_rate: float = 1e-3,
    batch_size: int = 128,
    secondary_metrics: dict | None = None,
) -> RunRecord:
    return RunRecord(
        run_id=run_id,
        dataset_name=dataset_name,
        dataset_source="openml",
        dataset_n_rows=1000,
        dataset_n_features=10,
        config_hash=config_hash,
        combiner=combiner,
        input_encoders=json.dumps(["passthrough"]),
        output_decoder="mlp_classifier",
        learning_rate=learning_rate,
        batch_size=batch_size,
        n_epochs=10,
        seed=42,
        status=status,
        wall_seconds=60.0,
        gpu_type="T4",
        primary_metric="roc_auc",
        primary_metric_value=primary_metric_value,
        secondary_metrics=json.dumps(secondary_metrics or {"accuracy": 0.9}),
        error_message="",
        checkpoint_path="",
    )


@pytest.fixture()
def populated_db(tmp_path: Path) -> BenchmarkDB:
    db = BenchmarkDB(results_dir=str(tmp_path / "results"))

    # Dataset A — 3 runs with different combiners
    db.upsert_run(_make_record("r1", "ds_a", "hash_tabnet", "tabnet", 0.92))
    db.upsert_run(_make_record("r2", "ds_a", "hash_concat", "concat", 0.88))
    db.upsert_run(_make_record("r3", "ds_a", "hash_transform", "transformer", 0.85))

    # Dataset B — 2 done + 1 failed
    db.upsert_run(_make_record("r4", "ds_b", "hash_tabnet", "tabnet", 0.78))
    db.upsert_run(_make_record("r5", "ds_b", "hash_concat", "concat", 0.81))
    db.upsert_run(_make_record("r6", "ds_b", "hash_transform", "transformer", None, status="failed"))

    # Dataset A — baseline
    db.upsert_run(_make_record("r7", "ds_a", "baseline_xgboost", "baseline_xgboost", 0.90))

    return db


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_export_creates_summary(populated_db: BenchmarkDB, tmp_path: Path) -> None:
    out_dir = export_dashboard(populated_db, output_dir=str(tmp_path / "dash"))

    summary_path = out_dir / "summary.json"
    assert summary_path.exists(), "summary.json must be created"

    with summary_path.open() as f:
        summary = json.load(f)

    assert summary["n_datasets"] == 2
    assert summary["n_runs_done"] == 6  # r1..r5 + r7
    assert summary["n_runs_failed"] == 1


def test_export_creates_datasets_json(populated_db: BenchmarkDB, tmp_path: Path) -> None:
    out_dir = export_dashboard(populated_db, output_dir=str(tmp_path / "dash"))

    ds_path = out_dir / "datasets.json"
    assert ds_path.exists()

    with ds_path.open() as f:
        datasets = json.load(f)

    names = {d["name"] for d in datasets}
    assert "ds_a" in names
    assert "ds_b" in names


def test_export_dataset_detail_ranked(populated_db: BenchmarkDB, tmp_path: Path) -> None:
    out_dir = export_dashboard(populated_db, output_dir=str(tmp_path / "dash"))

    ds_a_path = out_dir / "datasets" / "ds_a.json"
    assert ds_a_path.exists()

    with ds_a_path.open() as f:
        ds_a = json.load(f)

    # Best score should be tabnet at 0.92
    assert abs(ds_a["best_score"] - 0.92) < 1e-9
    assert ds_a["best_config_hash"] == "hash_tabnet"

    # Runs should be sorted by score descending
    done_runs = [r for r in ds_a["runs"] if r["status"] == "done" and not r["combiner"].startswith("baseline_")]
    assert done_runs[0]["rank"] == 1
    assert done_runs[0]["primary_metric_value"] >= done_runs[1]["primary_metric_value"]


def test_export_dataset_baseline_scores(populated_db: BenchmarkDB, tmp_path: Path) -> None:
    out_dir = export_dashboard(populated_db, output_dir=str(tmp_path / "dash"))

    with (out_dir / "datasets" / "ds_a.json").open() as f:
        ds_a = json.load(f)

    assert "xgboost" in ds_a["baseline_scores"]
    assert abs(ds_a["baseline_scores"]["xgboost"] - 0.90) < 1e-9


def test_export_creates_configs_json(populated_db: BenchmarkDB, tmp_path: Path) -> None:
    out_dir = export_dashboard(populated_db, output_dir=str(tmp_path / "dash"))

    configs_path = out_dir / "configs.json"
    assert configs_path.exists()

    with configs_path.open() as f:
        configs = json.load(f)

    hashes = {c["config_hash"] for c in configs}
    assert "hash_tabnet" in hashes
    assert "hash_concat" in hashes
    # Baseline should NOT appear in configs.json
    assert "baseline_xgboost" not in hashes


def test_export_config_detail(populated_db: BenchmarkDB, tmp_path: Path) -> None:
    out_dir = export_dashboard(populated_db, output_dir=str(tmp_path / "dash"))

    tabnet_path = out_dir / "configs" / "hash_tabnet.json"
    assert tabnet_path.exists()

    with tabnet_path.open() as f:
        tabnet = json.load(f)

    assert tabnet["combiner"] == "tabnet"
    assert tabnet["n_datasets_tested"] == 2  # ds_a and ds_b
    # tabnet is best on ds_a (0.92 vs concat 0.88), but not on ds_b (concat 0.81 > tabnet 0.78)
    assert tabnet["n_wins"] == 1


def test_export_combiners_json(populated_db: BenchmarkDB, tmp_path: Path) -> None:
    out_dir = export_dashboard(populated_db, output_dir=str(tmp_path / "dash"))

    combiners_path = out_dir / "combiners.json"
    assert combiners_path.exists()

    with combiners_path.open() as f:
        combiners = json.load(f)

    combiner_names = {c["combiner"] for c in combiners}
    assert "tabnet" in combiner_names
    assert "concat" in combiner_names
    # Baseline should not appear
    assert "baseline_xgboost" not in combiner_names


def test_export_run_details(populated_db: BenchmarkDB, tmp_path: Path) -> None:
    out_dir = export_dashboard(populated_db, output_dir=str(tmp_path / "dash"), export_run_details=True)

    run_path = out_dir / "runs" / "r1.json"
    assert run_path.exists()

    with run_path.open() as f:
        run = json.load(f)

    assert run["run_id"] == "r1"
    assert run["dataset_name"] == "ds_a"
    assert run["combiner"] == "tabnet"
    assert abs(run["primary_metric_value"] - 0.92) < 1e-9


def test_export_no_run_details(populated_db: BenchmarkDB, tmp_path: Path) -> None:
    out_dir = export_dashboard(populated_db, output_dir=str(tmp_path / "dash"), export_run_details=False)

    runs_dir = out_dir / "runs"
    # Directory should either not exist or be empty
    if runs_dir.exists():
        assert not list(runs_dir.glob("*.json")), "No run JSON files should be written"


def test_export_empty_db(tmp_path: Path) -> None:
    db = BenchmarkDB(results_dir=str(tmp_path / "empty_results"))
    out_dir = export_dashboard(db, output_dir=str(tmp_path / "dash"))

    summary_path = out_dir / "summary.json"
    assert summary_path.exists()

    with summary_path.open() as f:
        summary = json.load(f)

    assert summary["n_datasets"] == 0
    assert summary["n_runs_total"] == 0


def test_export_with_registry_metadata(populated_db: BenchmarkDB, tmp_path: Path) -> None:
    registry = {
        "ds_a": {"source": "openml", "n_rows": 12960, "task_type": "binary", "target_column": "income"},
        "ds_b": {"source": "openml", "n_rows": 5000, "task_type": "regression", "target_column": "price"},
    }
    out_dir = export_dashboard(populated_db, output_dir=str(tmp_path / "dash"), registry=registry)

    with (out_dir / "datasets" / "ds_a.json").open() as f:
        ds_a = json.load(f)

    assert ds_a["n_rows"] == 12960
    assert ds_a["task_type"] == "binary"
    assert ds_a["target_column"] == "income"


def test_db_export_dashboard_method(populated_db: BenchmarkDB, tmp_path: Path) -> None:
    out_dir = populated_db.export_dashboard(output_dir=str(tmp_path / "dash"))
    assert (out_dir / "summary.json").exists()
    assert (out_dir / "datasets.json").exists()


def test_nan_values_handled(tmp_path: Path) -> None:
    """NaN primary_metric_value must not crash the exporter."""
    db = BenchmarkDB(results_dir=str(tmp_path / "results"))
    db.upsert_run(_make_record("rnan", "ds_nan", "hash_nan", "concat", float("nan")))
    # Should not raise
    out_dir = export_dashboard(db, output_dir=str(tmp_path / "dash"))
    with (out_dir / "summary.json").open() as f:
        summary = json.load(f)
    assert summary["n_datasets"] == 1
