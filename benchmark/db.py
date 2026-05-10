"""Results database for the Ludwig Mega-AutoML Benchmark.

Uses DuckDB for queries over Parquet files. Zero-server, analytical, cross-platform.
"""
from __future__ import annotations

import json
import threading
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Optional

import duckdb
import pandas as pd

RUNS_SCHEMA = {
    "run_id": "VARCHAR",
    "dataset_name": "VARCHAR",
    "dataset_source": "VARCHAR",       # "openml" | "kaggle" | "ludwig"
    "dataset_n_rows": "INTEGER",
    "dataset_n_features": "INTEGER",
    "config_hash": "VARCHAR",
    "combiner": "VARCHAR",
    "input_encoders": "VARCHAR",       # JSON string
    "output_decoder": "VARCHAR",
    "learning_rate": "DOUBLE",
    "batch_size": "INTEGER",
    "n_epochs": "INTEGER",
    "seed": "INTEGER",
    "status": "VARCHAR",               # queued/running/done/failed
    "start_time": "TIMESTAMP",
    "end_time": "TIMESTAMP",
    "wall_seconds": "DOUBLE",
    "gpu_type": "VARCHAR",
    "primary_metric": "VARCHAR",
    "primary_metric_value": "DOUBLE",
    "secondary_metrics": "VARCHAR",    # JSON string
    "error_message": "VARCHAR",
    "checkpoint_path": "VARCHAR",
}


@dataclass
class RunRecord:
    """Represents a single benchmark run."""
    run_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    dataset_name: str = ""
    dataset_source: str = ""
    dataset_n_rows: int = 0
    dataset_n_features: int = 0
    config_hash: str = ""
    combiner: str = ""
    input_encoders: str = ""           # JSON string
    output_decoder: str = ""
    learning_rate: float = 0.0
    batch_size: int = 0
    n_epochs: int = 0
    seed: int = 42
    status: str = "queued"
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None
    wall_seconds: float = 0.0
    gpu_type: str = ""
    primary_metric: str = ""
    primary_metric_value: Optional[float] = None
    secondary_metrics: str = "{}"      # JSON string
    error_message: str = ""
    checkpoint_path: str = ""

    def to_dict(self) -> dict:
        return {
            "run_id": self.run_id,
            "dataset_name": self.dataset_name,
            "dataset_source": self.dataset_source,
            "dataset_n_rows": self.dataset_n_rows,
            "dataset_n_features": self.dataset_n_features,
            "config_hash": self.config_hash,
            "combiner": self.combiner,
            "input_encoders": self.input_encoders,
            "output_decoder": self.output_decoder,
            "learning_rate": self.learning_rate,
            "batch_size": self.batch_size,
            "n_epochs": self.n_epochs,
            "seed": self.seed,
            "status": self.status,
            "start_time": self.start_time,
            "end_time": self.end_time,
            "wall_seconds": self.wall_seconds,
            "gpu_type": self.gpu_type,
            "primary_metric": self.primary_metric,
            "primary_metric_value": self.primary_metric_value,
            "secondary_metrics": self.secondary_metrics,
            "error_message": self.error_message,
            "checkpoint_path": self.checkpoint_path,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "RunRecord":
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})

    @classmethod
    def from_series(cls, s: pd.Series) -> "RunRecord":
        return cls.from_dict(s.to_dict())


class BenchmarkDB:
    """Manages the benchmark results database.

    Stores one Parquet file per run under {results_dir}/runs/{run_id}.parquet.
    A consolidated index at {results_dir}/runs_index.parquet is rebuilt on demand.
    DuckDB is used for in-process analytical queries.
    Thread safety is ensured by a per-instance lock and a file-level lock file.
    """

    def __init__(self, results_dir: str | Path):
        """Initialize the database at results_dir/runs_index.parquet."""
        self.results_dir = Path(results_dir)
        self.runs_dir = self.results_dir / "runs"
        self.index_path = self.results_dir / "runs_index.parquet"
        self.lock_path = self.results_dir / ".db.lock"

        self.runs_dir.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _run_parquet_path(self, run_id: str) -> Path:
        return self.runs_dir / f"{run_id}.parquet"

    def _acquire_file_lock(self) -> None:
        """Busy-wait on a lock file (cross-process)."""
        import time
        deadline = time.monotonic() + 30.0
        while True:
            try:
                fd = self.lock_path.open("x")
                fd.close()
                return
            except FileExistsError:
                if time.monotonic() > deadline:
                    # Stale lock — remove and retry once
                    self.lock_path.unlink(missing_ok=True)
                    continue
                time.sleep(0.05)

    def _release_file_lock(self) -> None:
        self.lock_path.unlink(missing_ok=True)

    def _rebuild_index(self) -> None:
        """Rebuild runs_index.parquet from all per-run Parquet files."""
        parquet_files = sorted(self.runs_dir.glob("*.parquet"))
        if not parquet_files:
            # Write an empty index with the correct schema
            empty = pd.DataFrame(columns=list(RUNS_SCHEMA.keys()))
            empty.to_parquet(self.index_path, index=False)
            return
        frames = [pd.read_parquet(p) for p in parquet_files]
        combined = pd.concat(frames, ignore_index=True)
        combined.to_parquet(self.index_path, index=False)

    def _load_index(self) -> pd.DataFrame:
        """Load the consolidated index, rebuilding if necessary."""
        if not self.index_path.exists():
            self._rebuild_index()
        if not self.index_path.exists():
            return pd.DataFrame(columns=list(RUNS_SCHEMA.keys()))
        return pd.read_parquet(self.index_path)

    def _query(self, sql: str, rebuild: bool = True) -> pd.DataFrame:
        """Execute a DuckDB SQL query against the runs index."""
        if rebuild:
            self._rebuild_index()
        if not self.index_path.exists():
            return pd.DataFrame(columns=list(RUNS_SCHEMA.keys()))
        con = duckdb.connect(":memory:")
        # Register the parquet file as a virtual table
        result = con.execute(
            sql.replace("{index}", str(self.index_path))
        ).fetchdf()
        con.close()
        return result

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def upsert_run(self, run: RunRecord) -> None:
        """Insert or update a run record."""
        with self._lock:
            self._acquire_file_lock()
            try:
                run_path = self._run_parquet_path(run.run_id)
                df = pd.DataFrame([run.to_dict()])
                df.to_parquet(run_path, index=False)
                # Invalidate index so next query rebuilds it
                self.index_path.unlink(missing_ok=True)
            finally:
                self._release_file_lock()

    def get_run(self, run_id: str) -> Optional[RunRecord]:
        """Retrieve a run by ID."""
        run_path = self._run_parquet_path(run_id)
        if not run_path.exists():
            return None
        df = pd.read_parquet(run_path)
        if df.empty:
            return None
        return RunRecord.from_series(df.iloc[0])

    def list_runs(
        self,
        dataset_name: str | None = None,
        status: str | None = None,
        combiner: str | None = None,
    ) -> pd.DataFrame:
        """Query runs with optional filters. Returns DataFrame."""
        self._rebuild_index()
        if not self.index_path.exists():
            return pd.DataFrame(columns=list(RUNS_SCHEMA.keys()))

        con = duckdb.connect(":memory:")
        df = pd.read_parquet(self.index_path)
        con.register("runs", df)

        clauses = []
        if dataset_name is not None:
            clauses.append(f"dataset_name = '{dataset_name}'")
        if status is not None:
            clauses.append(f"status = '{status}'")
        if combiner is not None:
            clauses.append(f"combiner = '{combiner}'")

        where = f"WHERE {' AND '.join(clauses)}" if clauses else ""
        result = con.execute(f"SELECT * FROM runs {where}").fetchdf()
        con.close()
        return result

    def best_per_dataset(self) -> pd.DataFrame:
        """Returns the best run (highest primary_metric_value) per dataset."""
        self._rebuild_index()
        if not self.index_path.exists():
            return pd.DataFrame()

        con = duckdb.connect(":memory:")
        df = pd.read_parquet(self.index_path)
        con.register("runs", df)
        result = con.execute(
            """
            SELECT r.*
            FROM runs r
            INNER JOIN (
                SELECT dataset_name, MAX(primary_metric_value) AS best_val
                FROM runs
                WHERE status = 'done'
                GROUP BY dataset_name
            ) b
            ON r.dataset_name = b.dataset_name
            AND r.primary_metric_value = b.best_val
            AND r.status = 'done'
            """
        ).fetchdf()
        con.close()
        return result

    def combiner_win_rates(self) -> pd.DataFrame:
        """Returns how often each combiner is the best on its dataset."""
        best = self.best_per_dataset()
        if best.empty:
            return pd.DataFrame(columns=["combiner", "wins", "win_rate"])

        self._rebuild_index()
        con = duckdb.connect(":memory:")
        con.register("best", best)

        df_all = pd.read_parquet(self.index_path)
        con.register("runs", df_all)

        result = con.execute(
            """
            WITH dataset_counts AS (
                SELECT COUNT(DISTINCT dataset_name) AS n_datasets
                FROM runs
                WHERE status = 'done'
            )
            SELECT
                b.combiner,
                COUNT(*) AS wins,
                ROUND(COUNT(*) * 100.0 / (SELECT n_datasets FROM dataset_counts), 2) AS win_rate
            FROM best b
            GROUP BY b.combiner
            ORDER BY wins DESC
            """
        ).fetchdf()
        con.close()
        return result

    def progress_summary(self) -> dict:
        """Returns dict with counts: total/queued/running/done/failed."""
        self._rebuild_index()
        if not self.index_path.exists():
            return {"total": 0, "queued": 0, "running": 0, "done": 0, "failed": 0}

        con = duckdb.connect(":memory:")
        df = pd.read_parquet(self.index_path)
        con.register("runs", df)
        rows = con.execute(
            """
            SELECT status, COUNT(*) AS cnt
            FROM runs
            GROUP BY status
            """
        ).fetchdf()
        con.close()

        counts = dict(zip(rows["status"], rows["cnt"]))
        total = sum(counts.values())
        return {
            "total": total,
            "queued": counts.get("queued", 0),
            "running": counts.get("running", 0),
            "done": counts.get("done", 0),
            "failed": counts.get("failed", 0),
        }

    def export_csv(self, output_path: str) -> None:
        """Export all done runs to CSV."""
        df = self.list_runs(status="done")
        df.to_csv(output_path, index=False)
