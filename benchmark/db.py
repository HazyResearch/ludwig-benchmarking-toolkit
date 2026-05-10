"""Results database for the Ludwig Mega-AutoML Benchmark.

Uses DuckDB for queries over Parquet files. Zero-server, analytical, cross-platform.
"""
from __future__ import annotations

import dataclasses
import threading
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

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
    start_time: datetime | None = None
    end_time: datetime | None = None
    wall_seconds: float = 0.0
    gpu_type: str = ""
    primary_metric: str = ""
    primary_metric_value: float | None = None
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
        valid_keys = {f.name for f in dataclasses.fields(cls)}
        return cls(**{k: v for k, v in d.items() if k in valid_keys})

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

    def _file_lock(self):
        """Returns a filelock.FileLock context manager for the results directory."""
        from filelock import FileLock
        return FileLock(str(self.lock_path), timeout=30)

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
        try:
            import duckdb
        except ImportError:
            raise ImportError("duckdb is required for analytical queries: pip install duckdb")
        if rebuild:
            self._rebuild_index()
        if not self.index_path.exists():
            return pd.DataFrame(columns=list(RUNS_SCHEMA.keys()))
        con = duckdb.connect(":memory:")
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
        with self._lock, self._file_lock():
            run_path = self._run_parquet_path(run.run_id)
            df = pd.DataFrame([run.to_dict()])
            df.to_parquet(run_path, index=False)
            # Invalidate index so next query rebuilds it
            self.index_path.unlink(missing_ok=True)

    def get_run(self, run_id: str) -> "RunRecord | None":
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

        df = pd.read_parquet(self.index_path)
        if dataset_name is not None:
            df = df[df["dataset_name"] == dataset_name]
        if status is not None:
            df = df[df["status"] == status]
        if combiner is not None:
            df = df[df["combiner"] == combiner]
        return df.reset_index(drop=True)

    def best_per_dataset(self) -> pd.DataFrame:
        """Returns the best run (highest primary_metric_value) per dataset."""
        self._rebuild_index()
        if not self.index_path.exists():
            return pd.DataFrame()

        df = pd.read_parquet(self.index_path)
        done = df[df["status"] == "done"].copy()
        if done.empty:
            return pd.DataFrame()
        idx = done.groupby("dataset_name")["primary_metric_value"].idxmax()
        return done.loc[idx].reset_index(drop=True)

    def combiner_win_rates(self) -> pd.DataFrame:
        """Returns how often each combiner is the best on its dataset."""
        best = self.best_per_dataset()
        if best.empty:
            return pd.DataFrame(columns=["combiner", "wins", "win_rate"])

        df_all = pd.read_parquet(self.index_path)
        n_datasets = df_all[df_all["status"] == "done"]["dataset_name"].nunique()

        wins = best.groupby("combiner").size().reset_index(name="wins")
        wins["win_rate"] = (wins["wins"] * 100.0 / max(n_datasets, 1)).round(2)
        return wins.sort_values("wins", ascending=False).reset_index(drop=True)

    def progress_summary(self) -> dict:
        """Returns dict with counts: total/queued/running/done/failed."""
        self._rebuild_index()
        if not self.index_path.exists():
            return {"total": 0, "queued": 0, "running": 0, "done": 0, "failed": 0}

        df = pd.read_parquet(self.index_path)
        counts = df["status"].value_counts().to_dict()
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

    def export_dashboard(
        self,
        output_dir: str,
        registry: "dict | None" = None,
        export_run_details: bool = True,
    ) -> "Path":
        """Export all results into the structured JSON hierarchy for the dashboard.

        Delegates to :func:`benchmark.exporter.export_dashboard`. See that module for
        a description of the output layout and the meaning of each parameter.
        """
        from benchmark.exporter import export_dashboard as _export  # noqa: PLC0415

        return _export(
            db=self,
            output_dir=output_dir,
            registry=registry,
            export_run_details=export_run_details,
        )
