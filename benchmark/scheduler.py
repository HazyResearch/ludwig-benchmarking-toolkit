"""Schedules and orchestrates N-datasets × M-configs experiments."""
from __future__ import annotations

import json
import logging
import sqlite3
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------

@dataclass
class BenchmarkJob:
    job_id: str
    dataset_name: str
    dataset_source: str
    config_path: str        # Path to the specific config in configs.jsonl
    config_index: int       # Index in configs.jsonl
    config_hash: str
    seed: int = 42
    priority: int = 0       # Higher = runs first
    status: str = "queued"  # queued/running/done/failed
    attempts: int = 0
    max_attempts: int = 3


# ---------------------------------------------------------------------------
# Scheduler
# ---------------------------------------------------------------------------

class BenchmarkScheduler:
    """Manages the job queue and dispatches experiments.

    Uses a SQLite database for job state persistence (survives restarts).
    """

    _CREATE_TABLE = """
    CREATE TABLE IF NOT EXISTS jobs (
        job_id       TEXT PRIMARY KEY,
        dataset_name TEXT NOT NULL,
        dataset_source TEXT NOT NULL,
        config_path  TEXT NOT NULL,
        config_index INTEGER NOT NULL,
        config_hash  TEXT NOT NULL,
        seed         INTEGER NOT NULL DEFAULT 42,
        priority     INTEGER NOT NULL DEFAULT 0,
        status       TEXT NOT NULL DEFAULT 'queued',
        attempts     INTEGER NOT NULL DEFAULT 0,
        max_attempts INTEGER NOT NULL DEFAULT 3,
        run_id       TEXT,
        error        TEXT,
        updated_at   TEXT
    )
    """

    def __init__(self, benchmark_dir: str | Path, db: "BenchmarkDB"):  # noqa: F821
        """Initialize scheduler with job queue at benchmark_dir/jobs.db."""
        self.benchmark_dir = Path(benchmark_dir)
        self.benchmark_dir.mkdir(parents=True, exist_ok=True)
        self.db = db
        self._db_path = self.benchmark_dir / "jobs.db"
        self._init_db()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(str(self._db_path), timeout=30, check_same_thread=False)
        conn.row_factory = sqlite3.Row
        return conn

    def _init_db(self) -> None:
        with self._connect() as conn:
            conn.execute(self._CREATE_TABLE)
            conn.execute("CREATE INDEX IF NOT EXISTS idx_status ON jobs(status)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_priority ON jobs(priority DESC)")
            conn.commit()

    def _row_to_job(self, row: sqlite3.Row) -> BenchmarkJob:
        return BenchmarkJob(
            job_id=row["job_id"],
            dataset_name=row["dataset_name"],
            dataset_source=row["dataset_source"],
            config_path=row["config_path"],
            config_index=row["config_index"],
            config_hash=row["config_hash"],
            seed=row["seed"],
            priority=row["priority"],
            status=row["status"],
            attempts=row["attempts"],
            max_attempts=row["max_attempts"],
        )

    # ------------------------------------------------------------------
    # Population
    # ------------------------------------------------------------------

    def populate_from_config_dir(
        self,
        configs_dir: str | Path,
        dataset_registry: dict,
    ) -> int:
        """Scan configs_dir/{dataset}/*.jsonl and enqueue all jobs.

        Each line in a .jsonl file is a separate config dict.
        Returns count of newly enqueued jobs.
        """
        configs_dir = Path(configs_dir)
        count = 0

        with self._connect() as conn:
            for dataset_dir in sorted(configs_dir.iterdir()):
                if not dataset_dir.is_dir():
                    continue
                dataset_name = dataset_dir.name
                if dataset_name not in dataset_registry:
                    logger.debug("Skipping unknown dataset: %s", dataset_name)
                    continue

                ds_meta = dataset_registry[dataset_name]
                dataset_source = ds_meta.get("source", "path")

                for jsonl_path in sorted(dataset_dir.glob("*.jsonl")):
                    with jsonl_path.open() as f:
                        for idx, line in enumerate(f):
                            line = line.strip()
                            if not line:
                                continue
                            try:
                                cfg = json.loads(line)
                            except json.JSONDecodeError as exc:
                                logger.warning("Bad JSON in %s line %d: %s", jsonl_path, idx, exc)
                                continue

                            config_hash = _hash_config(cfg)
                            job_id = str(uuid.uuid4())

                            # Skip if already enqueued for this (dataset, config_hash)
                            existing = conn.execute(
                                "SELECT job_id FROM jobs WHERE dataset_name=? AND config_hash=?",
                                (dataset_name, config_hash),
                            ).fetchone()
                            if existing:
                                continue

                            conn.execute(
                                """
                                INSERT INTO jobs
                                    (job_id, dataset_name, dataset_source, config_path,
                                     config_index, config_hash, seed, priority, status,
                                     attempts, max_attempts, updated_at)
                                VALUES (?,?,?,?,?,?,?,?,?,?,?,?)
                                """,
                                (
                                    job_id,
                                    dataset_name,
                                    dataset_source,
                                    str(jsonl_path),
                                    idx,
                                    config_hash,
                                    ds_meta.get("seed", 42),
                                    ds_meta.get("priority", 0),
                                    "queued",
                                    0,
                                    3,
                                    _now(),
                                ),
                            )
                            count += 1
            conn.commit()

        logger.info("Enqueued %d new jobs", count)
        return count

    # ------------------------------------------------------------------
    # Queue management
    # ------------------------------------------------------------------

    def next_job(self) -> BenchmarkJob | None:
        """Returns next queued job (highest priority, FIFO within priority)."""
        with self._connect() as conn:
            row = conn.execute(
                """
                SELECT * FROM jobs
                WHERE status = 'queued' AND attempts < max_attempts
                ORDER BY priority DESC, rowid ASC
                LIMIT 1
                """
            ).fetchone()
        return self._row_to_job(row) if row else None

    def mark_running(self, job_id: str) -> None:
        with self._connect() as conn:
            conn.execute(
                "UPDATE jobs SET status='running', attempts=attempts+1, updated_at=? WHERE job_id=?",
                (_now(), job_id),
            )
            conn.commit()

    def mark_done(self, job_id: str, run_id: str) -> None:
        with self._connect() as conn:
            conn.execute(
                "UPDATE jobs SET status='done', run_id=?, updated_at=? WHERE job_id=?",
                (run_id, _now(), job_id),
            )
            conn.commit()

    def mark_failed(self, job_id: str, error: str) -> None:
        with self._connect() as conn:
            # If max_attempts reached, set status=failed; else requeue
            row = conn.execute(
                "SELECT attempts, max_attempts FROM jobs WHERE job_id=?", (job_id,)
            ).fetchone()
            if row and row["attempts"] >= row["max_attempts"]:
                new_status = "failed"
            else:
                new_status = "queued"  # Will be retried
            conn.execute(
                "UPDATE jobs SET status=?, error=?, updated_at=? WHERE job_id=?",
                (new_status, error, _now(), job_id),
            )
            conn.commit()

    # ------------------------------------------------------------------
    # Execution modes
    # ------------------------------------------------------------------

    def run_sequential(self, time_limit_per_job: int = 1800) -> None:
        """Run all queued jobs sequentially (for single-machine use)."""
        from benchmark.runner import RunResult, run_experiment

        while True:
            job = self.next_job()
            if job is None:
                logger.info("No more queued jobs — done.")
                break

            run_id = str(uuid.uuid4())
            cfg = _job_to_run_config(job, run_id, time_limit_per_job)

            self.mark_running(job.job_id)
            logger.info("Running job %s / run %s (%s / %s)", job.job_id, run_id, job.dataset_name, job.config_hash)

            result: RunResult = run_experiment(cfg)

            record = _result_to_record(result, job, cfg)
            self.db.upsert_run(record)

            if result.status == "done":
                self.mark_done(job.job_id, run_id)
            else:
                self.mark_failed(job.job_id, result.error_message or result.status)

    def run_with_ray(
        self,
        max_concurrent: int = 16,
        gpus_per_trial: float = 1.0,
        time_limit_per_job: int = 1800,
    ) -> None:
        """Run all queued jobs via Ray for distributed execution."""
        try:
            import ray
        except ImportError:
            raise RuntimeError("Ray is required for distributed execution: pip install ray")

        from benchmark.runner import RunConfig, RunResult, run_experiment

        if not ray.is_initialized():
            ray.init(ignore_reinit_error=True)

        @ray.remote(num_gpus=gpus_per_trial, max_retries=0)
        def _remote_run(cfg: RunConfig) -> RunResult:
            return run_experiment(cfg)

        pending: dict[ray.ObjectRef, tuple[str, str]] = {}  # ref → (job_id, run_id)

        def _submit_next() -> bool:
            job = self.next_job()
            if job is None:
                return False
            run_id = str(uuid.uuid4())
            cfg = _job_to_run_config(job, run_id, time_limit_per_job)
            self.mark_running(job.job_id)
            ref = _remote_run.remote(cfg)
            pending[ref] = (job.job_id, run_id)
            logger.info("Submitted job %s → run %s", job.job_id, run_id)
            return True

        # Fill the initial pool
        while len(pending) < max_concurrent and _submit_next():
            pass

        while pending:
            ready, _ = ray.wait(list(pending.keys()), num_returns=1, timeout=10.0)
            for ref in ready:
                job_id, run_id = pending.pop(ref)
                try:
                    result: RunResult = ray.get(ref)
                except Exception as exc:
                    logger.error("Ray task failed for job %s: %s", job_id, exc)
                    self.mark_failed(job_id, str(exc))
                    # Look up config to create a minimal record
                    continue

                job_row = self._get_job_row(job_id)
                if job_row:
                    record = _result_to_record(result, job_row, _job_to_run_config(job_row, run_id, time_limit_per_job))
                    self.db.upsert_run(record)

                if result.status == "done":
                    self.mark_done(job_id, run_id)
                else:
                    self.mark_failed(job_id, result.error_message or result.status)

                # Submit next
                while len(pending) < max_concurrent and _submit_next():
                    pass

        logger.info("All Ray jobs completed.")

    def _get_job_row(self, job_id: str) -> BenchmarkJob | None:
        with self._connect() as conn:
            row = conn.execute("SELECT * FROM jobs WHERE job_id=?", (job_id,)).fetchone()
        return self._row_to_job(row) if row else None

    # ------------------------------------------------------------------
    # Progress
    # ------------------------------------------------------------------

    def progress(self) -> dict:
        """Returns progress dict: total/queued/running/done/failed."""
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT status, COUNT(*) AS cnt FROM jobs GROUP BY status"
            ).fetchall()
        counts = {row["status"]: row["cnt"] for row in rows}
        total = sum(counts.values())
        return {
            "total": total,
            "queued": counts.get("queued", 0),
            "running": counts.get("running", 0),
            "done": counts.get("done", 0),
            "failed": counts.get("failed", 0),
        }

    def jobs_iter(self) -> Iterator[BenchmarkJob]:
        """Yield all jobs regardless of status."""
        with self._connect() as conn:
            rows = conn.execute("SELECT * FROM jobs ORDER BY rowid ASC").fetchall()
        for row in rows:
            yield self._row_to_job(row)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _now() -> str:
    from datetime import datetime, timezone
    return datetime.now(timezone.utc).isoformat()


def _hash_config(cfg: dict) -> str:
    import hashlib
    blob = json.dumps(cfg, sort_keys=True).encode()
    return hashlib.sha256(blob).hexdigest()[:16]


def _job_to_run_config(job: BenchmarkJob, run_id: str, time_limit_s: int) -> "RunConfig":  # noqa: F821
    import json

    from benchmark.runner import RunConfig

    # Load the config dict from the jsonl file at config_index
    config_dict: dict = {}
    try:
        with open(job.config_path) as f:
            for i, line in enumerate(f):
                if i == job.config_index:
                    config_dict = json.loads(line.strip())
                    break
    except Exception as exc:
        logger.warning("Could not load config from %s[%d]: %s", job.config_path, job.config_index, exc)

    # For OpenML sources, extract task ID from dataset_name (e.g. "openml_task_7592").
    openml_task_id: int | None = None
    if job.dataset_source == "openml":
        try:
            openml_task_id = int(job.dataset_name.split("_")[-1])
        except (ValueError, IndexError):
            logger.warning("Could not parse openml_task_id from dataset_name '%s'", job.dataset_name)

    return RunConfig(
        run_id=run_id,
        dataset_name=job.dataset_name,
        dataset_source=job.dataset_source,
        dataset_path=None,
        openml_task_id=openml_task_id,
        config_dict=config_dict,
        config_hash=job.config_hash,
        output_dir=str(Path(job.config_path).parent.parent.parent / "output" / run_id),
        seed=job.seed,
        time_limit_s=time_limit_s,
    )


def _result_to_record(result: "RunResult", job: BenchmarkJob, cfg: "RunConfig") -> "RunRecord":  # noqa: F821
    import json
    from datetime import datetime, timezone

    from benchmark.db import RunRecord

    config_dict = cfg.config_dict
    combiner = config_dict.get("combiner", {}).get("type", "") if isinstance(config_dict.get("combiner"), dict) else config_dict.get("combiner", "")
    input_features = config_dict.get("input_features", [])
    encoders = [f.get("encoder", {}).get("type", f.get("encoder", "")) for f in input_features if isinstance(f, dict)]
    output_features = config_dict.get("output_features", [])
    decoder = output_features[0].get("decoder", {}).get("type", "") if output_features and isinstance(output_features[0], dict) else ""
    trainer = config_dict.get("trainer", {})
    lr = trainer.get("learning_rate", 0.0) if isinstance(trainer, dict) else 0.0
    batch_size = trainer.get("batch_size", 0) if isinstance(trainer, dict) else 0
    n_epochs = trainer.get("epochs", 0) if isinstance(trainer, dict) else 0

    return RunRecord(
        run_id=result.run_id,
        dataset_name=job.dataset_name,
        dataset_source=job.dataset_source,
        dataset_n_rows=result.dataset_n_rows,
        dataset_n_features=result.dataset_n_features,
        config_hash=job.config_hash,
        combiner=str(combiner),
        input_encoders=json.dumps(encoders),
        output_decoder=str(decoder),
        learning_rate=float(lr),
        batch_size=int(batch_size),
        n_epochs=int(n_epochs),
        seed=job.seed,
        status=result.status,
        start_time=None,
        end_time=datetime.now(timezone.utc),
        wall_seconds=result.wall_seconds,
        gpu_type="",
        primary_metric=result.primary_metric or "",
        primary_metric_value=result.primary_metric_value,
        secondary_metrics=json.dumps(result.secondary_metrics),
        error_message=result.error_message or "",
        checkpoint_path=result.checkpoint_path or "",
    )
