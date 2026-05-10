"""Ludwig Mega-AutoML Benchmark — unified pipeline CLI.

Ties together dataset registration, config generation, job scheduling,
execution, baseline runs, and dashboard export into a single entry point.

Usage examples:

    # Dry-run: show what would run on CC18 suite
    python scripts/run_benchmark.py --openml-suite 99 --dry-run

    # Full CC18 run, sequential, with baselines
    python scripts/run_benchmark.py \\
        --openml-suite 99 \\
        --registry ./registry.json \\
        --results-dir ./results \\
        --benchmark-dir ./benchmark_run \\
        --run-baselines \\
        --export-dashboard

    # Ray parallel run with live dashboard
    python scripts/run_benchmark.py \\
        --openml-suite 99 \\
        --mode ray --max-concurrent 8 --gpus-per-trial 1.0 \\
        --live-dashboard \\
        --export-dashboard

    # Cost estimate only
    python scripts/run_benchmark.py --openml-suite 99 --estimate-cost
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
import threading
import time
from pathlib import Path

# Allow running as `python scripts/run_benchmark.py` from repo root
sys.path.insert(0, str(Path(__file__).parent.parent))

# ---------------------------------------------------------------------------
# Rich logging setup (optional)
# ---------------------------------------------------------------------------

try:
    from rich.console import Console as _RichConsole
    from rich.logging import RichHandler as _RichHandler
    from rich.panel import Panel as _RichPanel
    _RICH = True
except ImportError:
    _RICH = False

_console = _RichConsole() if _RICH else None


def _setup_logging(level_str: str) -> None:
    level = getattr(logging, level_str.upper(), logging.INFO)
    if _RICH:
        logging.basicConfig(
            level=level,
            format="%(message)s",
            datefmt="[%X]",
            handlers=[_RichHandler(rich_tracebacks=True, show_path=False)],
        )
    else:
        logging.basicConfig(
            level=level,
            format="%(asctime)s %(levelname)-8s %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )


logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Banner
# ---------------------------------------------------------------------------

def _print_banner(args: argparse.Namespace, n_datasets: int, n_total_jobs: int) -> None:
    lines = [
        "Ludwig Mega-AutoML Benchmark",
        "",
        f"  Registry      : {args.registry}",
        f"  Configs dir   : {args.configs_dir}",
        f"  Results dir   : {args.results_dir}",
        f"  Benchmark dir : {args.benchmark_dir}",
        "",
        f"  Datasets      : {n_datasets}",
        f"  Configs/ds    : {args.n_configs}",
        f"  Total jobs    : {n_total_jobs}",
        "",
        f"  Mode          : {args.mode}",
    ]
    if args.mode == "ray":
        lines += [
            f"  Max concurrent: {args.max_concurrent}",
            f"  GPUs/trial    : {args.gpus_per_trial}",
        ]
    lines += [
        f"  Time limit/job: {args.time_limit_per_job}s",
        f"  Max attempts  : {args.max_attempts}",
        f"  Seed          : {args.seed}",
        "",
        f"  Run baselines : {args.run_baselines}",
        f"  Run AutoGluon : {args.run_autogluon}",
        f"  Export dash   : {args.export_dashboard}",
    ]
    if args.dry_run:
        lines += ["", "  *** DRY RUN — no files will be written, no jobs run ***"]

    body = "\n".join(lines)

    if _RICH and _console:
        _console.print(_RichPanel(body, title="Benchmark Plan", border_style="bright_blue"))
    else:
        sep = "=" * 60
        print(f"\n{sep}")
        print(body)
        print(f"{sep}\n")


# ---------------------------------------------------------------------------
# Dataset selection
# ---------------------------------------------------------------------------

def _build_registry(args: argparse.Namespace) -> "DatasetRegistry":  # noqa: F821
    from benchmark.dataset_registry import (
        DatasetRegistry,
        register_ludwig_builtins,
        register_openml_suite,
    )

    registry = DatasetRegistry(args.registry)

    if args.openml_suite is not None:
        logger.info("Registering OpenML suite %d ...", args.openml_suite)
        added = register_openml_suite(registry, args.openml_suite)
        logger.info("Added %d new datasets from suite %d", added, args.openml_suite)
        if not args.dry_run:
            registry.save()

    if args.ludwig_builtins:
        logger.info("Registering Ludwig built-in datasets ...")
        added = register_ludwig_builtins(registry)
        logger.info("Added %d new Ludwig built-in datasets", added)
        if not args.dry_run:
            registry.save()

    return registry


def _select_entries(
    registry: "DatasetRegistry",  # noqa: F821
    args: argparse.Namespace,
) -> list:
    """Filter and sort registry entries according to CLI flags."""
    entries = registry.all()

    # Source filter
    if args.source_filter:
        entries = [e for e in entries if e.source == args.source_filter]
        logger.info("Source filter '%s': %d datasets remain", args.source_filter, len(entries))

    # Priority filter
    if args.priority_min > 0:
        entries = [e for e in entries if e.priority >= args.priority_min]
        logger.info("Priority >= %d: %d datasets remain", args.priority_min, len(entries))

    # Named dataset filter
    if args.datasets:
        names = set(args.datasets)
        entries = [e for e in entries if e.name in names]
        missing = names - {e.name for e in entries}
        for m in missing:
            logger.warning("Dataset '%s' not found in registry — skipping", m)

    # Sort: highest priority first, then alphabetical
    entries = sorted(entries, key=lambda e: (-e.priority, e.name))
    return entries


# ---------------------------------------------------------------------------
# Config generation
# ---------------------------------------------------------------------------

def _configs_jsonl_path(configs_dir: Path, dataset_name: str) -> Path:
    return configs_dir / dataset_name / "configs.jsonl"


def _run_config_generation(
    entries: list,
    args: argparse.Namespace,
) -> None:
    """Generate configs.jsonl for every entry that needs one."""
    from benchmark.dataset_registry import DatasetRegistry

    configs_dir = Path(args.configs_dir)
    registry = DatasetRegistry(args.registry)

    logger.info("Config generation: %d datasets", len(entries))

    for entry in entries:
        jsonl_path = _configs_jsonl_path(configs_dir, entry.name)

        if jsonl_path.exists() and not args.regenerate_configs:
            logger.debug("[%s] configs.jsonl exists — skipping (use --regenerate-configs to force)", entry.name)
            continue

        if not entry.target_column:
            logger.warning("[%s] No target_column in registry — cannot generate configs", entry.name)
            continue

        logger.info("[%s] Generating %d configs ...", entry.name, args.n_configs)

        if args.dry_run:
            logger.info("[DRY RUN] Would write %s", jsonl_path)
            continue

        try:
            _generate_configs_for_entry(entry, configs_dir, args.n_configs, args.seed)
            registry_entry = registry.get(entry.name)
            if registry_entry is not None:
                registry_entry.n_configs = _count_configs(jsonl_path)
            registry.save()
        except Exception as exc:
            logger.error("[%s] Config generation failed: %s", entry.name, exc)


def _generate_configs_for_entry(
    entry,
    configs_dir: Path,
    n: int,
    seed: int,
) -> int:
    """Generate, validate, and write configs.jsonl. Returns count written."""
    from ludwig.automl.config_sampler import configs_from_dataframe
    from ludwig.automl.config_validator import validate_config_for_dataset

    df = _load_dataframe_for_entry(entry)
    sampled = configs_from_dataframe(df, target_column=entry.target_column, n=n, seed=seed)
    valid_configs = [
        sc.config_dict for sc in sampled
        if validate_config_for_dataset(sc.config_dict, df).is_valid
    ]

    if valid_configs:
        out_dir = configs_dir / entry.name
        out_dir.mkdir(parents=True, exist_ok=True)
        jsonl_path = out_dir / "configs.jsonl"
        with jsonl_path.open("w") as f:
            for cfg in valid_configs:
                f.write(json.dumps(cfg) + "\n")
        logger.info("[%s] Wrote %d valid configs to %s", entry.name, len(valid_configs), jsonl_path)

    return len(valid_configs)


def _load_dataframe_for_entry(entry) -> "pd.DataFrame":  # noqa: F821
    """Load the full un-split DataFrame for a DatasetEntry."""
    from pathlib import Path as _Path

    import pandas as pd

    source = entry.source

    if source == "path":
        if not entry.local_path:
            raise ValueError(f"[{entry.name}] source='path' but local_path is not set")
        p = _Path(entry.local_path)
        return pd.read_parquet(p) if p.suffix == ".parquet" else pd.read_csv(p)

    elif source == "openml":
        if entry.openml_task_id is None:
            raise ValueError(f"[{entry.name}] source='openml' but openml_task_id is not set")
        import openml
        task = openml.tasks.get_task(entry.openml_task_id)
        dataset = task.get_dataset()
        X, y, _, _ = dataset.get_data(task=task)
        X[task.target_name] = y
        return X

    elif source == "ludwig":
        from ludwig.datasets import get_dataset
        loader = get_dataset(entry.name)
        train, val, test = loader.load(split=True)
        frames = [d for d in (train, val, test) if d is not None and len(d) > 0]
        return pd.concat(frames, ignore_index=True)

    elif source == "kaggle":
        if not entry.local_path:
            raise ValueError(f"[{entry.name}] source='kaggle' but local_path is not set")
        p = _Path(entry.local_path)
        return pd.read_parquet(p) if p.suffix == ".parquet" else pd.read_csv(p)

    else:
        raise ValueError(f"Unknown source: {source!r}")


def _count_configs(jsonl_path: Path) -> int:
    try:
        with jsonl_path.open() as f:
            return sum(1 for line in f if line.strip())
    except FileNotFoundError:
        return 0


# ---------------------------------------------------------------------------
# Baseline runs
# ---------------------------------------------------------------------------

def _run_baselines(
    entries: list,
    registry_dict: dict,
    db: "BenchmarkDB",  # noqa: F821
    args: argparse.Namespace,
) -> None:
    from benchmark.baselines import run_all_baselines

    # Build target_column_map and task_type_map from registry entries
    target_column_map = {e.name: e.target_column for e in entries if e.target_column}
    task_type_map = {e.name: e.task_type for e in entries if e.task_type}

    missing_target = [e.name for e in entries if not e.target_column]
    missing_task = [e.name for e in entries if not e.task_type]
    if missing_target:
        logger.warning(
            "Skipping baselines for %d datasets missing target_column: %s",
            len(missing_target),
            ", ".join(missing_target[:5]) + ("..." if len(missing_target) > 5 else ""),
        )
    if missing_task:
        logger.warning(
            "Skipping baselines for %d datasets missing task_type: %s",
            len(missing_task),
            ", ".join(missing_task[:5]) + ("..." if len(missing_task) > 5 else ""),
        )

    logger.info(
        "Running baselines on %d datasets (xgboost=%s, lgbm=%s, autogluon=%s)",
        len(target_column_map),
        True,
        True,
        args.run_autogluon,
    )

    if args.dry_run:
        logger.info("[DRY RUN] Would run baselines on %d datasets", len(target_column_map))
        return

    run_all_baselines(
        db=db,
        dataset_registry=registry_dict,
        target_column_map=target_column_map,
        task_type_map=task_type_map,
        run_xgboost=True,
        run_lgbm=True,
        run_autogluon=args.run_autogluon,
        time_limit_s=args.time_limit_per_job,
    )


# ---------------------------------------------------------------------------
# Job scheduling and execution
# ---------------------------------------------------------------------------

def _populate_scheduler(
    scheduler: "BenchmarkScheduler",  # noqa: F821
    entries: list,
    registry_dict: dict,
    args: argparse.Namespace,
) -> int:
    """Scan configs_dir and enqueue all jobs. Returns count of new jobs."""
    configs_dir = Path(args.configs_dir)
    # Use only the entries we care about
    filtered_registry = {e.name: registry_dict[e.name] for e in entries if e.name in registry_dict}
    count = scheduler.populate_from_config_dir(configs_dir, filtered_registry)
    logger.info("Enqueued %d new jobs", count)
    return count


def _apply_max_jobs(scheduler: "BenchmarkScheduler", max_jobs: int) -> None:  # noqa: F821
    """If max_jobs is set, cancel queued jobs beyond the limit."""
    if max_jobs <= 0:
        return
    prog = scheduler.progress()
    total_queued = prog["queued"]
    if total_queued <= max_jobs:
        return
    # Mark excess queued jobs as cancelled by fetching and immediately failing them
    # We do this by reading from SQLite directly
    import sqlite3
    db_path = scheduler._db_path
    with sqlite3.connect(str(db_path), timeout=30) as conn:
        conn.row_factory = sqlite3.Row
        rows = conn.execute(
            "SELECT job_id FROM jobs WHERE status='queued' ORDER BY priority DESC, rowid ASC"
        ).fetchall()
    to_cancel = [r["job_id"] for r in rows[max_jobs:]]
    if to_cancel:
        logger.info("--max-jobs %d: cancelling %d excess queued jobs", max_jobs, len(to_cancel))
        with sqlite3.connect(str(db_path), timeout=30) as conn:
            conn.executemany(
                "UPDATE jobs SET status='failed', error='cancelled by --max-jobs' WHERE job_id=?",
                [(jid,) for jid in to_cancel],
            )
            conn.commit()


def _run_with_max_attempts(scheduler: "BenchmarkScheduler", max_attempts: int) -> None:  # noqa: F821
    """Patch max_attempts into the jobs table for all queued jobs."""
    import sqlite3
    db_path = scheduler._db_path
    with sqlite3.connect(str(db_path), timeout=30) as conn:
        conn.execute(
            "UPDATE jobs SET max_attempts=? WHERE status='queued'",
            (max_attempts,),
        )
        conn.commit()


# ---------------------------------------------------------------------------
# Live dashboard (background thread)
# ---------------------------------------------------------------------------

def _start_live_dashboard_thread(
    db: "BenchmarkDB",  # noqa: F821
    scheduler: "BenchmarkScheduler",  # noqa: F821
    refresh_seconds: int,
    stop_event: threading.Event,
) -> threading.Thread:
    from benchmark.dashboard import print_progress

    def _loop() -> None:
        while not stop_event.is_set():
            try:
                print_progress(db, scheduler)
            except Exception as exc:
                logger.debug("Dashboard refresh error: %s", exc)
            stop_event.wait(timeout=refresh_seconds)

    t = threading.Thread(target=_loop, daemon=True, name="live-dashboard")
    t.start()
    return t


# ---------------------------------------------------------------------------
# Final summary
# ---------------------------------------------------------------------------

def _print_final_summary(
    db: "BenchmarkDB",  # noqa: F821
    scheduler: "BenchmarkScheduler",  # noqa: F821
    t_start: float,
) -> None:
    elapsed = time.monotonic() - t_start
    sched_prog = scheduler.progress()
    db_prog = db.progress_summary()

    n_done = sched_prog["done"]
    n_failed = sched_prog["failed"]
    n_total = sched_prog["total"]
    pct = 100.0 * n_done / max(n_total, 1)

    lines = [
        "Benchmark Run Complete",
        "",
        f"  Wall time     : {elapsed / 3600:.2f}h  ({elapsed:.0f}s)",
        f"  Jobs total    : {n_total}",
        f"  Jobs done     : {n_done}  ({pct:.1f}%)",
        f"  Jobs failed   : {n_failed}",
        "",
        f"  DB done       : {db_prog['done']}",
        f"  DB failed     : {db_prog['failed']}",
    ]

    # Best score per dataset (top 10)
    try:
        best_df = db.best_per_dataset()
        if not best_df.empty:
            lines += ["", "  Best result per dataset (top 10):"]
            for _, row in best_df.head(10).iterrows():
                val = row.get("primary_metric_value")
                val_str = f"{val:.4f}" if val is not None else "n/a"
                lines.append(
                    f"    {row['dataset_name']:<35} "
                    f"{row.get('combiner', ''):<18} "
                    f"{row.get('primary_metric', ''):<10} {val_str}"
                )
    except Exception as exc:
        logger.debug("Could not fetch best_per_dataset: %s", exc)

    # Combiner win rates
    try:
        win_df = db.combiner_win_rates()
        if not win_df.empty:
            lines += ["", "  Combiner win rates:"]
            for _, row in win_df.iterrows():
                lines.append(
                    f"    {row['combiner']:<20} wins={row['wins']}  rate={row['win_rate']:.1f}%"
                )
    except Exception as exc:
        logger.debug("Could not fetch combiner_win_rates: %s", exc)

    body = "\n".join(lines)

    if _RICH and _console:
        border = "green" if n_failed == 0 else "yellow"
        _console.print(_RichPanel(body, title="Summary", border_style=border))
    else:
        sep = "=" * 60
        print(f"\n{sep}")
        print(body)
        print(f"{sep}\n")


# ---------------------------------------------------------------------------
# Cost estimation
# ---------------------------------------------------------------------------

def _do_cost_estimate(entries: list, args: argparse.Namespace) -> None:
    from benchmark.cost_estimator import estimate_experiment_cost, print_cost_report

    # Build a minimal registry dict with n_rows for the estimator
    registry_dict = {
        e.name: {
            "source": e.source,
            "n_rows": e.n_rows or 0,
        }
        for e in entries
    }

    estimate = estimate_experiment_cost(
        dataset_registry=registry_dict,
        n_configs_per_dataset=args.n_configs,
        include_breakdown=True,
    )
    print_cost_report(estimate)


# ---------------------------------------------------------------------------
# Argument parser
# ---------------------------------------------------------------------------

def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="run_benchmark.py",
        description=(
            "Ludwig Mega-AutoML Benchmark — unified pipeline CLI.\n\n"
            "Runs the full pipeline: registry → config generation → scheduling → "
            "execution → baselines → dashboard export."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  # Dry-run on CC18:\n"
            "  python scripts/run_benchmark.py --openml-suite 99 --dry-run\n\n"
            "  # Full sequential run with baselines and dashboard:\n"
            "  python scripts/run_benchmark.py --openml-suite 99 --run-baselines --export-dashboard\n\n"
            "  # Ray run:\n"
            "  python scripts/run_benchmark.py --openml-suite 99 --mode ray --max-concurrent 8\n\n"
            "  # Cost estimate only:\n"
            "  python scripts/run_benchmark.py --openml-suite 99 --estimate-cost\n"
        ),
    )

    # ---- Dataset selection -----------------------------------------------
    ds_group = parser.add_argument_group(
        "Dataset selection",
        "Any combination of sources can be used together.",
    )
    ds_group.add_argument(
        "--registry",
        default="./registry.json",
        metavar="PATH",
        help="Path to load/save the dataset registry JSON (default: ./registry.json)",
    )
    ds_group.add_argument(
        "--openml-suite",
        type=int,
        metavar="ID",
        help="Add all tasks from an OpenML benchmark suite (e.g. 99=CC18, 271=CTR23)",
    )
    ds_group.add_argument(
        "--ludwig-builtins",
        action="store_true",
        help="Add all Ludwig built-in datasets to the registry",
    )
    ds_group.add_argument(
        "--datasets",
        nargs="+",
        metavar="NAME",
        help="Run only specific named datasets from the registry",
    )
    ds_group.add_argument(
        "--priority-min",
        type=int,
        default=0,
        metavar="INT",
        help="Skip datasets with priority below this value (default: 0)",
    )
    ds_group.add_argument(
        "--source-filter",
        choices=["openml", "ludwig", "kaggle", "path"],
        metavar="SOURCE",
        help="Only run datasets from this source (openml|ludwig|kaggle|path)",
    )

    # ---- Config generation -----------------------------------------------
    cfg_group = parser.add_argument_group(
        "Config generation",
        "Config generation is skipped if configs.jsonl already exists (override with flags below).",
    )
    cfg_group.add_argument(
        "--configs-dir",
        default="./benchmark_configs",
        metavar="PATH",
        help="Where to find/write per-dataset configs.jsonl files (default: ./benchmark_configs)",
    )
    cfg_group.add_argument(
        "--n-configs",
        type=int,
        default=100,
        metavar="INT",
        help="Number of configs to generate per dataset (default: 100)",
    )
    cfg_group.add_argument(
        "--seed",
        type=int,
        default=42,
        metavar="INT",
        help="Random seed for config generation (default: 42)",
    )
    cfg_group.add_argument(
        "--regenerate-configs",
        action="store_true",
        help="Force regenerate configs even if configs.jsonl already exists",
    )
    cfg_group.add_argument(
        "--skip-config-gen",
        action="store_true",
        help="Assume configs already exist — skip generation entirely",
    )

    # ---- Execution -------------------------------------------------------
    exec_group = parser.add_argument_group("Execution")
    exec_group.add_argument(
        "--mode",
        choices=["sequential", "ray"],
        default="sequential",
        help="Execution backend: sequential (single machine) or ray (distributed). Default: sequential",
    )
    exec_group.add_argument(
        "--max-concurrent",
        type=int,
        default=16,
        metavar="INT",
        help="(ray) Maximum concurrent jobs (default: 16)",
    )
    exec_group.add_argument(
        "--gpus-per-trial",
        type=float,
        default=1.0,
        metavar="FLOAT",
        help="(ray) GPUs to allocate per job (default: 1.0)",
    )
    exec_group.add_argument(
        "--time-limit-per-job",
        type=int,
        default=1800,
        metavar="SECONDS",
        help="Wall-clock time limit per job in seconds (default: 1800 = 30min)",
    )
    exec_group.add_argument(
        "--max-jobs",
        type=int,
        default=0,
        metavar="INT",
        help="Stop after this many total jobs (0 = unlimited, default: 0)",
    )
    exec_group.add_argument(
        "--max-attempts",
        type=int,
        default=3,
        metavar="INT",
        help="Retry failed jobs this many times (default: 3)",
    )
    exec_group.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the plan and exit — no files written, no jobs run",
    )

    # ---- Baselines -------------------------------------------------------
    bl_group = parser.add_argument_group("Baselines")
    bl_group.add_argument(
        "--run-baselines",
        action="store_true",
        help="Run XGBoost + LightGBM baselines before Ludwig runs",
    )
    bl_group.add_argument(
        "--run-autogluon",
        action="store_true",
        help="Also run AutoGluon baseline (slow, requires autogluon.tabular)",
    )

    # ---- Results ---------------------------------------------------------
    res_group = parser.add_argument_group("Results storage")
    res_group.add_argument(
        "--results-dir",
        default="./results",
        metavar="PATH",
        help="Directory where BenchmarkDB stores Parquet result files (default: ./results)",
    )
    res_group.add_argument(
        "--benchmark-dir",
        default="./benchmark_run",
        metavar="PATH",
        help="Directory where BenchmarkScheduler stores jobs.db (default: ./benchmark_run)",
    )

    # ---- Export ----------------------------------------------------------
    exp_group = parser.add_argument_group("Dashboard export")
    exp_group.add_argument(
        "--export-dashboard",
        action="store_true",
        help="Export structured JSON dashboard after the run completes",
    )
    exp_group.add_argument(
        "--dashboard-dir",
        default="./dashboard",
        metavar="PATH",
        help="Where to write the dashboard data/ directory (default: ./dashboard)",
    )
    exp_group.add_argument(
        "--no-run-details",
        action="store_true",
        help="Skip per-run JSON files in dashboard export (faster, smaller output)",
    )

    # ---- Progress --------------------------------------------------------
    prog_group = parser.add_argument_group("Progress display")
    prog_group.add_argument(
        "--live-dashboard",
        action="store_true",
        help="Show a live-updating terminal dashboard while jobs are running",
    )
    prog_group.add_argument(
        "--refresh-seconds",
        type=int,
        default=30,
        metavar="INT",
        help="Refresh interval for the live dashboard in seconds (default: 30)",
    )
    prog_group.add_argument(
        "--log-level",
        choices=["DEBUG", "INFO", "WARNING"],
        default="INFO",
        help="Logging verbosity (default: INFO)",
    )

    # ---- Cost estimation -------------------------------------------------
    cost_group = parser.add_argument_group("Cost estimation (standalone mode)")
    cost_group.add_argument(
        "--estimate-cost",
        action="store_true",
        help="Print GPU-hour and cloud cost estimate then exit (no training)",
    )

    return parser


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()

    _setup_logging(args.log_level)

    # Require at least one dataset source (unless just estimating cost from an
    # existing registry file)
    has_source = any([
        args.openml_suite is not None,
        args.ludwig_builtins,
        args.datasets,
    ])
    registry_path = Path(args.registry)
    if not has_source and not registry_path.exists():
        parser.error(
            "No dataset source specified and no existing registry found.\n"
            "Use --openml-suite, --ludwig-builtins, or --datasets to add datasets."
        )

    t_start = time.monotonic()

    # ------------------------------------------------------------------
    # 1. Build / load registry
    # ------------------------------------------------------------------

    registry = _build_registry(args)
    entries = _select_entries(registry, args)

    if not entries:
        logger.error("No datasets selected — nothing to do. Check --datasets / --source-filter / --priority-min.")
        sys.exit(1)

    # Build the scheduler-compatible dict view of the registry
    registry_dict = registry.to_scheduler_dict()

    # ------------------------------------------------------------------
    # 2. Cost estimate (standalone mode — exits after printing)
    # ------------------------------------------------------------------
    if args.estimate_cost:
        _do_cost_estimate(entries, args)
        return

    # ------------------------------------------------------------------
    # 3. Compute job count for the banner
    # ------------------------------------------------------------------
    n_datasets = len(entries)
    # Count existing configs to estimate total jobs
    configs_dir = Path(args.configs_dir)
    n_existing_configs = sum(
        _count_configs(_configs_jsonl_path(configs_dir, e.name)) for e in entries
    )
    # For banner: use existing count if skip-gen, else n_configs * n_datasets
    if args.skip_config_gen:
        estimated_jobs = n_existing_configs
    else:
        estimated_jobs = max(n_existing_configs, n_datasets * args.n_configs)
    if args.max_jobs > 0:
        estimated_jobs = min(estimated_jobs, args.max_jobs)

    _print_banner(args, n_datasets, estimated_jobs)

    if args.dry_run:
        logger.info("[DRY RUN] Plan printed — exiting without making changes.")
        return

    # ------------------------------------------------------------------
    # 4. Config generation
    # ------------------------------------------------------------------
    if not args.skip_config_gen:
        _run_config_generation(entries, args)
    else:
        logger.info("--skip-config-gen: assuming configs.jsonl files already exist")

    # ------------------------------------------------------------------
    # 5. Set up DB and scheduler
    # ------------------------------------------------------------------
    from benchmark.db import BenchmarkDB
    from benchmark.scheduler import BenchmarkScheduler

    db = BenchmarkDB(results_dir=args.results_dir)
    scheduler = BenchmarkScheduler(benchmark_dir=args.benchmark_dir, db=db)

    # ------------------------------------------------------------------
    # 6. Baselines (run before Ludwig to get reference scores early)
    # ------------------------------------------------------------------
    if args.run_baselines or args.run_autogluon:
        logger.info("=== Baseline runs ===")
        _run_baselines(entries, registry_dict, db, args)

    # ------------------------------------------------------------------
    # 7. Populate job queue
    # ------------------------------------------------------------------
    logger.info("=== Populating job queue ===")
    _populate_scheduler(scheduler, entries, registry_dict, args)

    if args.max_attempts != 3:
        _run_with_max_attempts(scheduler, args.max_attempts)

    if args.max_jobs > 0:
        _apply_max_jobs(scheduler, args.max_jobs)

    prog = scheduler.progress()
    logger.info(
        "Queue: total=%d queued=%d running=%d done=%d failed=%d",
        prog["total"], prog["queued"], prog["running"], prog["done"], prog["failed"],
    )

    if prog["queued"] == 0:
        logger.info("No queued jobs — nothing to run.")
    else:
        # ------------------------------------------------------------------
        # 8. Run jobs
        # ------------------------------------------------------------------
        logger.info("=== Running %d jobs (mode=%s) ===", prog["queued"], args.mode)

        # Start live dashboard in background thread if requested
        stop_event = threading.Event()
        dashboard_thread = None
        if args.live_dashboard:
            dashboard_thread = _start_live_dashboard_thread(
                db, scheduler, args.refresh_seconds, stop_event
            )

        try:
            if args.mode == "sequential":
                scheduler.run_sequential(time_limit_per_job=args.time_limit_per_job)
            elif args.mode == "ray":
                scheduler.run_with_ray(
                    max_concurrent=args.max_concurrent,
                    gpus_per_trial=args.gpus_per_trial,
                    time_limit_per_job=args.time_limit_per_job,
                )
        except KeyboardInterrupt:
            logger.warning("Interrupted — stopping gracefully.")
        finally:
            if dashboard_thread is not None:
                stop_event.set()
                dashboard_thread.join(timeout=5)

    # ------------------------------------------------------------------
    # 9. Dashboard export
    # ------------------------------------------------------------------
    if args.export_dashboard:
        logger.info("=== Exporting dashboard to %s ===", args.dashboard_dir)
        try:
            out_path = db.export_dashboard(
                output_dir=args.dashboard_dir,
                registry=registry_dict,
                export_run_details=not args.no_run_details,
            )
            logger.info("Dashboard written to: %s", out_path)
        except Exception as exc:
            logger.error("Dashboard export failed: %s", exc)

    # ------------------------------------------------------------------
    # 10. Final summary
    # ------------------------------------------------------------------
    _print_final_summary(db, scheduler, t_start)


if __name__ == "__main__":
    main()
