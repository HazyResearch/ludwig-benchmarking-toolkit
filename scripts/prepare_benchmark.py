"""Full benchmark preparation pipeline.

Usage:
    # Prepare OpenML-CC18 suite (72 datasets)
    python scripts/prepare_benchmark.py --openml-suite 99 \\
        --registry benchmark/dataset_registry.json \\
        --configs-dir benchmark/configs

    # Prepare Ludwig builtins
    python scripts/prepare_benchmark.py --ludwig-builtins \\
        --registry benchmark/dataset_registry.json \\
        --configs-dir benchmark/configs

    # Prepare from dataset_metadata.yaml (YAML-driven, most flexible)
    python scripts/prepare_benchmark.py --metadata-yaml dataset_metadata.yaml \\
        --registry benchmark/dataset_registry.json \\
        --configs-dir benchmark/configs

    # Prepare specific datasets
    python scripts/prepare_benchmark.py --datasets openml_task_7592 titanic \\
        --registry benchmark/dataset_registry.json \\
        --configs-dir benchmark/configs

    # Dry run — show what would happen
    python scripts/prepare_benchmark.py --openml-suite 99 --dry-run
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)






# ---------------------------------------------------------------------------
# Config generation (mirrors generate_configs.py logic, importable here)
# ---------------------------------------------------------------------------

def _generate_and_write_configs(
    entry,
    df: "pd.DataFrame",
    configs_dir: Path,
    n: int,
    seed: int,
) -> int:
    """Generate, validate, and write configs. Returns count of valid configs written."""
    from ludwig.automl.config_sampler import configs_from_dataframe
    from ludwig.automl.config_validator import validate_config_for_dataset

    target_column = entry.target_column
    if not target_column:
        raise ValueError(f"[{entry.name}] target_column is not set")

    sampled = configs_from_dataframe(df, target_column=target_column, n=n, seed=seed)
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

    return len(valid_configs)


# ---------------------------------------------------------------------------
# Summary table (Rich if available, plain text fallback)
# ---------------------------------------------------------------------------

@dataclass
class _DatasetSummaryRow:
    name: str
    source: str
    n_rows: int
    n_features: int
    task_type: str
    target_column: str
    quality: str
    n_configs: int
    elapsed_s: float
    error: str


def _print_summary_table(rows: list[_DatasetSummaryRow]) -> None:
    try:
        from rich.console import Console
        from rich.table import Table

        table = Table(title="Benchmark Preparation Summary", show_lines=False)
        table.add_column("Dataset", style="cyan", no_wrap=True)
        table.add_column("Source")
        table.add_column("Rows", justify="right")
        table.add_column("Feats", justify="right")
        table.add_column("Task")
        table.add_column("Target")
        table.add_column("Quality")
        table.add_column("Configs", justify="right")
        table.add_column("Time(s)", justify="right")
        table.add_column("Error")

        for row in rows:
            quality_style = "green" if row.quality == "PASS" else "red"
            table.add_row(
                row.name,
                row.source,
                str(row.n_rows) if row.n_rows else "-",
                str(row.n_features) if row.n_features else "-",
                row.task_type or "-",
                row.target_column or "-",
                f"[{quality_style}]{row.quality}[/{quality_style}]",
                str(row.n_configs) if row.n_configs else "-",
                f"{row.elapsed_s:.1f}",
                row.error or "",
            )

        Console().print(table)

    except ImportError:
        # Plain text fallback
        header = f"{'Dataset':<35} {'Source':<8} {'Rows':>7} {'Feats':>5} {'Task':<12} {'Target':<20} {'Quality':<6} {'Configs':>7} {'Time':>6}  Error"
        print()
        print(header)
        print("-" * len(header))
        for row in rows:
            print(
                f"{row.name:<35} {row.source:<8} {str(row.n_rows) if row.n_rows else '-':>7} "
                f"{str(row.n_features) if row.n_features else '-':>5} {row.task_type or '-':<12} "
                f"{(row.target_column or '-'):<20} {row.quality:<6} "
                f"{str(row.n_configs) if row.n_configs else '-':>7} {row.elapsed_s:>6.1f}  {row.error or ''}"
            )


# ---------------------------------------------------------------------------
# Core pipeline
# ---------------------------------------------------------------------------

def _prepare_entry(
    entry,
    configs_dir: Path,
    n: int,
    seed: int,
    skip_quality_check: bool,
    dry_run: bool,
) -> _DatasetSummaryRow:
    """Run the full prepare pipeline for one DatasetEntry."""
    t0 = time.monotonic()
    summary = _DatasetSummaryRow(
        name=entry.name,
        source=entry.source,
        n_rows=entry.n_rows or 0,
        n_features=entry.n_features or 0,
        task_type=entry.task_type or "",
        target_column=entry.target_column or "",
        quality="",
        n_configs=0,
        elapsed_s=0.0,
        error="",
    )

    try:
        # 1. Load data
        from benchmark.dataset_registry import load_dataframe_for_entry
        df = load_dataframe_for_entry(entry)

        # 2. Quality check
        if not skip_quality_check:
            from ludwig.utils.dataset_quality import check_dataset_quality
            qr = check_dataset_quality(df, target_column=entry.target_column, dataset_name=entry.name)
            summary.quality = "PASS" if qr.passed else "FAIL"
            if not qr.passed:
                failures = "; ".join(c.message for c in qr.failures)
                logger.warning("[%s] Quality FAIL: %s — skipping", entry.name, failures)
                if not dry_run:
                    entry.quality_passed = False
                summary.elapsed_s = time.monotonic() - t0
                return summary
        else:
            summary.quality = "SKIP"

        # 3. Target detection (if not already set)
        if not entry.target_column:
            # OpenML tasks already set _openml_target during load
            if hasattr(entry, "_openml_target") and entry._openml_target:
                entry.target_column = entry._openml_target
            else:
                from ludwig.automl.target_detection import detect_target_column
                det = detect_target_column(df)
                entry.target_column = det.column
                logger.info(
                    "[%s] Auto-detected target: %s (confidence=%.2f)",
                    entry.name, det.column, det.confidence
                )

        # 4. Infer task type from target column if not set
        if not entry.task_type and entry.target_column in df.columns:
            from ludwig.automl.target_detection import infer_task_type
            entry.task_type = infer_task_type(df[entry.target_column]).value

        # 5. Update metadata
        entry.n_rows = len(df)
        entry.n_features = df.shape[1]
        if not skip_quality_check:
            entry.quality_passed = True

        # Populate summary fields
        summary.n_rows = entry.n_rows
        summary.n_features = entry.n_features
        summary.task_type = entry.task_type or ""
        summary.target_column = entry.target_column or ""

        # 6. Generate configs
        if not dry_run:
            n_configs = _generate_and_write_configs(entry, df, configs_dir, n, seed)
            entry.n_configs = n_configs
            summary.n_configs = n_configs
        else:
            # In dry-run mode, count without writing
            from ludwig.automl.config_sampler import configs_from_dataframe
            from ludwig.automl.config_validator import validate_config_for_dataset
            sampled = configs_from_dataframe(df, target_column=entry.target_column, n=n, seed=seed)
            n_valid = sum(
                1 for sc in sampled
                if validate_config_for_dataset(sc.config_dict, df).is_valid
            )
            summary.n_configs = n_valid

    except Exception as exc:
        logger.error("[%s] Failed: %s", entry.name, exc)
        summary.error = str(exc)
        summary.quality = summary.quality or "ERROR"

    summary.elapsed_s = time.monotonic() - t0
    return summary


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="End-to-end benchmark preparation: download → quality check → configs.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    source_group = parser.add_argument_group("Dataset source (pick one or more)")
    source_group.add_argument(
        "--openml-suite",
        type=int,
        metavar="SUITE_ID",
        help="Add and prepare all tasks from an OpenML benchmark suite (e.g. 99 for CC18)",
    )
    source_group.add_argument(
        "--ludwig-builtins",
        action="store_true",
        help="Add and prepare all Ludwig built-in datasets",
    )
    source_group.add_argument(
        "--datasets",
        nargs="+",
        metavar="DATASET_NAME",
        help="Prepare specific datasets already in the registry",
    )
    source_group.add_argument(
        "--metadata-yaml",
        metavar="PATH",
        help=(
            "Register datasets from a dataset_metadata.yaml file and prepare them. "
            "Equivalent to --openml-suite / --ludwig-builtins but driven by a YAML manifest. "
            "Default path: dataset_metadata.yaml in the repo root."
        ),
    )

    parser.add_argument(
        "--registry",
        default="benchmark/dataset_registry.json",
        help="Path to dataset_registry.json (default: benchmark/dataset_registry.json)",
    )
    parser.add_argument(
        "--configs-dir",
        default="benchmark/configs",
        help="Directory to write configs.jsonl files (default: benchmark/configs)",
    )
    parser.add_argument(
        "--n",
        type=int,
        default=100,
        help="Number of configs to generate per dataset (default: 100)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed (default: 42)",
    )
    parser.add_argument(
        "--skip-quality-check",
        action="store_true",
        help="Skip the quality check step",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Run the pipeline but do not write any files or update registry",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Skip datasets that already have a configs.jsonl",
    )

    return parser


def main() -> None:
    # Allow running as `python scripts/prepare_benchmark.py` from repo root
    sys.path.insert(0, str(Path(__file__).parent.parent))
    from benchmark.dataset_registry import (
        DatasetRegistry,
        register_from_metadata_yaml,
        register_ludwig_builtins,
        register_openml_suite,
    )

    parser = _build_parser()
    args = parser.parse_args()

    if not any([args.openml_suite, args.ludwig_builtins, args.datasets, args.metadata_yaml]):
        parser.print_help()
        sys.exit(1)

    registry_path = Path(args.registry)
    configs_dir = Path(args.configs_dir)
    registry = DatasetRegistry(registry_path)

    # Expand dataset list from requested sources
    if args.openml_suite is not None:
        added = register_openml_suite(registry, args.openml_suite)
        logger.info("Registered %d datasets from OpenML suite %d", added, args.openml_suite)
        if not args.dry_run:
            registry.save()

    if args.ludwig_builtins:
        added = register_ludwig_builtins(registry)
        logger.info("Registered %d Ludwig built-in datasets", added)
        if not args.dry_run:
            registry.save()

    if args.metadata_yaml is not None:
        yaml_path = Path(args.metadata_yaml)
        if not yaml_path.exists():
            logger.error("--metadata-yaml path does not exist: %s", yaml_path)
            sys.exit(1)
        added = register_from_metadata_yaml(registry, yaml_path)
        logger.info("Registered %d datasets from metadata YAML: %s", added, yaml_path)
        if not args.dry_run:
            registry.save()

    # Determine which entries to process
    if args.datasets:
        entries = []
        for name in args.datasets:
            e = registry.get(name)
            if e is None:
                logger.error("Dataset '%s' not found in registry — skipping", name)
            else:
                entries.append(e)
    elif args.openml_suite is not None and not args.ludwig_builtins:
        # Only the suite datasets
        suite_tag = f"suite_{args.openml_suite}"
        entries = [e for e in registry.all() if suite_tag in e.tags]
    elif args.ludwig_builtins and not args.openml_suite:
        entries = [e for e in registry.all() if "ludwig_builtin" in e.tags]
    else:
        entries = registry.all()

    # Sort by priority descending
    entries = sorted(entries, key=lambda e: -e.priority)

    # --resume: skip datasets that already have configs.jsonl
    if args.resume:
        before = len(entries)
        entries = [
            e for e in entries
            if not (configs_dir / e.name / "configs.jsonl").exists()
        ]
        logger.info("--resume: skipping %d datasets that already have configs.jsonl", before - len(entries))

    logger.info("Preparing %d datasets ...", len(entries))

    summary_rows: list[_DatasetSummaryRow] = []
    for entry in entries:
        logger.info("Processing: %s", entry.name)
        row = _prepare_entry(
            entry,
            configs_dir=configs_dir,
            n=args.n,
            seed=args.seed,
            skip_quality_check=args.skip_quality_check,
            dry_run=args.dry_run,
        )
        summary_rows.append(row)

        # Save registry after each dataset (crash-safe progress)
        if not args.dry_run:
            registry.save()

    _print_summary_table(summary_rows)

    n_ok = sum(1 for r in summary_rows if not r.error and r.quality in ("PASS", "SKIP"))
    n_fail_quality = sum(1 for r in summary_rows if r.quality == "FAIL")
    n_error = sum(1 for r in summary_rows if r.error)
    total_configs = sum(r.n_configs for r in summary_rows)

    print(f"\nTotal: {len(summary_rows)} datasets | "
          f"{n_ok} OK | {n_fail_quality} quality fail | {n_error} error | "
          f"{total_configs} configs generated")
    if args.dry_run:
        print("[DRY RUN] No files were written.")


if __name__ == "__main__":
    main()
