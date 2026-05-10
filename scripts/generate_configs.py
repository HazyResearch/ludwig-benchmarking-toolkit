"""Generate Ludwig configs for benchmark datasets.

Usage:
    # Generate configs for all datasets in registry
    python scripts/generate_configs.py --registry benchmark/dataset_registry.json \\
        --configs-dir benchmark/configs --n 100

    # Single dataset
    python scripts/generate_configs.py --dataset openml_task_7592 \\
        --registry benchmark/dataset_registry.json --configs-dir benchmark/configs

    # From a CSV directly (auto-detect target)
    python scripts/generate_configs.py --csv mydata.csv --target label \\
        --name my_dataset --configs-dir benchmark/configs
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Dataset loading helpers (reuse logic from runner.py)
# ---------------------------------------------------------------------------

def _load_dataframe(entry) -> "pd.DataFrame":  # noqa: F821
    """Load a full (un-split) DataFrame for config generation."""
    import pandas as pd

    source = entry.source
    if source == "path":
        if not entry.local_path:
            raise ValueError(f"[{entry.name}] source='path' but local_path is not set")
        p = Path(entry.local_path)
        return pd.read_parquet(p) if p.suffix == ".parquet" else pd.read_csv(p)

    elif source == "openml":
        if entry.openml_task_id is None:
            raise ValueError(f"[{entry.name}] source='openml' but openml_task_id is not set")
        import openml
        task = openml.tasks.get_task(entry.openml_task_id)
        dataset = task.get_dataset()
        X, y, _, _ = dataset.get_data(task=task)
        target_name = task.target_name
        X[target_name] = y
        return X

    elif source == "ludwig":
        from ludwig.datasets import get_dataset
        loader = get_dataset(entry.name)
        # load() without split=True returns (train+val+test) merged DataFrame
        train, val, test = loader.load(split=True)
        return _concat_splits(train, val, test)

    elif source == "kaggle":
        if not entry.local_path:
            raise ValueError(
                f"[{entry.name}] source='kaggle' but local_path is not set — download first"
            )
        p = Path(entry.local_path)
        return pd.read_parquet(p) if p.suffix == ".parquet" else pd.read_csv(p)

    else:
        raise ValueError(f"Unknown source: {source!r}")


def _concat_splits(*dfs: "pd.DataFrame") -> "pd.DataFrame":
    import pandas as pd
    return pd.concat([d for d in dfs if d is not None and len(d) > 0], ignore_index=True)


# ---------------------------------------------------------------------------
# Per-dataset config generation
# ---------------------------------------------------------------------------

def _generate_for_dataset(
    entry,
    df: "pd.DataFrame",
    configs_dir: Path,
    n: int,
    seed: int,
    dry_run: bool,
) -> tuple[int, int]:
    """Generate configs for one dataset entry.

    Returns (n_valid, n_generated).
    """
    from ludwig.automl.config_sampler import configs_from_dataframe
    from ludwig.automl.config_validator import validate_config_for_dataset

    target_column = entry.target_column
    if not target_column:
        raise ValueError(f"[{entry.name}] target_column is not set — run prepare_benchmark.py first")

    sampled = configs_from_dataframe(df, target_column=target_column, n=n, seed=seed)
    n_generated = len(sampled)

    valid_configs: list[dict] = []
    for sc in sampled:
        result = validate_config_for_dataset(sc.config_dict, df)
        if result.is_valid:
            valid_configs.append(sc.config_dict)

    n_valid = len(valid_configs)

    if not dry_run and valid_configs:
        out_dir = configs_dir / entry.name
        out_dir.mkdir(parents=True, exist_ok=True)
        jsonl_path = out_dir / "configs.jsonl"
        with jsonl_path.open("w") as f:
            for cfg in valid_configs:
                f.write(json.dumps(cfg) + "\n")
        logger.debug("[%s] Wrote %d configs to %s", entry.name, n_valid, jsonl_path)

    return n_valid, n_generated


# ---------------------------------------------------------------------------
# CSV-only mode (no registry)
# ---------------------------------------------------------------------------

def _run_csv_mode(args: argparse.Namespace) -> None:
    """Generate configs directly from a CSV file without a registry entry."""
    import pandas as pd
    from ludwig.automl.config_sampler import configs_from_dataframe
    from ludwig.automl.config_validator import validate_config_for_dataset

    csv_path = Path(args.csv)
    if not csv_path.exists():
        logger.error("CSV file not found: %s", csv_path)
        sys.exit(1)

    df = pd.read_csv(csv_path)

    target = args.target
    if not target:
        from ludwig.automl.target_detection import detect_target_column
        result = detect_target_column(df)
        target = result.column
        logger.info("Auto-detected target column: %s (confidence=%.2f)", target, result.confidence)

    dataset_name = args.name or csv_path.stem
    configs_dir = Path(args.configs_dir)

    t0 = time.monotonic()
    sampled = configs_from_dataframe(df, target_column=target, n=args.n, seed=args.seed)
    n_generated = len(sampled)

    valid_configs = [
        sc.config_dict for sc in sampled
        if validate_config_for_dataset(sc.config_dict, df).is_valid
    ]
    n_valid = len(valid_configs)
    elapsed = time.monotonic() - t0

    if not args.dry_run and valid_configs:
        out_dir = configs_dir / dataset_name
        out_dir.mkdir(parents=True, exist_ok=True)
        jsonl_path = out_dir / "configs.jsonl"
        with jsonl_path.open("w") as f:
            for cfg in valid_configs:
                f.write(json.dumps(cfg) + "\n")

    print(f"{'[DRY RUN] ' if args.dry_run else ''}{dataset_name}: "
          f"{n_valid}/{n_generated} valid configs in {elapsed:.1f}s")


# ---------------------------------------------------------------------------
# Registry-based mode
# ---------------------------------------------------------------------------

def _run_registry_mode(args: argparse.Namespace) -> None:
    """Generate configs for one or all datasets in the registry."""
    # Import here so the module is usable as a library without installed path
    sys.path.insert(0, str(Path(__file__).parent.parent))
    from benchmark.dataset_registry import DatasetRegistry

    registry_path = Path(args.registry)
    if not registry_path.exists():
        logger.error("Registry file not found: %s", registry_path)
        sys.exit(1)

    registry = DatasetRegistry(registry_path)
    configs_dir = Path(args.configs_dir)

    # Determine which datasets to process
    if args.dataset:
        entries = [registry.get(args.dataset)]
        if entries[0] is None:
            logger.error("Dataset '%s' not found in registry", args.dataset)
            sys.exit(1)
    else:
        entries = sorted(registry.all(), key=lambda e: -e.priority)

    total_valid = 0
    total_generated = 0
    n_ok = 0
    n_fail = 0

    for entry in entries:
        if not entry.target_column:
            logger.warning("[%s] Skipping — no target_column in registry", entry.name)
            n_fail += 1
            continue

        # Skip if already has configs (unless --force)
        jsonl_path = configs_dir / entry.name / "configs.jsonl"
        if jsonl_path.exists() and not args.force:
            logger.info("[%s] Skipping — configs.jsonl already exists (use --force to regenerate)", entry.name)
            continue

        t0 = time.monotonic()
        try:
            df = _load_dataframe(entry)
            n_valid, n_generated = _generate_for_dataset(
                entry, df, configs_dir, args.n, args.seed, args.dry_run
            )
            elapsed = time.monotonic() - t0

            # Update registry n_configs field
            if not args.dry_run:
                entry.n_configs = n_valid
                registry.save()

            total_valid += n_valid
            total_generated += n_generated
            n_ok += 1
            print(f"{'[DRY RUN] ' if args.dry_run else ''}"
                  f"{entry.name}: {n_valid}/{n_generated} valid configs in {elapsed:.1f}s")

        except Exception as exc:
            logger.error("[%s] Failed: %s", entry.name, exc)
            n_fail += 1

    print(f"\nSummary: {n_ok} datasets OK, {n_fail} failed, "
          f"{total_valid}/{total_generated} valid configs total")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Generate Ludwig configs for benchmark datasets.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    # Registry / dataset mode
    reg_group = parser.add_argument_group("Registry mode")
    reg_group.add_argument("--registry", help="Path to dataset_registry.json")
    reg_group.add_argument(
        "--dataset",
        help="Generate configs for a single dataset (by name in registry)",
    )
    reg_group.add_argument(
        "--force",
        action="store_true",
        help="Regenerate even if configs.jsonl already exists",
    )

    # CSV mode
    csv_group = parser.add_argument_group("CSV mode (no registry required)")
    csv_group.add_argument("--csv", help="Path to a CSV file (bypasses registry)")
    csv_group.add_argument("--target", help="Target column name (auto-detected if omitted)")
    csv_group.add_argument("--name", help="Dataset name (defaults to CSV stem)")

    # Shared
    parser.add_argument("--configs-dir", default="benchmark/configs",
                        help="Directory to write configs.jsonl files (default: benchmark/configs)")
    parser.add_argument("--n", type=int, default=100,
                        help="Number of configs to generate per dataset (default: 100)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed (default: 42)")
    parser.add_argument("--dry-run", action="store_true",
                        help="Validate and count but do not write any files")

    return parser


if __name__ == "__main__":
    parser = _build_parser()
    args = parser.parse_args()

    if args.csv:
        _run_csv_mode(args)
    elif args.registry:
        _run_registry_mode(args)
    else:
        parser.print_help()
        sys.exit(1)
