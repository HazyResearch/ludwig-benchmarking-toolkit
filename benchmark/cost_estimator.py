"""Compute cost and time estimator for the Ludwig Mega-AutoML Benchmark.

Estimates total GPU-hours and cloud cost before launching experiments.
"""
from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

# Reference wall times per config (minutes) based on dataset size x model type.
# Conservative estimates calibrated against TabZilla/AMLB observations.
WALL_TIME_TABLE: dict[str, dict[str, float]] = {
    # combiner -> {small: <10k rows, medium: 10k-100k, large: >100k}
    "tabnet":         {"small": 5,  "medium": 12, "large": 30},
    "transformer":    {"small": 8,  "medium": 18, "large": 45},
    "ft_transformer": {"small": 7,  "medium": 15, "large": 35},
    "concat":         {"small": 3,  "medium": 8,  "large": 20},
    "bert":           {"small": 15, "medium": 30, "large": 60},  # text encoder
    "default":        {"small": 5,  "medium": 12, "large": 30},
}

DEFAULT_INSTANCE_TYPES: dict[str, float] = {
    "g4dn.xlarge (T4, 1 GPU)":    0.15,
    "g5.xlarge (A10G, 1 GPU)":    0.50,
    "g5.12xlarge (A10G, 4 GPUs)": 3.00,
    "g5.48xlarge (A10G, 8 GPUs)": 6.00,
    "p3.2xlarge (V100, 1 GPU)":   0.90,
    "p3.8xlarge (V100, 4 GPUs)":  3.60,
}

# GPUs available per instance type (extracted from instance name heuristic)
_INSTANCE_GPU_COUNT: dict[str, int] = {
    "g4dn.xlarge (T4, 1 GPU)":    1,
    "g5.xlarge (A10G, 1 GPU)":    1,
    "g5.12xlarge (A10G, 4 GPUs)": 4,
    "g5.48xlarge (A10G, 8 GPUs)": 8,
    "p3.2xlarge (V100, 1 GPU)":   1,
    "p3.8xlarge (V100, 4 GPUs)":  4,
}


@dataclass
class CostEstimate:
    n_experiments: int
    total_gpu_hours: float
    wall_hours_at_n_gpus: dict[int, float]      # n_gpus -> wall hours
    estimated_cost_usd: dict[str, float]         # instance_type -> cost USD
    breakdown_by_dataset: Optional[list[dict]] = None


def _size_bucket(n_rows: int) -> str:
    if n_rows < 10_000:
        return "small"
    elif n_rows < 100_000:
        return "medium"
    else:
        return "large"


def _avg_wall_time_minutes(n_rows: int) -> float:
    """Return the average wall time in minutes across all combiner types for a given dataset size."""
    bucket = _size_bucket(n_rows)
    times = [entry[bucket] for entry in WALL_TIME_TABLE.values()]
    return sum(times) / len(times)


def estimate_experiment_cost(
    dataset_registry: dict,
    n_configs_per_dataset: int = 100,
    instance_types: dict | None = None,
    n_gpu_options: list[int] | None = None,
    include_breakdown: bool = False,
) -> CostEstimate:
    """Estimates total GPU-hours and cloud cost for the full benchmark run.

    Args:
        dataset_registry: Registry dict with n_rows per dataset entry.
        n_configs_per_dataset: How many configs per dataset.
        instance_types: Dict of {name: cost_per_gpu_hour}. Defaults to common AWS instances.
        n_gpu_options: List of GPU counts to estimate wall time for.
        include_breakdown: If True, include per-dataset breakdown in result.

    Returns:
        CostEstimate dataclass with all cost/time projections.
    """
    if instance_types is None:
        instance_types = DEFAULT_INSTANCE_TYPES
    if n_gpu_options is None:
        n_gpu_options = [1, 4, 8, 16, 32]

    total_gpu_minutes = 0.0
    n_experiments = 0
    breakdown: list[dict] = []

    for dataset_name, entry in dataset_registry.items():
        n_rows = entry.get("n_rows", 0)
        if n_rows == 0:
            # Fall back to medium estimate if n_rows unknown
            n_rows = 50_000

        avg_wall_min = _avg_wall_time_minutes(n_rows)
        dataset_gpu_minutes = avg_wall_min * n_configs_per_dataset
        total_gpu_minutes += dataset_gpu_minutes
        n_experiments += n_configs_per_dataset

        if include_breakdown:
            breakdown.append({
                "dataset_name": dataset_name,
                "n_rows": n_rows,
                "size_bucket": _size_bucket(n_rows),
                "avg_wall_min_per_config": round(avg_wall_min, 1),
                "n_configs": n_configs_per_dataset,
                "dataset_gpu_hours": round(dataset_gpu_minutes / 60, 2),
            })

    total_gpu_hours = total_gpu_minutes / 60.0

    # Wall clock hours at different GPU parallelism levels
    wall_hours_at_n_gpus: dict[int, float] = {}
    for n_gpus in n_gpu_options:
        wall_hours_at_n_gpus[n_gpus] = total_gpu_hours / n_gpus

    # Estimated cost per instance type
    estimated_cost_usd: dict[str, float] = {}
    for instance_name, cost_per_gpu_hr in instance_types.items():
        n_gpus_on_instance = _INSTANCE_GPU_COUNT.get(instance_name, 1)
        wall_hrs = total_gpu_hours / n_gpus_on_instance
        estimated_cost_usd[instance_name] = round(wall_hrs * cost_per_gpu_hr * n_gpus_on_instance, 2)

    return CostEstimate(
        n_experiments=n_experiments,
        total_gpu_hours=round(total_gpu_hours, 2),
        wall_hours_at_n_gpus={k: round(v, 2) for k, v in wall_hours_at_n_gpus.items()},
        estimated_cost_usd=estimated_cost_usd,
        breakdown_by_dataset=breakdown if include_breakdown else None,
    )


def print_cost_report(estimate: CostEstimate) -> None:
    """Prints a formatted cost report (Rich if available, else plain text)."""
    try:
        from rich.console import Console
        from rich.table import Table
        from rich import print as rprint
        _rich_available = True
    except ImportError:
        _rich_available = False

    if _rich_available:
        _print_cost_report_rich(estimate)
    else:
        _print_cost_report_plain(estimate)


def _print_cost_report_rich(estimate: CostEstimate) -> None:
    from rich.console import Console
    from rich.table import Table
    from rich.panel import Panel

    console = Console()

    console.print(Panel.fit(
        f"[bold]Ludwig Mega-AutoML Benchmark — Cost Estimate[/bold]\n"
        f"Total experiments: [cyan]{estimate.n_experiments:,}[/cyan]   "
        f"Total GPU-hours: [yellow]{estimate.total_gpu_hours:,.1f}[/yellow]",
        border_style="blue",
    ))

    # Wall time table
    wall_table = Table(title="Wall Clock Time by GPU Count", show_header=True, header_style="bold magenta")
    wall_table.add_column("GPUs", justify="right")
    wall_table.add_column("Wall Hours", justify="right")
    wall_table.add_column("Wall Days", justify="right")
    for n_gpus, wall_hrs in sorted(estimate.wall_hours_at_n_gpus.items()):
        wall_table.add_row(str(n_gpus), f"{wall_hrs:,.1f}", f"{wall_hrs / 24:.1f}")
    console.print(wall_table)

    # Cost table
    cost_table = Table(title="Estimated Cloud Cost (AWS Spot ~May 2026)", show_header=True, header_style="bold green")
    cost_table.add_column("Instance Type")
    cost_table.add_column("Cost (USD)", justify="right")
    for instance_name, cost in sorted(estimate.estimated_cost_usd.items(), key=lambda x: x[1]):
        cost_table.add_row(instance_name, f"${cost:,.2f}")
    console.print(cost_table)

    # Dataset breakdown
    if estimate.breakdown_by_dataset:
        bd_table = Table(title="Per-Dataset Breakdown", show_header=True, header_style="bold cyan")
        bd_table.add_column("Dataset")
        bd_table.add_column("Rows", justify="right")
        bd_table.add_column("Size", justify="center")
        bd_table.add_column("Avg min/config", justify="right")
        bd_table.add_column("GPU-hours", justify="right")
        for row in estimate.breakdown_by_dataset:
            bd_table.add_row(
                row["dataset_name"],
                f"{row['n_rows']:,}",
                row["size_bucket"],
                str(row["avg_wall_min_per_config"]),
                f"{row['dataset_gpu_hours']:.2f}",
            )
        console.print(bd_table)


def _print_cost_report_plain(estimate: CostEstimate) -> None:
    sep = "-" * 60
    print(sep)
    print("Ludwig Mega-AutoML Benchmark -- Cost Estimate")
    print(sep)
    print(f"  Total experiments : {estimate.n_experiments:,}")
    print(f"  Total GPU-hours   : {estimate.total_gpu_hours:,.1f}")
    print()
    print("  Wall clock time by GPU count:")
    for n_gpus, wall_hrs in sorted(estimate.wall_hours_at_n_gpus.items()):
        print(f"    {n_gpus:3d} GPUs -> {wall_hrs:8.1f} hrs  ({wall_hrs / 24:.1f} days)")
    print()
    print("  Estimated cloud cost (AWS spot ~May 2026):")
    for instance_name, cost in sorted(estimate.estimated_cost_usd.items(), key=lambda x: x[1]):
        print(f"    {instance_name:<40s} ${cost:>10,.2f}")
    if estimate.breakdown_by_dataset:
        print()
        print("  Per-dataset breakdown:")
        header = f"  {'Dataset':<35} {'Rows':>10} {'Size':>8} {'Avg min':>8} {'GPU-hrs':>8}"
        print(header)
        print("  " + "-" * (len(header) - 2))
        for row in estimate.breakdown_by_dataset:
            print(
                f"  {row['dataset_name']:<35} {row['n_rows']:>10,} {row['size_bucket']:>8} "
                f"{row['avg_wall_min_per_config']:>8.1f} {row['dataset_gpu_hours']:>8.2f}"
            )
    print(sep)


def estimate_from_registry_file(
    registry_path: str,
    n_configs: int = 100,
) -> CostEstimate:
    """Convenience function: load registry JSON and estimate cost."""
    with open(registry_path) as f:
        registry = json.load(f)
    return estimate_experiment_cost(
        dataset_registry=registry,
        n_configs_per_dataset=n_configs,
        include_breakdown=True,
    )
