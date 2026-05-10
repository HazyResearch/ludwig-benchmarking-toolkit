"""Exports benchmark results into a structured JSON hierarchy for the AutoML dashboard.

Output layout (all under `output_dir`):

    data/
        summary.json            ← global index (dataset list, model list, top stats)
        datasets.json           ← per-dataset summary rows (for the dataset list page)
        combiners.json          ← per-combiner aggregate stats (model family view)
        configs.json            ← per-config-hash aggregate stats (unique model configs)
        datasets/
            {name}.json         ← full run list for one dataset, ranked by score
        configs/
            {config_hash}.json  ← cross-dataset perf for one config
        runs/
            {run_id}.json       ← individual run detail (full metadata)

The dataset-centric view loads `datasets/{name}.json`.
The model-centric view loads `configs/{config_hash}.json`.
The run detail view loads `runs/{run_id}.json`.
The landing page loads `summary.json`, `datasets.json`, `combiners.json`.

All JSON files are generated from the BenchmarkDB Parquet store + an optional
dataset registry dict (for metadata like n_rows, task_type, etc.).
"""
from __future__ import annotations

import json
import logging
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import pandas as pd

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _nan_safe(v: object) -> object:
    """Replace NaN/inf with None so json.dumps doesn't choke."""
    if isinstance(v, float) and (math.isnan(v) or math.isinf(v)):
        return None
    return v


def _row_to_run_summary(row: pd.Series) -> dict:
    """Convert a DataFrame row to a compact run summary dict."""
    sec_raw = row.get("secondary_metrics", "{}")
    try:
        secondary = json.loads(sec_raw) if isinstance(sec_raw, str) else sec_raw or {}
    except Exception:
        secondary = {}

    enc_raw = row.get("input_encoders", "[]")
    try:
        encoders = json.loads(enc_raw) if isinstance(enc_raw, str) else enc_raw or []
    except Exception:
        encoders = []

    return {
        "run_id": row["run_id"],
        "config_hash": row["config_hash"],
        "combiner": row["combiner"],
        "input_encoders": encoders,
        "output_decoder": row.get("output_decoder", ""),
        "learning_rate": _nan_safe(row.get("learning_rate")),
        "batch_size": _nan_safe(row.get("batch_size")),
        "n_epochs": _nan_safe(row.get("n_epochs")),
        "primary_metric": row.get("primary_metric", ""),
        "primary_metric_value": _nan_safe(row.get("primary_metric_value")),
        "secondary_metrics": {k: _nan_safe(v) for k, v in secondary.items()},
        "wall_seconds": _nan_safe(row.get("wall_seconds")),
        "status": row.get("status", ""),
        "error_message": row.get("error_message", "") or "",
    }


def _write_json(path: Path, obj: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        json.dump(obj, f, separators=(",", ":"), default=str)


# ---------------------------------------------------------------------------
# Dataset-level export
# ---------------------------------------------------------------------------


def _export_dataset(
    name: str,
    df_dataset: pd.DataFrame,
    registry_entry: Optional[dict],
    out_dir: Path,
) -> dict:
    """Write data/datasets/{name}.json and return a summary row for datasets.json."""
    done = df_dataset[df_dataset["status"] == "done"].copy()
    failed = df_dataset[df_dataset["status"] != "done"]

    # Sort done runs by primary_metric_value descending (best first)
    if not done.empty and "primary_metric_value" in done.columns:
        done = done.sort_values("primary_metric_value", ascending=False).reset_index(drop=True)

    runs = []
    for rank, (_, row) in enumerate(done.iterrows(), start=1):
        entry = _row_to_run_summary(row)
        entry["rank"] = rank
        runs.append(entry)

    # Add failed runs at the end (no rank)
    for _, row in failed.iterrows():
        entry = _row_to_run_summary(row)
        entry["rank"] = None
        runs.append(entry)

    best_run = runs[0] if runs else None
    best_score = best_run["primary_metric_value"] if best_run else None
    primary_metric = best_run["primary_metric"] if best_run else ""

    # Baseline scores (runs with combiner like "baseline_*")
    baseline_rows = df_dataset[df_dataset["combiner"].str.startswith("baseline_", na=False)]
    baseline_scores: dict = {}
    for _, row in baseline_rows.iterrows():
        key = row["combiner"].replace("baseline_", "")
        baseline_scores[key] = _nan_safe(row.get("primary_metric_value"))

    # Combiner breakdown — win count for each combiner on this dataset
    combiner_best: dict[str, float] = {}
    for _, row in done.iterrows():
        c = row.get("combiner", "")
        v = row.get("primary_metric_value")
        if c and not c.startswith("baseline_") and v is not None and not (isinstance(v, float) and math.isnan(v)):
            if c not in combiner_best or v > combiner_best[c]:
                combiner_best[c] = float(v)
    combiner_scores = [{"combiner": c, "best_score": _nan_safe(v)} for c, v in sorted(combiner_best.items(), key=lambda x: -(x[1] or 0))]

    reg = registry_entry or {}
    dataset_doc = {
        "name": name,
        "source": reg.get("source", df_dataset["dataset_source"].iloc[0] if len(df_dataset) else ""),
        "n_rows": reg.get("n_rows") or (int(df_dataset["dataset_n_rows"].iloc[0]) if len(df_dataset) and "dataset_n_rows" in df_dataset.columns else None),
        "n_features": reg.get("n_features") or (int(df_dataset["dataset_n_features"].iloc[0]) if len(df_dataset) and "dataset_n_features" in df_dataset.columns else None),
        "task_type": reg.get("task_type"),
        "target_column": reg.get("target_column"),
        "primary_metric": primary_metric,
        "best_score": _nan_safe(best_score),
        "best_config_hash": best_run["config_hash"] if best_run else None,
        "best_run_id": best_run["run_id"] if best_run else None,
        "n_runs_done": len(done),
        "n_runs_total": len(df_dataset),
        "baseline_scores": baseline_scores,
        "combiner_scores": combiner_scores,
        "runs": runs,
    }

    _write_json(out_dir / "datasets" / f"{name}.json", dataset_doc)

    # Return compact summary row for datasets.json
    return {
        "name": name,
        "source": dataset_doc["source"],
        "n_rows": dataset_doc["n_rows"],
        "n_features": dataset_doc["n_features"],
        "task_type": dataset_doc["task_type"],
        "target_column": dataset_doc["target_column"],
        "primary_metric": primary_metric,
        "best_score": _nan_safe(best_score),
        "best_combiner": best_run["combiner"] if best_run else None,
        "n_runs_done": len(done),
        "n_runs_total": len(df_dataset),
        "baseline_scores": baseline_scores,
    }


# ---------------------------------------------------------------------------
# Config-level export
# ---------------------------------------------------------------------------


def _export_config(
    config_hash: str,
    df_config: pd.DataFrame,
    dataset_ranks: dict[str, int],
    out_dir: Path,
) -> dict:
    """Write data/configs/{config_hash}.json and return a summary row for configs.json."""
    done = df_config[df_config["status"] == "done"].copy()

    # Representative row (use first done run for config metadata)
    rep = done.iloc[0] if not done.empty else df_config.iloc[0]

    enc_raw = rep.get("input_encoders", "[]")
    try:
        encoders = json.loads(enc_raw) if isinstance(enc_raw, str) else enc_raw or []
    except Exception:
        encoders = []

    dataset_scores = []
    ranks = []
    scores = []
    for _, row in done.iterrows():
        ds_name = row["dataset_name"]
        rank = dataset_ranks.get((ds_name, config_hash))
        score = _nan_safe(row.get("primary_metric_value"))
        dataset_scores.append({
            "dataset_name": ds_name,
            "run_id": row["run_id"],
            "primary_metric": row.get("primary_metric", ""),
            "primary_metric_value": score,
            "rank_on_dataset": rank,
            "wall_seconds": _nan_safe(row.get("wall_seconds")),
            "status": row["status"],
        })
        if rank is not None:
            ranks.append(rank)
        if score is not None:
            scores.append(float(score))

    dataset_scores.sort(key=lambda x: (x.get("primary_metric_value") or 0), reverse=True)

    n_wins = sum(1 for r in ranks if r == 1)
    mean_rank = sum(ranks) / len(ranks) if ranks else None
    mean_score = sum(scores) / len(scores) if scores else None

    config_doc = {
        "config_hash": config_hash,
        "combiner": rep.get("combiner", ""),
        "input_encoders": encoders,
        "output_decoder": rep.get("output_decoder", ""),
        "learning_rate": _nan_safe(rep.get("learning_rate")),
        "batch_size": _nan_safe(rep.get("batch_size")),
        "n_epochs": _nan_safe(rep.get("n_epochs")),
        "n_datasets_tested": len(done["dataset_name"].unique()) if not done.empty else 0,
        "n_wins": n_wins,
        "win_rate": round(100.0 * n_wins / len(ranks), 2) if ranks else None,
        "mean_rank": round(mean_rank, 2) if mean_rank is not None else None,
        "mean_score": round(mean_score, 4) if mean_score is not None else None,
        "dataset_scores": dataset_scores,
    }

    _write_json(out_dir / "configs" / f"{config_hash}.json", config_doc)

    return {
        "config_hash": config_hash,
        "combiner": config_doc["combiner"],
        "input_encoders": encoders,
        "output_decoder": config_doc["output_decoder"],
        "learning_rate": config_doc["learning_rate"],
        "batch_size": config_doc["batch_size"],
        "n_epochs": config_doc["n_epochs"],
        "n_datasets_tested": config_doc["n_datasets_tested"],
        "n_wins": n_wins,
        "win_rate": config_doc["win_rate"],
        "mean_rank": config_doc["mean_rank"],
        "mean_score": config_doc["mean_score"],
    }


# ---------------------------------------------------------------------------
# Combiner aggregate stats
# ---------------------------------------------------------------------------


def _compute_combiner_stats(
    df: pd.DataFrame,
    dataset_ranks: dict[tuple[str, str], int],
) -> list[dict]:
    """Compute per-combiner aggregate stats across all datasets."""
    done = df[
        (df["status"] == "done") &
        (~df["combiner"].str.startswith("baseline_", na=False))
    ].copy()

    if done.empty:
        return []

    groups: dict[str, list] = {}
    for _, row in done.iterrows():
        c = row["combiner"]
        if c not in groups:
            groups[c] = []
        groups[c].append(row)

    stats = []
    for combiner, rows in groups.items():
        n_runs = len(rows)
        scores = [float(r["primary_metric_value"]) for r in rows
                  if r.get("primary_metric_value") is not None
                  and not (isinstance(r["primary_metric_value"], float) and math.isnan(r["primary_metric_value"]))]

        all_ranks = [dataset_ranks.get((r["dataset_name"], r["config_hash"])) for r in rows]
        valid_ranks = [rk for rk in all_ranks if rk is not None]
        n_wins = sum(1 for rk in valid_ranks if rk == 1)

        n_datasets = len({r["dataset_name"] for r in rows})
        win_rate = round(100.0 * n_wins / n_datasets, 2) if n_datasets else 0.0

        s_arr = sorted(scores)
        p25 = s_arr[len(s_arr) // 4] if s_arr else None
        p75 = s_arr[3 * len(s_arr) // 4] if s_arr else None
        mean_s = sum(s_arr) / len(s_arr) if s_arr else None
        mean_r = sum(valid_ranks) / len(valid_ranks) if valid_ranks else None

        stats.append({
            "combiner": combiner,
            "n_runs": n_runs,
            "n_datasets_tested": n_datasets,
            "n_wins": n_wins,
            "win_rate": win_rate,
            "mean_score": round(mean_s, 4) if mean_s is not None else None,
            "p25_score": round(p25, 4) if p25 is not None else None,
            "p75_score": round(p75, 4) if p75 is not None else None,
            "mean_rank": round(mean_r, 2) if mean_r is not None else None,
        })

    stats.sort(key=lambda x: (-x["n_wins"], x.get("mean_rank") or 999))
    return stats


# ---------------------------------------------------------------------------
# Individual run detail
# ---------------------------------------------------------------------------


def _export_run(row: pd.Series, out_dir: Path) -> None:
    """Write data/runs/{run_id}.json with full run metadata."""
    run_doc = {
        "run_id": row["run_id"],
        "dataset_name": row["dataset_name"],
        "dataset_source": row.get("dataset_source", ""),
        "dataset_n_rows": _nan_safe(row.get("dataset_n_rows")),
        "dataset_n_features": _nan_safe(row.get("dataset_n_features")),
        "config_hash": row["config_hash"],
        "combiner": row.get("combiner", ""),
        "input_encoders": json.loads(row["input_encoders"]) if isinstance(row.get("input_encoders"), str) else (row.get("input_encoders") or []),
        "output_decoder": row.get("output_decoder", ""),
        "learning_rate": _nan_safe(row.get("learning_rate")),
        "batch_size": _nan_safe(row.get("batch_size")),
        "n_epochs": _nan_safe(row.get("n_epochs")),
        "seed": row.get("seed"),
        "status": row.get("status", ""),
        "start_time": str(row["start_time"]) if row.get("start_time") is not None else None,
        "end_time": str(row["end_time"]) if row.get("end_time") is not None else None,
        "wall_seconds": _nan_safe(row.get("wall_seconds")),
        "gpu_type": row.get("gpu_type", ""),
        "primary_metric": row.get("primary_metric", ""),
        "primary_metric_value": _nan_safe(row.get("primary_metric_value")),
        "secondary_metrics": json.loads(row["secondary_metrics"]) if isinstance(row.get("secondary_metrics"), str) else (row.get("secondary_metrics") or {}),
        "error_message": row.get("error_message", "") or "",
        "checkpoint_path": row.get("checkpoint_path", "") or "",
    }
    _write_json(out_dir / "runs" / f"{row['run_id']}.json", run_doc)


# ---------------------------------------------------------------------------
# Main export function
# ---------------------------------------------------------------------------


def export_dashboard(
    db: "BenchmarkDB",  # noqa: F821
    output_dir: str | Path,
    registry: Optional[dict] = None,
    export_run_details: bool = True,
) -> Path:
    """Export all benchmark results into a structured JSON hierarchy.

    Generates:
    - data/summary.json         — global stats + lists
    - data/datasets.json        — per-dataset summary rows
    - data/combiners.json       — per-combiner aggregate stats
    - data/configs.json         — per-config-hash aggregate stats
    - data/datasets/{name}.json — full run list per dataset
    - data/configs/{hash}.json  — cross-dataset performance per config
    - data/runs/{run_id}.json   — individual run detail (if export_run_details=True)

    Args:
        db: BenchmarkDB instance to read from.
        output_dir: Root directory to write the data/ subtree into.
        registry: Optional dict from DatasetRegistry.to_scheduler_dict() or similar,
                  keyed by dataset_name. Provides metadata (n_rows, task_type, etc.)
        export_run_details: If True, write individual run JSON files (can be large).

    Returns:
        Path to the output_dir/data/ directory.
    """
    from benchmark.db import BenchmarkDB  # noqa: F811 (TYPE_CHECKING import)

    out_dir = Path(output_dir) / "data"
    out_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Exporting dashboard to %s ...", out_dir)

    # Load all runs
    all_runs = db.list_runs()
    if all_runs.empty:
        logger.warning("No runs found in database — writing empty dashboard.")
        _write_json(out_dir / "summary.json", {
            "generated_at": datetime.now(tz=timezone.utc).isoformat(),
            "n_datasets": 0,
            "n_runs_total": 0,
            "n_runs_done": 0,
        })
        return out_dir

    reg = registry or {}

    # ------------------------------------------------------------------
    # Compute dataset-level ranks: (dataset_name, config_hash) → rank
    # Rank 1 = best score on that dataset
    # ------------------------------------------------------------------
    dataset_ranks: dict[tuple[str, str], int] = {}
    done_runs = all_runs[all_runs["status"] == "done"].copy()

    if not done_runs.empty:
        for ds_name, grp in done_runs.groupby("dataset_name"):
            # Exclude baseline_ entries from ranking
            grp_ranked = grp[~grp["combiner"].str.startswith("baseline_", na=False)].copy()
            grp_ranked = grp_ranked.sort_values("primary_metric_value", ascending=False).reset_index(drop=True)
            for rank, (_, row) in enumerate(grp_ranked.iterrows(), start=1):
                dataset_ranks[(ds_name, row["config_hash"])] = rank

    # ------------------------------------------------------------------
    # Per-dataset export
    # ------------------------------------------------------------------
    dataset_summaries = []
    for ds_name, grp in all_runs.groupby("dataset_name"):
        ds_reg = reg.get(ds_name, {})
        summary_row = _export_dataset(ds_name, grp, ds_reg, out_dir)
        dataset_summaries.append(summary_row)
        logger.debug("Exported dataset: %s (%d runs)", ds_name, len(grp))

    dataset_summaries.sort(key=lambda x: x["name"])
    _write_json(out_dir / "datasets.json", dataset_summaries)

    # ------------------------------------------------------------------
    # Per-config export (skip baseline_ entries)
    # ------------------------------------------------------------------
    non_baseline_runs = all_runs[~all_runs["combiner"].str.startswith("baseline_", na=False)]
    config_summaries = []
    for config_hash, grp in non_baseline_runs.groupby("config_hash"):
        summary_row = _export_config(config_hash, grp, dataset_ranks, out_dir)
        config_summaries.append(summary_row)

    config_summaries.sort(key=lambda x: (-(x.get("n_wins") or 0), x.get("mean_rank") or 999))
    _write_json(out_dir / "configs.json", config_summaries)

    # ------------------------------------------------------------------
    # Per-combiner aggregate stats
    # ------------------------------------------------------------------
    combiner_stats = _compute_combiner_stats(all_runs, dataset_ranks)
    _write_json(out_dir / "combiners.json", combiner_stats)

    # ------------------------------------------------------------------
    # Individual run details
    # ------------------------------------------------------------------
    if export_run_details:
        for _, row in all_runs.iterrows():
            _export_run(row, out_dir)

    # ------------------------------------------------------------------
    # Global summary
    # ------------------------------------------------------------------
    done_count = int((all_runs["status"] == "done").sum())
    failed_count = int((all_runs["status"].isin(["failed", "timeout", "oom"])).sum())
    n_datasets = all_runs["dataset_name"].nunique()
    n_configs = non_baseline_runs["config_hash"].nunique()

    top_combiners = combiner_stats[:10]

    summary = {
        "generated_at": datetime.now(tz=timezone.utc).isoformat(),
        "n_datasets": n_datasets,
        "n_unique_configs": n_configs,
        "n_runs_total": len(all_runs),
        "n_runs_done": done_count,
        "n_runs_failed": failed_count,
        "completion_rate": round(100.0 * done_count / max(len(all_runs), 1), 2),
        "top_combiners": top_combiners,
        "dataset_names": sorted(all_runs["dataset_name"].unique().tolist()),
        "config_hashes": non_baseline_runs["config_hash"].unique().tolist(),
    }
    _write_json(out_dir / "summary.json", summary)

    logger.info(
        "Dashboard export complete: %d datasets, %d configs, %d runs → %s",
        n_datasets, n_configs, len(all_runs), out_dir,
    )
    return out_dir


# ---------------------------------------------------------------------------
# CLI convenience
# ---------------------------------------------------------------------------


def _cli_main() -> None:
    import argparse
    import sys

    sys.path.insert(0, str(Path(__file__).parent.parent))

    parser = argparse.ArgumentParser(
        description="Export benchmark results to a JSON dashboard hierarchy.",
    )
    parser.add_argument("--results-dir", required=True, help="BenchmarkDB results directory")
    parser.add_argument("--output-dir", required=True, help="Dashboard output directory")
    parser.add_argument("--registry", default=None, help="Path to dataset_registry.json (optional)")
    parser.add_argument(
        "--no-run-details",
        action="store_true",
        help="Skip exporting individual run JSON files (faster, smaller output)",
    )
    args = parser.parse_args()

    from benchmark.db import BenchmarkDB

    db = BenchmarkDB(results_dir=args.results_dir)

    registry_dict: dict = {}
    if args.registry:
        with open(args.registry) as f:
            registry_dict = json.load(f)

    out_dir = export_dashboard(
        db=db,
        output_dir=args.output_dir,
        registry=registry_dict,
        export_run_details=not args.no_run_details,
    )
    print(f"Dashboard data written to: {out_dir}")


if __name__ == "__main__":
    _cli_main()
