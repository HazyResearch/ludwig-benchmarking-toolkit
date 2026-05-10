"""Dataset registry for the Ludwig Mega-AutoML Benchmark.

The registry maps dataset_name → DatasetEntry and is persisted as JSON.
The scheduler reads it to know where to find each dataset.
"""
from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import pandas as pd
logger = logging.getLogger(__name__)


@dataclass
class DatasetEntry:
    name: str
    source: str              # "openml" | "kaggle" | "ludwig" | "path"
    openml_task_id: int | None = None
    kaggle_ref: str | None = None   # e.g. "titanic" or "user/dataset"
    local_path: str | None = None   # absolute path for source="path"
    target_column: str | None = None
    n_rows: int | None = None
    n_features: int | None = None
    task_type: str | None = None    # "binary" | "multiclass" | "regression"
    priority: int = 0                  # higher = runs first
    seed: int = 42
    tags: list[str] = field(default_factory=list)
    quality_passed: bool | None = None
    n_configs: int = 0                 # number of configs generated so far
    notes: str = ""


class DatasetRegistry:
    """Loads, saves, and queries the dataset registry JSON file."""

    def __init__(self, registry_path: str | Path):
        self.registry_path = Path(registry_path)
        self._entries: dict[str, DatasetEntry] = {}
        if self.registry_path.exists():
            self._load()

    def _load(self) -> None:
        with self.registry_path.open() as f:
            raw = json.load(f)
        self._entries = {k: DatasetEntry(**v) for k, v in raw.items()}
        logger.info("Loaded %d datasets from registry", len(self._entries))

    def save(self) -> None:
        self.registry_path.parent.mkdir(parents=True, exist_ok=True)
        with self.registry_path.open("w") as f:
            json.dump({k: asdict(v) for k, v in self._entries.items()}, f, indent=2)

    def add(self, entry: DatasetEntry, overwrite: bool = False) -> None:
        if entry.name in self._entries and not overwrite:
            return
        self._entries[entry.name] = entry

    def get(self, name: str) -> DatasetEntry | None:
        return self._entries.get(name)

    def all(self) -> list[DatasetEntry]:
        return list(self._entries.values())

    def by_source(self, source: str) -> list[DatasetEntry]:
        return [e for e in self._entries.values() if e.source == source]

    def to_scheduler_dict(self) -> dict:
        """Returns the dict format expected by BenchmarkScheduler.populate_from_config_dir()."""
        return {
            name: {
                "source": e.source,
                "openml_task_id": e.openml_task_id,
                "local_path": e.local_path,
                "seed": e.seed,
                "priority": e.priority,
            }
            for name, e in self._entries.items()
        }

    def summary(self) -> str:
        by_source: dict[str, int] = {}
        for e in self._entries.values():
            by_source[e.source] = by_source.get(e.source, 0) + 1
        parts = [f"{s}={n}" for s, n in sorted(by_source.items())]
        return f"DatasetRegistry({len(self._entries)} datasets: {', '.join(parts)})"

    def __len__(self) -> int:
        return len(self._entries)

    def __contains__(self, name: str) -> bool:
        return name in self._entries


def load_dataframe_for_entry(entry: DatasetEntry) -> "pd.DataFrame":
    """Load the full un-split DataFrame for a DatasetEntry.

    Supports sources: ``path``, ``openml``, ``ludwig``, ``kaggle``.
    For OpenML entries, sets ``entry._openml_target`` as a side-effect so the
    caller can access the task-provided target column name.
    """
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
        entry._openml_target = target_name  # type: ignore[attr-defined]
        return X

    elif source == "ludwig":
        from ludwig.datasets import get_dataset
        loader = get_dataset(entry.name)
        train, val, test = loader.load(split=True)
        frames = [d for d in (train, val, test) if d is not None and len(d) > 0]
        return pd.concat(frames, ignore_index=True)

    elif source == "kaggle":
        if not entry.local_path:
            raise ValueError(
                f"[{entry.name}] source='kaggle' but local_path is not set — download first"
            )
        p = Path(entry.local_path)
        return pd.read_parquet(p) if p.suffix == ".parquet" else pd.read_csv(p)

    else:
        raise ValueError(f"[{entry.name}] Unknown source: {source!r}")


def register_openml_suite(
    registry: DatasetRegistry,
    suite_id: int,
    priority: int = 10,
) -> int:
    """Add all tasks from an OpenML benchmark suite to the registry. Returns count added."""
    try:
        import openml
    except ImportError:
        raise ImportError("openml package required: pip install openml")
    suite = openml.study.get_suite(suite_id)
    added = 0
    for task_id in suite.tasks:
        name = f"openml_task_{task_id}"
        if name not in registry:
            registry.add(DatasetEntry(
                name=name,
                source="openml",
                openml_task_id=task_id,
                priority=priority,
                tags=[f"suite_{suite_id}"],
            ))
            added += 1
    return added


def register_ludwig_builtins(
    registry: DatasetRegistry,
    priority: int = 5,
) -> int:
    """Add all Ludwig built-in datasets to the registry."""
    from ludwig.datasets import list_datasets
    added = 0
    for name in list_datasets():
        if name not in registry:
            registry.add(DatasetEntry(
                name=name,
                source="ludwig",
                priority=priority,
                tags=["ludwig_builtin"],
            ))
            added += 1
    return added


def register_from_metadata_yaml(
    registry: DatasetRegistry,
    metadata_yaml_path: str | Path,
    priority_override: int | None = None,
) -> int:
    """Register datasets from a dataset_metadata.yaml file.

    Each top-level key in the YAML becomes the dataset name. Supported fields
    map directly to DatasetEntry attributes:
        source, openml_task_id, kaggle_ref, local_path, target_column,
        task_type, priority, tags, notes, description (stored in notes).

    Args:
        registry: The DatasetRegistry to populate.
        metadata_yaml_path: Path to a YAML file in dataset_metadata.yaml format.
        priority_override: When set, overrides the priority for every entry.

    Returns:
        Count of newly added entries (entries already in the registry are skipped).
    """
    try:
        import yaml
    except ImportError:
        raise ImportError("PyYAML required: pip install pyyaml")

    path = Path(metadata_yaml_path)
    with path.open() as f:
        raw: dict = yaml.safe_load(f)

    if not raw:
        return 0

    added = 0
    for name, meta in raw.items():
        if name in registry:
            continue

        priority = priority_override if priority_override is not None else meta.get("priority", 0)

        # "description" is informational; fold it into notes if notes not set
        notes = meta.get("notes", "") or meta.get("description", "")

        entry = DatasetEntry(
            name=name,
            source=meta["source"],
            openml_task_id=meta.get("openml_task_id"),
            kaggle_ref=meta.get("kaggle_ref"),
            local_path=meta.get("local_path"),
            target_column=meta.get("target_column"),
            task_type=meta.get("task_type"),
            priority=priority,
            tags=list(meta.get("tags") or []),
            notes=notes,
        )
        registry.add(entry)
        added += 1

    return added


def register_kaggle_filtered(
    registry: DatasetRegistry,
    ml_ready_json: str | Path,
    kaggle_data_dir: str | Path,
    priority: int = 8,
) -> int:
    """Add filtered Kaggle datasets from ml_ready.json to the registry."""
    with open(ml_ready_json) as f:
        datasets = json.load(f)
    data_dir = Path(kaggle_data_dir)
    added = 0
    for ds in datasets:
        ref = ds.get("ref", "")
        safe_name = "kaggle_" + ref.replace("/", "_").replace("-", "_")
        if safe_name in registry:
            continue
        # Find local path if already downloaded
        local_path = None
        candidate = data_dir / ref.replace("/", "_")
        for ext in [".csv", ".parquet"]:
            p = candidate.with_suffix(ext)
            if p.exists():
                local_path = str(p)
                break
        registry.add(DatasetEntry(
            name=safe_name,
            source="kaggle",
            kaggle_ref=ref,
            local_path=local_path,
            priority=priority,
            tags=["kaggle"] + ds.get("tags", []),
            notes=ds.get("title", ""),
        ))
        added += 1
    return added
