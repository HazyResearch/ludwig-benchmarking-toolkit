"""End-to-end smoke test: 1 epoch, minimal model, across all datasets in the metadata YAML.

Loads up to SAMPLE_ROWS rows from each dataset, runs one epoch of a tiny concat model,
and reports PASS / FAIL for each. Does not save any results.

Usage:
    python scripts/smoke_test.py [--metadata-yaml dataset_metadata.yaml] [--gpu-id 0]
"""
from __future__ import annotations

import argparse
import logging
import os
import sys
import time
import traceback
from pathlib import Path

import yaml

logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(message)s")
logger = logging.getLogger(__name__)

SAMPLE_ROWS = 1000  # rows to use per dataset (fast, still exercises full pipeline)

MINIMAL_CONFIG = {
    "combiner": {"type": "concat"},
    "trainer": {
        "epochs": 1,
        "batch_size": 32,
        "early_stop": -1,
    },
}


def _load_df(name: str, meta: dict):
    """Load up to SAMPLE_ROWS rows for a dataset entry."""
    import pandas as pd
    from sklearn.model_selection import train_test_split

    source = meta["source"]

    if source == "ludwig":
        from ludwig.datasets import get_dataset
        loader = get_dataset(name)
        try:
            train, val, test = loader.load(split=True)
            frames = [d for d in (train, val, test) if d is not None and len(d) > 0]
            df = pd.concat(frames, ignore_index=True)
        except (ValueError, TypeError):
            df = loader.load(split=False)

    elif source == "openml":
        import openml
        import signal

        def _timeout(signum, frame):
            raise TimeoutError("OpenML fetch timed out after 60s")

        old = signal.signal(signal.SIGALRM, _timeout) if hasattr(signal, "SIGALRM") else None
        if old is not None:
            signal.alarm(120)
        try:
            task = openml.tasks.get_task(meta["openml_task_id"])
            dataset = task.get_dataset()
            target_name = task.target_name
            X, y, _, _ = dataset.get_data(target=target_name, dataset_format="dataframe")
            if y is not None:
                X[target_name] = y
            df = X
        finally:
            if old is not None:
                signal.alarm(0)
                signal.signal(signal.SIGALRM, old)

    elif source in ("path", "kaggle"):
        local_path = meta.get("local_path")
        if not local_path or not Path(local_path).exists():
            raise FileNotFoundError(f"local_path not set or missing: {local_path!r}")
        p = Path(local_path)
        df = pd.read_parquet(p) if p.suffix == ".parquet" else pd.read_csv(p)

    else:
        raise ValueError(f"Unknown source: {source!r}")

    # Sample down
    if len(df) > SAMPLE_ROWS:
        df = df.sample(n=SAMPLE_ROWS, random_state=42).reset_index(drop=True)

    return df


def _make_config(df, target_column: str, task_type: str) -> dict:
    import pandas as pd

    output_type_map = {
        "binary": "binary",
        "multiclass": "category",
        "regression": "number",
    }
    out_type = output_type_map.get(task_type, "category")

    input_features = []
    for col in df.columns:
        if col == target_column:
            continue
        dtype = df[col].dtype
        if pd.api.types.is_float_dtype(dtype) or pd.api.types.is_integer_dtype(dtype):
            feat_type = "number"
        else:
            # Treat as category; Ludwig will handle strings
            feat_type = "category"
        input_features.append({"name": col, "type": feat_type})

    cfg = {
        "input_features": input_features,
        "output_features": [{"name": target_column, "type": out_type}],
        **MINIMAL_CONFIG,
    }
    return cfg


def _run_one(name: str, meta: dict, gpu_id: int | None) -> tuple[str, float, str | None]:
    """Returns (status, wall_seconds, error_message)."""
    import tempfile
    from ludwig.api import LudwigModel

    if gpu_id is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

    t0 = time.monotonic()
    try:
        df = _load_df(name, meta)

        target_col = meta["target_column"]
        task_type = meta.get("task_type", "binary")

        # Drop rows where target is NaN (unlabeled competition splits)
        df = df.dropna(subset=[target_col]).reset_index(drop=True)
        if len(df) < 10:
            return "skip", 0.0, f"only {len(df)} labeled rows after dropna"

        # Train/val split (80/20)
        from sklearn.model_selection import train_test_split
        train_df, val_df = train_test_split(df, test_size=0.2, random_state=42)

        config = _make_config(df, target_col, task_type)

        with tempfile.TemporaryDirectory() as tmpdir:
            model = LudwigModel(config=config, logging_level=logging.ERROR)
            model.train(
                training_set=train_df.reset_index(drop=True),
                validation_set=val_df.reset_index(drop=True),
                output_directory=tmpdir,
                skip_save_training_description=True,
                skip_save_training_statistics=True,
                skip_save_model=True,
                random_seed=42,
            )

        return "pass", time.monotonic() - t0, None

    except Exception:
        return "fail", time.monotonic() - t0, traceback.format_exc()


_OUTPUT_TYPE_TO_TASK = {
    "binary": "binary",
    "category": "multiclass",
    "number": "regression",
}

# Output types that are not suitable for the generic automl pipeline (generative tasks)
_SKIP_OUTPUT_TYPES = {"text", "image", "audio", "sequence", "set", "bag", "vector", "timeseries"}


def _discover_ludwig_datasets() -> dict:
    """Auto-discover all Ludwig built-in datasets and build a metadata dict."""
    from ludwig.datasets import list_datasets, get_dataset

    meta = {}
    for name in sorted(list_datasets()):
        try:
            loader = get_dataset(name)
            cfg = loader.config
            ofs = getattr(cfg, "output_features", [])
            if not ofs:
                continue
            out_type = ofs[0].get("type", "")
            if out_type in _SKIP_OUTPUT_TYPES:
                continue
            task_type = _OUTPUT_TYPE_TO_TASK.get(out_type)
            if task_type is None:
                continue
            meta[name] = {
                "source": "ludwig",
                "target_column": ofs[0]["name"],
                "task_type": task_type,
            }
        except Exception:
            pass
    return meta


def main():
    parser = argparse.ArgumentParser(description="Smoke-test all datasets: 1 epoch, minimal model")
    parser.add_argument("--metadata-yaml", default=None,
        help="YAML file of datasets to test. If omitted, auto-discovers all Ludwig built-ins.")
    parser.add_argument("--gpu-id", type=int, default=None)
    parser.add_argument(
        "--datasets", nargs="*", default=None,
        help="Subset of dataset names to test. Default: all.",
    )
    parser.add_argument(
        "--skip-sources", nargs="*", default=["kaggle"],
        help="Skip datasets from these sources (default: kaggle).",
    )
    args = parser.parse_args()

    if args.metadata_yaml:
        with open(args.metadata_yaml) as f:
            meta_all: dict = yaml.safe_load(f)
    else:
        meta_all = _discover_ludwig_datasets()
        print(f"Auto-discovered {len(meta_all)} Ludwig datasets")

    names = args.datasets if args.datasets else list(meta_all.keys())
    skip_sources = set(args.skip_sources or [])

    results: list[dict] = []
    passes = fails = skips = 0

    col_w = max(len(n) for n in names) + 2

    print(f"\n{'DATASET':<{col_w}}  {'STATUS':<6}  {'TIME':>6}  NOTE")
    print("-" * (col_w + 30))

    for name in names:
        meta = meta_all.get(name)
        if meta is None:
            print(f"{name:<{col_w}}  {'skip':<6}  {'':>6}  not in YAML")
            skips += 1
            continue

        if meta.get("source") in skip_sources:
            print(f"{name:<{col_w}}  {'skip':<6}  {'':>6}  source={meta['source']}")
            skips += 1
            continue

        if "kaggle_credentials_required" in (meta.get("tags") or []):
            print(f"{name:<{col_w}}  {'skip':<6}  {'':>6}  kaggle_credentials_required")
            skips += 1
            continue

        sys.stdout.write(f"{name:<{col_w}}  {'...':<6}\r")
        sys.stdout.flush()

        status, elapsed, err = _run_one(name, meta, args.gpu_id)

        note = ""
        if status == "fail":
            # Print only the last line of the traceback for brevity
            note = (err or "").strip().splitlines()[-1][:80]
            fails += 1
        elif status == "pass":
            passes += 1
        else:
            skips += 1

        print(f"{name:<{col_w}}  {status.upper():<6}  {elapsed:>5.1f}s  {note}")

        results.append({"dataset": name, "status": status, "elapsed": elapsed, "error": err})

    print("-" * (col_w + 30))
    print(f"PASS={passes}  FAIL={fails}  SKIP={skips}  total={passes+fails+skips}\n")

    if fails:
        print("=== FAILURES ===")
        for r in results:
            if r["status"] == "fail":
                print(f"\n-- {r['dataset']} --")
                print(r["error"])

    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main())
