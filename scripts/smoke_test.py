"""End-to-end smoke test: 1 epoch, minimal model, across all datasets in the metadata YAML.

Supports all Ludwig modalities: tabular, text, image, audio, and generative tasks.
Loads up to SAMPLE_ROWS rows from each dataset, runs one epoch, and reports PASS/FAIL.

Usage:
    python scripts/smoke_test.py [--metadata-yaml dataset_metadata.yaml] [--gpu-id 0]
"""
from __future__ import annotations

import argparse
import logging
import os
import re
import sys
import time
import traceback
from pathlib import Path

import yaml

logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(message)s")
logger = logging.getLogger(__name__)

SAMPLE_ROWS = 1000


def _check_kaggle_credentials() -> tuple[str, str] | None:
    username = os.environ.get("KAGGLE_USERNAME")
    key = os.environ.get("KAGGLE_KEY")
    if username and key:
        return username, key
    config_dir = os.environ.get("KAGGLE_CONFIG_DIR", os.path.expanduser("~/.kaggle"))
    token_path = Path(config_dir) / "kaggle.json"
    if token_path.exists():
        import json
        data = json.loads(token_path.read_text())
        if data.get("username") and data.get("key"):
            return data["username"], data["key"]
    return None


def _prompt_kaggle_setup():
    print(
        "\n  Kaggle credentials not found.\n"
        "\n"
        "  To set up:\n"
        "  1. Go to https://www.kaggle.com/settings\n"
        "  2. Scroll to 'API' and click 'Create New Token'\n"
        "  3. Save the downloaded kaggle.json to:  ~/.kaggle/kaggle.json\n"
        "     (or set KAGGLE_USERNAME and KAGGLE_KEY environment variables)\n"
        "  4. Run:  chmod 600 ~/.kaggle/kaggle.json\n"
        "  5. Re-run this script\n"
    )
    sys.exit(1)


def _kaggle_competition_for_dataset(name: str) -> str | None:
    try:
        from ludwig.datasets import get_dataset
        return get_dataset(name).config.kaggle_competition
    except Exception:
        return None


def _exit_competition_rules(competition: str, dataset_name: str):
    rules_url = f"https://www.kaggle.com/competitions/{competition}/rules"
    print(
        f"\n  Competition rules not accepted for '{dataset_name}'.\n"
        f"\n"
        f"  1. Open this URL in your browser:\n"
        f"     {rules_url}\n"
        f"  2. Click 'I Understand and Accept'\n"
        f"  3. Re-run this script\n"
    )
    sys.exit(1)


def _infer_feature_type(series) -> str:
    """Infer Ludwig feature type from a pandas Series."""
    import pandas as pd
    if pd.api.types.is_float_dtype(series) or pd.api.types.is_integer_dtype(series):
        return "number"
    sample = series.dropna().astype(str).head(30)
    if sample.empty:
        return "category"
    # Image / audio path detection
    if sample.str.match(r'.*\.(jpe?g|png|gif|bmp|tiff?)$', case=False).mean() > 0.5:
        return "image"
    if sample.str.match(r'.*\.(wav|mp3|flac|ogg|aac)$', case=False).mean() > 0.5:
        return "audio"
    # Long strings → text (avg > 30 chars or any value > 100 chars)
    avg_len = sample.str.len().mean()
    max_len = sample.str.len().max()
    if avg_len > 30 or max_len > 100:
        return "text"
    return "category"


def _make_config(df, target_column: str, task_type: str, loader_config=None) -> dict:
    """Build a minimal Ludwig config, using the dataset's own feature types when available."""
    import pandas as pd

    # Determine output features
    if loader_config is not None and loader_config.output_features:
        output_features = [dict(f) for f in loader_config.output_features]
    else:
        output_type_map = {
            "binary": "binary",
            "multiclass": "category",
            "binary_or_multiclass": "category",
            "regression": "number",
            "text_generation": "text",
            "image_segmentation": "image",
        }
        out_type = output_type_map.get(task_type, "category")
        output_features = [{"name": target_column, "type": out_type}]

    target_cols = {f["name"] for f in output_features}

    # Build input features with smart type detection
    input_features = []
    for col in df.columns:
        if col in target_cols:
            continue
        feat_type = _infer_feature_type(df[col])
        feat = {"name": col, "type": feat_type}
        # Minimal encoder overrides to keep smoke test fast
        if feat_type == "image":
            feat["encoder"] = {"type": "stacked_cnn", "num_filters": 8, "num_conv_layers": 1}
        elif feat_type == "audio":
            feat["encoder"] = {"type": "stacked_cnn", "num_filters": 8, "num_conv_layers": 1}
        input_features.append(feat)

    if not input_features:
        raise ValueError(f"No input features found — target_cols={target_cols}, df.cols={list(df.columns)}")

    input_types = {f["type"] for f in input_features}
    output_types = {f["type"] for f in output_features}
    has_text_out = "text" in output_types
    has_image_out = "image" in output_types
    has_image_in = "image" in input_types
    has_audio_in = "audio" in input_types
    has_text_in = "text" in input_types

    # Batch size scaled by modality cost
    if has_text_out or has_image_out:
        batch_size = 4
    elif has_image_in or has_audio_in:
        batch_size = 16
    elif has_text_in:
        batch_size = 16
    else:
        batch_size = 32

    return {
        "input_features": input_features,
        "output_features": output_features,
        "combiner": {"type": "concat"},
        "trainer": {"epochs": 1, "batch_size": batch_size, "early_stop": -1},
    }


def _load_df(name: str, meta: dict):
    """Load up to SAMPLE_ROWS rows for a dataset entry."""
    import pandas as pd

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
            raise TimeoutError("OpenML fetch timed out after 120s")

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

    if len(df) > SAMPLE_ROWS:
        df = df.sample(n=SAMPLE_ROWS, random_state=42).reset_index(drop=True)

    return df


def _get_loader_config(name: str, source: str):
    """Return the Ludwig DatasetConfig for a built-in dataset, or None."""
    if source != "ludwig":
        return None
    try:
        from ludwig.datasets import get_dataset
        return get_dataset(name).config
    except Exception:
        return None


def _run_one(name: str, meta: dict, gpu_id: int | None) -> tuple[str, float, str | None]:
    """Returns (status, wall_seconds, error_message)."""
    import tempfile
    from ludwig.api import LudwigModel

    if gpu_id is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

    t0 = time.monotonic()
    try:
        df = _load_df(name, meta)
        # For OpenML tasks, target_column comes from the task itself if not set in YAML
        target_col = meta.get("target_column")
        if not target_col and meta.get("source") == "openml":
            import openml
            task = openml.tasks.get_task(meta["openml_task_id"])
            target_col = task.target_name
        if not target_col:
            raise ValueError(f"target_column not set for {name!r}")
        task_type = meta.get("task_type", "binary")

        # Drop rows where target is NaN (unlabeled splits)
        df = df.dropna(subset=[target_col]).reset_index(drop=True)
        if len(df) < 10:
            return "skip", 0.0, f"only {len(df)} labeled rows after dropna"

        loader_config = _get_loader_config(name, meta.get("source", ""))

        from sklearn.model_selection import train_test_split
        train_df, val_df = train_test_split(df, test_size=0.2, random_state=42)

        config = _make_config(df, target_col, task_type, loader_config=loader_config)

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


def _discover_ludwig_datasets() -> dict:
    """Auto-discover all Ludwig built-in datasets."""
    from ludwig.datasets import list_datasets, get_dataset

    meta = {}
    for name in sorted(list_datasets()):
        try:
            loader = get_dataset(name)
            cfg = loader.config
            ofs = getattr(cfg, "output_features", [])
            if not ofs:
                continue
            out = ofs[0]
            out_type = out.get("type", "")
            type_map = {"binary": "binary", "category": "multiclass", "number": "regression",
                        "text": "text_generation", "image": "image"}
            task_type = type_map.get(out_type, out_type)
            meta[name] = {
                "source": "ludwig",
                "target_column": out["name"],
                "task_type": task_type,
            }
        except Exception:
            pass
    return meta


def _is_403_error(err: str | None) -> bool:
    return err is not None and ("403" in err or "Forbidden" in err)


def main():
    parser = argparse.ArgumentParser(description="Smoke-test all datasets: 1 epoch, minimal model")
    parser.add_argument("--metadata-yaml", default=None,
        help="Path to dataset_metadata.yaml. If omitted, auto-discovers Ludwig built-ins.")
    parser.add_argument("--gpu-id", type=int, default=None)
    parser.add_argument("--datasets", nargs="*", default=None,
        help="Subset of dataset names to test. Default: all.")
    parser.add_argument("--skip-sources", nargs="*", default=[],
        help="Skip datasets whose source matches any of these values.")
    parser.add_argument("--skip-kaggle", action="store_true", default=False,
        help="Skip all Kaggle datasets without prompting.")
    args = parser.parse_args()

    if args.metadata_yaml:
        with open(args.metadata_yaml) as f:
            meta_all: dict = yaml.safe_load(f)
    else:
        meta_all = _discover_ludwig_datasets()
        print(f"Auto-discovered {len(meta_all)} Ludwig datasets")

    names = args.datasets if args.datasets else list(meta_all.keys())
    skip_sources = set(args.skip_sources or [])

    kaggle_names = [
        n for n in names
        if "kaggle_credentials_required" in (meta_all.get(n, {}).get("tags") or [])
    ]
    has_kaggle = bool(kaggle_names) and not args.skip_kaggle

    kaggle_creds = None
    if has_kaggle:
        kaggle_creds = _check_kaggle_credentials()
        if not kaggle_creds:
            _prompt_kaggle_setup()

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

        is_kaggle = "kaggle_credentials_required" in (meta.get("tags") or [])

        if is_kaggle and args.skip_kaggle:
            print(f"{name:<{col_w}}  {'skip':<6}  {'':>6}  --skip-kaggle")
            skips += 1
            continue

        sys.stdout.write(f"{name:<{col_w}}  {'...':<6}\r")
        sys.stdout.flush()

        status, elapsed, err = _run_one(name, meta, args.gpu_id)

        if status == "fail" and is_kaggle and _is_403_error(err):
            competition = _kaggle_competition_for_dataset(name)
            if competition:
                _exit_competition_rules(competition, name)

        note = ""
        if status == "fail":
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
