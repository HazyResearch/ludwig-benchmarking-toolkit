"""Runs a single Ludwig training experiment and records results."""
from __future__ import annotations

import logging
import os
import signal
import threading
import time
import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class RunConfig:
    run_id: str
    dataset_name: str
    dataset_source: Literal["openml", "kaggle", "ludwig", "path"]
    dataset_path: str | None    # Direct path if dataset_source == "path"
    openml_task_id: int | None
    config_dict: dict
    config_hash: str
    output_dir: str
    seed: int = 42
    time_limit_s: int = 1800    # 30 min
    gpu_id: int | None = None


@dataclass
class RunResult:
    run_id: str
    status: str                 # "done" | "failed" | "timeout" | "oom"
    wall_seconds: float
    primary_metric: str | None
    primary_metric_value: float | None
    secondary_metrics: dict
    error_message: str | None
    checkpoint_path: str | None
    dataset_n_rows: int
    dataset_n_features: int


# ---------------------------------------------------------------------------
# Dataset loading
# ---------------------------------------------------------------------------

def _load_dataset_openml(task_id: int) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Load dataset from OpenML, returning (train, val, test)."""
    from sklearn.model_selection import train_test_split
    import openml

    task = openml.tasks.get_task(task_id)
    dataset = task.get_dataset()
    target_name = task.target_name
    X, y, _, _ = dataset.get_data(target=target_name, dataset_format="dataframe")
    if y is not None:
        X[target_name] = y

    train_val, test = train_test_split(X, test_size=0.2, random_state=42)
    train, val = train_test_split(train_val, test_size=0.1, random_state=42)
    return train.reset_index(drop=True), val.reset_index(drop=True), test.reset_index(drop=True)


def _load_dataset_path(path: str) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Load a CSV/Parquet dataset from a local path."""
    from sklearn.model_selection import train_test_split

    p = Path(path)
    if p.suffix == ".parquet":
        df = pd.read_parquet(p)
    else:
        df = pd.read_csv(p)

    train_val, test = train_test_split(df, test_size=0.2, random_state=42)
    train, val = train_test_split(train_val, test_size=0.1, random_state=42)
    return train.reset_index(drop=True), val.reset_index(drop=True), test.reset_index(drop=True)


def _load_dataset_ludwig(name: str) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Load a Ludwig built-in dataset."""
    from sklearn.model_selection import train_test_split
    from ludwig.datasets import get_dataset

    loader = get_dataset(name)
    try:
        train, val, test = loader.load(split=True)
    except (ValueError, TypeError):
        # Dataset has no 'split' column — load whole then split manually
        df = loader.load(split=False)
        train_val, test = train_test_split(df, test_size=0.2, random_state=42)
        train, val = train_test_split(train_val, test_size=0.1, random_state=42)
        return train.reset_index(drop=True), val.reset_index(drop=True), test.reset_index(drop=True)

    # Some Ludwig datasets have no dedicated test split
    if test is None or len(test) == 0:
        # Check if val is unlabeled (e.g. Kaggle competition splits where val = hidden test set)
        # Heuristic: if any column is >80% NaN, val is unlabeled — rebuild all splits from train
        val_max_null = val.isnull().mean().max() if val is not None and len(val) > 0 else 1.0
        if val_max_null > 0.8:
            train, test = train_test_split(train, test_size=0.15, random_state=42)
            train, val = train_test_split(train, test_size=0.15, random_state=42)
        else:
            val, test = train_test_split(val, test_size=0.2, random_state=42)
        val = val.reset_index(drop=True)
        test = test.reset_index(drop=True)
        train = train.reset_index(drop=True)

    return train, val, test


def _load_dataset(cfg: RunConfig) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Dispatch to the correct loader based on dataset_source."""
    source = cfg.dataset_source
    if source == "path":
        if cfg.dataset_path is None:
            raise ValueError("dataset_source='path' requires dataset_path to be set")
        return _load_dataset_path(cfg.dataset_path)
    elif source == "openml":
        if cfg.openml_task_id is None:
            raise ValueError("dataset_source='openml' requires openml_task_id to be set")
        return _load_dataset_openml(cfg.openml_task_id)
    elif source == "ludwig":
        return _load_dataset_ludwig(cfg.dataset_name)
    elif source == "kaggle":
        if cfg.dataset_path is None:
            raise ValueError(
                "dataset_source='kaggle' requires dataset_path to point to a locally downloaded file"
            )
        return _load_dataset_path(cfg.dataset_path)
    else:
        raise ValueError(f"Unknown dataset_source: {source!r}")


# ---------------------------------------------------------------------------
# GPU environment setup
# ---------------------------------------------------------------------------

def _setup_gpu_env(gpu_id: int | None) -> str | None:
    """Set CUDA_VISIBLE_DEVICES and return GPU type string."""
    if gpu_id is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    try:
        import torch
        if torch.cuda.is_available():
            return torch.cuda.get_device_name(0)
    except Exception:
        pass
    return None


# ---------------------------------------------------------------------------
# Timeout support (SIGALRM on Unix, threading.Timer fallback)
# ---------------------------------------------------------------------------

class _TimeoutExpired(Exception):
    pass


class _TimeoutContext:
    """Context manager that raises _TimeoutExpired after `seconds`."""

    def __init__(self, seconds: int):
        self.seconds = seconds
        self._timer: threading.Timer | None = None
        self._timed_out = False

    # Prefer SIGALRM (Unix only, no overhead)
    _use_signal = hasattr(signal, "SIGALRM")

    def _on_timeout(self) -> None:
        self._timed_out = True
        if not self._use_signal:
            # In timer-based mode we can't interrupt a running C extension;
            # set a flag that run_experiment checks, or raise in main thread
            # via ctypes (best-effort).
            import ctypes
            ctypes.pythonapi.PyThreadState_SetAsyncExc(
                ctypes.c_ulong(threading.main_thread().ident),  # type: ignore[arg-type]
                ctypes.py_object(_TimeoutExpired),
            )

    def __enter__(self) -> "_TimeoutContext":
        if self._use_signal:
            signal.signal(signal.SIGALRM, self._signal_handler)
            signal.alarm(self.seconds)
        else:
            self._timer = threading.Timer(self.seconds, self._on_timeout)
            self._timer.daemon = True
            self._timer.start()
        return self

    def _signal_handler(self, signum: int, frame: object) -> None:
        raise _TimeoutExpired(f"Exceeded time limit of {self.seconds}s")

    def __exit__(self, exc_type: object, exc_val: object, exc_tb: object) -> bool:
        if self._use_signal:
            signal.alarm(0)
        elif self._timer is not None:
            self._timer.cancel()
        return False


# ---------------------------------------------------------------------------
# NaN-loss guard
# ---------------------------------------------------------------------------

def _check_train_stats_for_nan(train_stats: dict) -> bool:
    """Return True if training loss went NaN in the first epoch."""
    try:
        # Ludwig train_stats structure: {split: {feature: {metric: [values]}}}
        for split_stats in train_stats.values():
            for feature_stats in split_stats.values():
                for metric_name, values in feature_stats.items():
                    if "loss" in metric_name and values:
                        first_val = values[0]
                        import math
                        if isinstance(first_val, float) and math.isnan(first_val):
                            return True
    except Exception:
        pass
    return False


# ---------------------------------------------------------------------------
# Primary metric extraction
# ---------------------------------------------------------------------------

def _extract_primary_metric(
    eval_stats: dict,
    config_dict: dict,
) -> tuple[str | None, float | None, dict]:
    """Extract primary metric name, value, and secondary metrics from eval_stats."""
    primary_metric = None
    primary_value = None
    secondary: dict = {}

    try:
        output_features = config_dict.get("output_features", [])
        if not output_features:
            return None, None, {}

        first_output = output_features[0]
        feature_name = first_output.get("name", "")
        feat_type = first_output.get("type", "")

        # Metric priority by task type
        metric_priority = {
            "binary": ["roc_auc", "accuracy", "f1"],
            "category": ["accuracy", "hits_at_k"],
            "number": ["r2", "mean_absolute_error", "mean_squared_error"],
            "text": ["token_accuracy", "perplexity"],
            "sequence": ["token_accuracy"],
            "vector": ["mean_squared_error"],
            "set": ["jaccard"],
            "bag": ["mean_absolute_error"],
        }

        stats = eval_stats.get(feature_name, eval_stats.get("combined", {}))
        candidates = metric_priority.get(feat_type, [])

        for metric in candidates:
            if metric in stats:
                primary_metric = metric
                primary_value = float(stats[metric])
                break

        # Collect all numeric metrics as secondary
        for k, v in stats.items():
            if isinstance(v, (int, float)) and k != primary_metric:
                secondary[k] = float(v)

    except Exception as exc:
        logger.debug("Failed to extract primary metric: %s", exc)

    return primary_metric, primary_value, secondary


# ---------------------------------------------------------------------------
# Core experiment runner
# ---------------------------------------------------------------------------

def run_experiment(cfg: RunConfig) -> RunResult:
    """Runs one Ludwig training experiment.

    Steps:
    1. Load dataset (from cache path, openml, or ludwig://)
    2. Enforce time_limit_s via signal.alarm (SIGALRM) or threading.Timer
    3. Call ludwig.api.LudwigModel(config_dict).train(training_set=df, ...)
    4. Evaluate on test set
    5. Return RunResult

    Error handling:
    - Catch RuntimeError with "out of memory" → status="oom"
    - Catch TimeoutError → status="timeout"
    - Catch all other exceptions → status="failed", record error_message
    - On NaN loss after first epoch → raise early to save compute
    """
    from ludwig.api import LudwigModel

    _setup_gpu_env(cfg.gpu_id)
    output_dir = Path(cfg.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    wall_start = time.monotonic()

    dataset_n_rows = 0
    dataset_n_features = 0

    try:
        with _TimeoutContext(cfg.time_limit_s):
            # 1. Load data
            logger.info("[%s] Loading dataset %s (source=%s)", cfg.run_id, cfg.dataset_name, cfg.dataset_source)
            train_df, val_df, test_df = _load_dataset(cfg)
            dataset_n_rows = len(train_df) + len(val_df) + len(test_df)
            dataset_n_features = train_df.shape[1]

            # 2. Train
            logger.info("[%s] Starting training", cfg.run_id)
            model = LudwigModel(
                config=cfg.config_dict,
                logging_level=logging.WARNING,
            )
            train_stats, _, model_output_dir = model.train(
                training_set=train_df,
                validation_set=val_df,
                output_directory=str(output_dir),
                skip_save_training_description=True,
                skip_save_training_statistics=False,
                skip_save_model=False,
                random_seed=cfg.seed,
            )

            # NaN-loss guard
            if _check_train_stats_for_nan(train_stats.to_dict() if hasattr(train_stats, "to_dict") else train_stats):
                raise RuntimeError("NaN loss detected after first epoch — aborting to save compute")

            # 3. Evaluate
            logger.info("[%s] Evaluating on test set", cfg.run_id)
            eval_stats, _, _ = model.evaluate(
                dataset=test_df,
                collect_overall_stats=True,
            )

            wall_seconds = time.monotonic() - wall_start

            primary_metric, primary_value, secondary = _extract_primary_metric(
                eval_stats, cfg.config_dict
            )

            checkpoint = str(model_output_dir) if model_output_dir else None

            return RunResult(
                run_id=cfg.run_id,
                status="done",
                wall_seconds=wall_seconds,
                primary_metric=primary_metric,
                primary_metric_value=primary_value,
                secondary_metrics=secondary,
                error_message=None,
                checkpoint_path=checkpoint,
                dataset_n_rows=dataset_n_rows,
                dataset_n_features=dataset_n_features,
            )

    except _TimeoutExpired:
        wall_seconds = time.monotonic() - wall_start
        logger.warning("[%s] Timed out after %.1fs", cfg.run_id, wall_seconds)
        return RunResult(
            run_id=cfg.run_id,
            status="timeout",
            wall_seconds=wall_seconds,
            primary_metric=None,
            primary_metric_value=None,
            secondary_metrics={},
            error_message=f"Exceeded time limit of {cfg.time_limit_s}s",
            checkpoint_path=None,
            dataset_n_rows=dataset_n_rows,
            dataset_n_features=dataset_n_features,
        )

    except RuntimeError as exc:
        wall_seconds = time.monotonic() - wall_start
        msg = str(exc)
        if "out of memory" in msg.lower():
            status = "oom"
            logger.warning("[%s] OOM after %.1fs", cfg.run_id, wall_seconds)
        else:
            status = "failed"
            logger.error("[%s] RuntimeError: %s", cfg.run_id, msg)
        return RunResult(
            run_id=cfg.run_id,
            status=status,
            wall_seconds=wall_seconds,
            primary_metric=None,
            primary_metric_value=None,
            secondary_metrics={},
            error_message=msg,
            checkpoint_path=None,
            dataset_n_rows=dataset_n_rows,
            dataset_n_features=dataset_n_features,
        )

    except Exception:
        wall_seconds = time.monotonic() - wall_start
        msg = traceback.format_exc()
        logger.error("[%s] Experiment failed:\n%s", cfg.run_id, msg)
        return RunResult(
            run_id=cfg.run_id,
            status="failed",
            wall_seconds=wall_seconds,
            primary_metric=None,
            primary_metric_value=None,
            secondary_metrics={},
            error_message=msg,
            checkpoint_path=None,
            dataset_n_rows=dataset_n_rows,
            dataset_n_features=dataset_n_features,
        )


# ---------------------------------------------------------------------------
# Ray remote wrapper
# ---------------------------------------------------------------------------

def run_experiment_remote(cfg: RunConfig) -> RunResult:
    """Ray remote version of run_experiment for distributed execution."""
    try:
        import ray

        @ray.remote(
            num_gpus=1 if cfg.gpu_id is not None else 0,
            max_retries=0,
        )
        def _remote(c: RunConfig) -> RunResult:
            return run_experiment(c)

        future = _remote.remote(cfg)
        return ray.get(future)
    except ImportError:
        logger.warning("Ray not installed — falling back to local execution")
        return run_experiment(cfg)
