"""Baseline model evaluations for comparison with Ludwig benchmark results.

Baselines provide the reference performance that Ludwig configs are compared against.
Uses XGBoost and LightGBM with sensible defaults and basic hyperparameter grids.
AutoGluon support is optional (if installed).

Also supports loading precomputed baselines from TabRepo for OpenML datasets:
https://github.com/autogluon/tabrepo (219,136 CPU-hours of precomputed results)
"""
from __future__ import annotations

import json
import logging
import time
import traceback
import uuid
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

if TYPE_CHECKING:
    from benchmark.db import BenchmarkDB, RunRecord
    from benchmark.runner import RunResult

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _encode_categoricals(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    target_column: str,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Apply pd.get_dummies to categorical columns, align columns across splits."""
    feature_cols = [c for c in train_df.columns if c != target_column]

    train_x = pd.get_dummies(train_df[feature_cols])
    val_x = pd.get_dummies(val_df[feature_cols])
    test_x = pd.get_dummies(test_df[feature_cols])

    # Align: reindex val/test to match train columns, fill missing with 0
    val_x = val_x.reindex(columns=train_x.columns, fill_value=0)
    test_x = test_x.reindex(columns=train_x.columns, fill_value=0)

    return train_x, val_x, test_x


def _encode_labels(
    train_y: pd.Series,
    val_y: pd.Series,
    test_y: pd.Series,
    task_type: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict]:
    """Encode labels for classification tasks; pass through for regression."""
    if task_type == "regression":
        return (
            train_y.to_numpy(dtype=float),
            val_y.to_numpy(dtype=float),
            test_y.to_numpy(dtype=float),
            {},
        )

    # Binary / multiclass — encode to integers
    classes = sorted(train_y.unique())
    class_to_int = {c: i for i, c in enumerate(classes)}

    def encode(s: pd.Series) -> np.ndarray:
        return s.map(class_to_int).to_numpy(dtype=int)

    return encode(train_y), encode(val_y), encode(test_y), {"classes": classes}


def _primary_metric(task_type: str, y_true: np.ndarray, y_pred: np.ndarray, y_prob: np.ndarray | None) -> tuple[str, float]:
    """Return (metric_name, value) for the appropriate primary metric."""
    from sklearn.metrics import accuracy_score, r2_score, roc_auc_score

    if task_type == "binary":
        if y_prob is not None:
            score = roc_auc_score(y_true, y_prob[:, 1] if y_prob.ndim == 2 else y_prob)
        else:
            score = roc_auc_score(y_true, y_pred)
        return "roc_auc", float(score)
    elif task_type == "multiclass":
        return "accuracy", float(accuracy_score(y_true, y_pred))
    else:
        return "r2", float(r2_score(y_true, y_pred))


def _secondary_metrics(task_type: str, y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """Collect secondary metrics."""
    from sklearn.metrics import accuracy_score, mean_absolute_error, mean_squared_error

    secondary: dict = {}
    if task_type in ("binary", "multiclass"):
        secondary["accuracy"] = float(accuracy_score(y_true, y_pred))
    else:
        secondary["mae"] = float(mean_absolute_error(y_true, y_pred))
        secondary["mse"] = float(mean_squared_error(y_true, y_pred))
    return secondary


def _make_run_result(
    run_id: str,
    status: str,
    wall_seconds: float,
    primary_metric: str | None,
    primary_metric_value: float | None,
    secondary_metrics: dict,
    error_message: str | None,
    dataset_n_rows: int,
    dataset_n_features: int,
) -> "RunResult":
    from benchmark.runner import RunResult
    return RunResult(
        run_id=run_id,
        status=status,
        wall_seconds=wall_seconds,
        primary_metric=primary_metric,
        primary_metric_value=primary_metric_value,
        secondary_metrics=secondary_metrics,
        error_message=error_message,
        checkpoint_path=None,
        dataset_n_rows=dataset_n_rows,
        dataset_n_features=dataset_n_features,
    )


# ---------------------------------------------------------------------------
# XGBoost baseline
# ---------------------------------------------------------------------------

def run_xgboost_baseline(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    target_column: str,
    task_type: str,
    dataset_name: str,
    seed: int = 42,
    n_estimators: int = 500,
    early_stopping_rounds: int = 50,
) -> "RunResult":
    """Train XGBoost with default params + early stopping, evaluate on test set."""
    try:
        import xgboost as xgb
    except ImportError as exc:
        raise ImportError("xgboost is required for XGBoost baseline. Install with: pip install xgboost") from exc

    run_id = str(uuid.uuid4())
    wall_start = time.monotonic()

    try:
        dataset_n_rows = len(train_df) + len(val_df) + len(test_df)
        dataset_n_features = train_df.shape[1] - 1  # exclude target

        train_x, val_x, test_x = _encode_categoricals(train_df, val_df, test_df, target_column)
        train_y_raw = train_df[target_column]
        val_y_raw = val_df[target_column]
        test_y_raw = test_df[target_column]

        train_y, val_y, test_y, label_meta = _encode_labels(train_y_raw, val_y_raw, test_y_raw, task_type)

        if task_type == "binary":
            objective = "binary:logistic"
            eval_metric = "logloss"
            n_classes = None
        elif task_type == "multiclass":
            n_classes = len(label_meta.get("classes", []))
            objective = "multi:softprob"
            eval_metric = "mlogloss"
        else:
            objective = "reg:squarederror"
            eval_metric = "rmse"
            n_classes = None

        params: dict = {
            "n_estimators": n_estimators,
            "max_depth": 6,
            "learning_rate": 0.05,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "objective": objective,
            "eval_metric": eval_metric,
            "seed": seed,
            "verbosity": 0,
            "early_stopping_rounds": early_stopping_rounds,
        }
        if n_classes is not None:
            params["num_class"] = n_classes

        model = xgb.XGBClassifier(**params) if task_type != "regression" else xgb.XGBRegressor(**params)
        model.fit(
            train_x, train_y,
            eval_set=[(val_x, val_y)],
            verbose=False,
        )

        y_pred = model.predict(test_x)
        y_prob: np.ndarray | None = None
        if task_type in ("binary", "multiclass"):
            y_prob = model.predict_proba(test_x)

        metric_name, metric_value = _primary_metric(task_type, test_y, y_pred, y_prob)
        secondary = _secondary_metrics(task_type, test_y, y_pred)

        return _make_run_result(
            run_id=run_id,
            status="done",
            wall_seconds=time.monotonic() - wall_start,
            primary_metric=metric_name,
            primary_metric_value=metric_value,
            secondary_metrics=secondary,
            error_message=None,
            dataset_n_rows=dataset_n_rows,
            dataset_n_features=dataset_n_features,
        )

    except Exception:
        return _make_run_result(
            run_id=run_id,
            status="failed",
            wall_seconds=time.monotonic() - wall_start,
            primary_metric=None,
            primary_metric_value=None,
            secondary_metrics={},
            error_message=traceback.format_exc(),
            dataset_n_rows=0,
            dataset_n_features=0,
        )


# ---------------------------------------------------------------------------
# LightGBM baseline
# ---------------------------------------------------------------------------

def run_lightgbm_baseline(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    target_column: str,
    task_type: str,
    dataset_name: str,
    seed: int = 42,
    n_estimators: int = 500,
    early_stopping_rounds: int = 50,
) -> "RunResult":
    """Train LightGBM with default params + early stopping."""
    try:
        import lightgbm as lgb
    except ImportError as exc:
        raise ImportError("lightgbm is required for LightGBM baseline. Install with: pip install lightgbm") from exc

    run_id = str(uuid.uuid4())
    wall_start = time.monotonic()

    try:
        dataset_n_rows = len(train_df) + len(val_df) + len(test_df)
        dataset_n_features = train_df.shape[1] - 1

        train_x, val_x, test_x = _encode_categoricals(train_df, val_df, test_df, target_column)
        train_y_raw = train_df[target_column]
        val_y_raw = val_df[target_column]
        test_y_raw = test_df[target_column]

        train_y, val_y, test_y, label_meta = _encode_labels(train_y_raw, val_y_raw, test_y_raw, task_type)

        if task_type == "binary":
            objective = "binary"
            metric = "binary_logloss"
        elif task_type == "multiclass":
            objective = "multiclass"
            metric = "multi_logloss"
        else:
            objective = "regression"
            metric = "rmse"

        params: dict = {
            "n_estimators": n_estimators,
            "num_leaves": 127,
            "learning_rate": 0.05,
            "subsample": 0.8,
            "subsample_freq": 1,
            "objective": objective,
            "metric": metric,
            "seed": seed,
            "verbosity": -1,
        }
        if task_type == "multiclass":
            params["num_class"] = len(label_meta.get("classes", []))

        callbacks = [lgb.early_stopping(early_stopping_rounds, verbose=False), lgb.log_evaluation(-1)]

        model = lgb.LGBMClassifier(**params) if task_type != "regression" else lgb.LGBMRegressor(**params)
        model.fit(
            train_x, train_y,
            eval_set=[(val_x, val_y)],
            callbacks=callbacks,
        )

        y_pred = model.predict(test_x)
        y_prob: np.ndarray | None = None
        if task_type in ("binary", "multiclass"):
            y_prob = model.predict_proba(test_x)

        metric_name, metric_value = _primary_metric(task_type, test_y, y_pred, y_prob)
        secondary = _secondary_metrics(task_type, test_y, y_pred)

        return _make_run_result(
            run_id=run_id,
            status="done",
            wall_seconds=time.monotonic() - wall_start,
            primary_metric=metric_name,
            primary_metric_value=metric_value,
            secondary_metrics=secondary,
            error_message=None,
            dataset_n_rows=dataset_n_rows,
            dataset_n_features=dataset_n_features,
        )

    except Exception:
        return _make_run_result(
            run_id=run_id,
            status="failed",
            wall_seconds=time.monotonic() - wall_start,
            primary_metric=None,
            primary_metric_value=None,
            secondary_metrics={},
            error_message=traceback.format_exc(),
            dataset_n_rows=0,
            dataset_n_features=0,
        )


# ---------------------------------------------------------------------------
# AutoGluon baseline
# ---------------------------------------------------------------------------

def run_autogluon_baseline(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    target_column: str,
    dataset_name: str,
    time_limit: int = 3600,
    seed: int = 42,
) -> "RunResult":
    """Run AutoGluon TabularPredictor if installed. Raises ImportError if not."""
    try:
        from autogluon.tabular import TabularPredictor
    except ImportError as exc:
        raise ImportError(
            "autogluon.tabular is required for AutoGluon baseline. "
            "Install with: pip install autogluon.tabular"
        ) from exc

    run_id = str(uuid.uuid4())
    wall_start = time.monotonic()

    try:
        dataset_n_rows = len(train_df) + len(test_df)
        dataset_n_features = train_df.shape[1] - 1

        predictor = TabularPredictor(
            label=target_column,
            verbosity=0,
        ).fit(
            train_data=train_df,
            time_limit=time_limit,
            random_state=seed,
        )

        leaderboard = predictor.leaderboard(test_df, silent=True)
        best_score = leaderboard["score_test"].iloc[0] if not leaderboard.empty else None
        metric_name = predictor.eval_metric.name if hasattr(predictor.eval_metric, "name") else str(predictor.eval_metric)

        secondary: dict = {}
        if not leaderboard.empty:
            secondary["n_models"] = int(len(leaderboard))
            secondary["best_model"] = str(leaderboard["model"].iloc[0])

        return _make_run_result(
            run_id=run_id,
            status="done",
            wall_seconds=time.monotonic() - wall_start,
            primary_metric=metric_name,
            primary_metric_value=float(best_score) if best_score is not None else None,
            secondary_metrics=secondary,
            error_message=None,
            dataset_n_rows=dataset_n_rows,
            dataset_n_features=dataset_n_features,
        )

    except Exception:
        return _make_run_result(
            run_id=run_id,
            status="failed",
            wall_seconds=time.monotonic() - wall_start,
            primary_metric=None,
            primary_metric_value=None,
            secondary_metrics={},
            error_message=traceback.format_exc(),
            dataset_n_rows=0,
            dataset_n_features=0,
        )


# ---------------------------------------------------------------------------
# TabRepo loader
# ---------------------------------------------------------------------------

_TABREPO_RELEASE_URL = (
    "https://github.com/autogluon/tabrepo/releases/download/v1.0.0/tabrepo_results.parquet"
)


def load_tabrepo_baselines(
    openml_task_ids: list[int],
    tabrepo_path: str | None = None,
) -> pd.DataFrame:
    """Load precomputed baselines from TabRepo for the given OpenML task IDs.

    Returns a DataFrame with columns: dataset_name, model, metric, value.
    If tabrepo_path is None, tries to download from the TabRepo GitHub release.
    TabRepo GitHub: https://github.com/autogluon/tabrepo
    """
    import tempfile
    from pathlib import Path

    if tabrepo_path is None:
        # Try importing tabrepo package first
        try:
            import tabrepo  # type: ignore[import]
            df = tabrepo.load_results()
        except ImportError:
            logger.info("tabrepo package not found; downloading from GitHub release: %s", _TABREPO_RELEASE_URL)
            import urllib.request
            with tempfile.NamedTemporaryFile(suffix=".parquet", delete=False) as f:
                tmp_path = f.name
            try:
                urllib.request.urlretrieve(_TABREPO_RELEASE_URL, tmp_path)
                df = pd.read_parquet(tmp_path)
            finally:
                Path(tmp_path).unlink(missing_ok=True)
    else:
        p = Path(tabrepo_path)
        if p.is_dir():
            # Try common filenames inside the directory
            candidates = ["results.parquet", "tabrepo_results.parquet", "metadata.parquet"]
            for name in candidates:
                candidate_path = p / name
                if candidate_path.exists():
                    df = pd.read_parquet(candidate_path)
                    break
            else:
                # Merge all parquet files in the directory
                parts = list(p.glob("*.parquet"))
                if not parts:
                    raise FileNotFoundError(f"No parquet files found in {tabrepo_path}")
                df = pd.concat([pd.read_parquet(f) for f in sorted(parts)], ignore_index=True)
        else:
            df = pd.read_parquet(p)

    # Normalize column names — TabRepo uses various schemas across versions
    col_map: dict[str, str] = {}
    for col in df.columns:
        lower = col.lower()
        if lower in ("tid", "task_id", "openml_task_id"):
            col_map[col] = "task_id"
        elif lower in ("framework", "model", "method", "algorithm"):
            col_map[col] = "model"
        elif lower in ("metric", "metric_name"):
            col_map[col] = "metric"
        elif lower in ("result", "value", "score", "metric_value"):
            col_map[col] = "value"
        elif lower in ("dataset", "dataset_name"):
            col_map[col] = "dataset_name"
    if col_map:
        df = df.rename(columns=col_map)

    # Filter to requested task IDs
    if "task_id" in df.columns:
        df = df[df["task_id"].isin(openml_task_ids)].copy()
        # Synthesize dataset_name from task_id if not present
        if "dataset_name" not in df.columns:
            df["dataset_name"] = df["task_id"].apply(lambda t: f"openml_task_{t}")
    else:
        logger.warning("Could not find a task_id column in TabRepo data; returning all rows")

    # Ensure required columns exist
    for required in ("dataset_name", "model", "metric", "value"):
        if required not in df.columns:
            df[required] = None

    return df[["dataset_name", "model", "metric", "value"]].reset_index(drop=True)


# ---------------------------------------------------------------------------
# DB integration helpers
# ---------------------------------------------------------------------------

def _result_to_run_record(
    result: "RunResult",
    dataset_name: str,
    dataset_source: str,
    dataset_n_rows: int,
    dataset_n_features: int,
    config_hash: str,
    combiner: str,
    seed: int,
) -> "RunRecord":
    from benchmark.db import RunRecord

    return RunRecord(
        run_id=result.run_id,
        dataset_name=dataset_name,
        dataset_source=dataset_source,
        dataset_n_rows=result.dataset_n_rows or dataset_n_rows,
        dataset_n_features=result.dataset_n_features or dataset_n_features,
        config_hash=config_hash,
        combiner=combiner,
        input_encoders="{}",
        output_decoder="",
        learning_rate=0.0,
        batch_size=0,
        n_epochs=0,
        seed=seed,
        status=result.status,
        start_time=None,
        end_time=None,
        wall_seconds=result.wall_seconds,
        gpu_type="cpu",
        primary_metric=result.primary_metric or "",
        primary_metric_value=result.primary_metric_value,
        secondary_metrics=json.dumps(result.secondary_metrics),
        error_message=result.error_message or "",
        checkpoint_path=result.checkpoint_path or "",
    )


def _load_dataset_for_baseline(
    dataset_name: str,
    dataset_registry: dict,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Load a dataset using the registry entry."""
    entry = dataset_registry.get(dataset_name, {})
    source = entry.get("source", "path")

    if source == "openml":
        from benchmark.runner import _load_dataset_openml
        task_id = entry.get("openml_task_id")
        if task_id is None:
            raise ValueError(f"Registry entry for {dataset_name!r} missing openml_task_id")
        return _load_dataset_openml(task_id)
    elif source == "ludwig":
        from benchmark.runner import _load_dataset_ludwig
        return _load_dataset_ludwig(dataset_name)
    else:
        from benchmark.runner import _load_dataset_path
        path = entry.get("path") or entry.get("dataset_path")
        if path is None:
            raise ValueError(f"Registry entry for {dataset_name!r} missing path")
        return _load_dataset_path(path)


# ---------------------------------------------------------------------------
# run_all_baselines
# ---------------------------------------------------------------------------

def run_all_baselines(
    db: "BenchmarkDB",
    dataset_registry: dict,
    target_column_map: dict,
    task_type_map: dict,
    run_xgboost: bool = True,
    run_lgbm: bool = True,
    run_autogluon: bool = False,
    time_limit_s: int = 1800,
) -> None:
    """Run baselines for all datasets and write results to BenchmarkDB."""

    datasets = list(dataset_registry.keys())
    logger.info("Running baselines on %d datasets", len(datasets))

    for dataset_name in datasets:
        entry = dataset_registry[dataset_name]
        target_column = target_column_map.get(dataset_name)
        task_type = task_type_map.get(dataset_name)

        if target_column is None:
            logger.warning("Skipping %s — no target_column in target_column_map", dataset_name)
            continue
        if task_type is None:
            logger.warning("Skipping %s — no task_type in task_type_map", dataset_name)
            continue

        logger.info("Loading dataset: %s", dataset_name)
        try:
            train_df, val_df, test_df = _load_dataset_for_baseline(dataset_name, dataset_registry)
        except Exception as exc:
            logger.error("Failed to load %s: %s", dataset_name, exc)
            continue

        dataset_n_rows = len(train_df) + len(val_df) + len(test_df)
        dataset_n_features = train_df.shape[1] - 1
        dataset_source = entry.get("source", "path")

        # --- XGBoost ---
        if run_xgboost:
            logger.info("[%s] Running XGBoost baseline", dataset_name)
            try:
                result = run_xgboost_baseline(
                    train_df=train_df,
                    val_df=val_df,
                    test_df=test_df,
                    target_column=target_column,
                    task_type=task_type,
                    dataset_name=dataset_name,
                )
                record = _result_to_run_record(
                    result=result,
                    dataset_name=dataset_name,
                    dataset_source=dataset_source,
                    dataset_n_rows=dataset_n_rows,
                    dataset_n_features=dataset_n_features,
                    config_hash="baseline_xgboost",
                    combiner="baseline_xgboost",
                    seed=42,
                )
                db.upsert_run(record)
                logger.info(
                    "[%s] XGBoost done: %s=%.4f (%.1fs)",
                    dataset_name,
                    result.primary_metric,
                    result.primary_metric_value or 0.0,
                    result.wall_seconds,
                )
            except ImportError as exc:
                logger.warning("[%s] XGBoost skipped: %s", dataset_name, exc)
            except Exception as exc:
                logger.error("[%s] XGBoost failed: %s", dataset_name, exc)

        # --- LightGBM ---
        if run_lgbm:
            logger.info("[%s] Running LightGBM baseline", dataset_name)
            try:
                result = run_lightgbm_baseline(
                    train_df=train_df,
                    val_df=val_df,
                    test_df=test_df,
                    target_column=target_column,
                    task_type=task_type,
                    dataset_name=dataset_name,
                )
                record = _result_to_run_record(
                    result=result,
                    dataset_name=dataset_name,
                    dataset_source=dataset_source,
                    dataset_n_rows=dataset_n_rows,
                    dataset_n_features=dataset_n_features,
                    config_hash="baseline_lightgbm",
                    combiner="baseline_lightgbm",
                    seed=42,
                )
                db.upsert_run(record)
                logger.info(
                    "[%s] LightGBM done: %s=%.4f (%.1fs)",
                    dataset_name,
                    result.primary_metric,
                    result.primary_metric_value or 0.0,
                    result.wall_seconds,
                )
            except ImportError as exc:
                logger.warning("[%s] LightGBM skipped: %s", dataset_name, exc)
            except Exception as exc:
                logger.error("[%s] LightGBM failed: %s", dataset_name, exc)

        # --- AutoGluon ---
        if run_autogluon:
            logger.info("[%s] Running AutoGluon baseline (time_limit=%ds)", dataset_name, time_limit_s)
            try:
                # AutoGluon handles its own train/val split — merge train+val
                full_train_df = pd.concat([train_df, val_df], ignore_index=True)
                result = run_autogluon_baseline(
                    train_df=full_train_df,
                    test_df=test_df,
                    target_column=target_column,
                    dataset_name=dataset_name,
                    time_limit=time_limit_s,
                )
                record = _result_to_run_record(
                    result=result,
                    dataset_name=dataset_name,
                    dataset_source=dataset_source,
                    dataset_n_rows=dataset_n_rows,
                    dataset_n_features=dataset_n_features,
                    config_hash="baseline_autogluon",
                    combiner="baseline_autogluon",
                    seed=42,
                )
                db.upsert_run(record)
                logger.info(
                    "[%s] AutoGluon done: %s=%.4f (%.1fs)",
                    dataset_name,
                    result.primary_metric,
                    result.primary_metric_value or 0.0,
                    result.wall_seconds,
                )
            except ImportError as exc:
                logger.warning("[%s] AutoGluon skipped: %s", dataset_name, exc)
            except Exception as exc:
                logger.error("[%s] AutoGluon failed: %s", dataset_name, exc)
