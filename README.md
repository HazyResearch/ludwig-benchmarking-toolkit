# Ludwig Mega-AutoML Benchmark

Run hundreds of tabular datasets against 100 sampled Ludwig configs each, collect results in a
Parquet/DuckDB store, and explore them through a structured JSON dashboard.

The full pipeline is: register datasets → generate configs → run experiments → export dashboard.

---

## Quick Start

```bash
# 1. Register and prepare the OpenML-CC18 suite (72 datasets, 100 configs each)
python scripts/prepare_benchmark.py \
    --openml-suite 99 \
    --registry benchmark/dataset_registry.json \
    --configs-dir benchmark/configs

# 2. (Optional) Estimate cost before launching
python -c "
from benchmark.cost_estimator import estimate_from_registry_file, print_cost_report
print_cost_report(estimate_from_registry_file('benchmark/dataset_registry.json'))
"

# 3. Run the benchmark (sequential; use --ray for distributed)
python scripts/run_benchmark.py \
    --registry benchmark/dataset_registry.json \
    --configs-dir benchmark/configs \
    --results-dir benchmark/results

# 4. Export dashboard data and serve
python -m benchmark.exporter \
    --results-dir benchmark/results \
    --output-dir dashboard \
    --registry benchmark/dataset_registry.json
python dashboard/serve.py --data-dir dashboard/data
```

---

## Full Pipeline Walkthrough

### 1. Register Datasets

The dataset registry lives at `benchmark/dataset_registry.json`. Each entry is a `DatasetEntry`
with fields: `name`, `source`, `openml_task_id`, `kaggle_ref`, `local_path`, `target_column`,
`task_type`, `priority`, `seed`, etc.

**OpenML benchmark suites**

```bash
# CC18 = suite 99 (72 classification datasets)
# CTR23 = suite 353 (tabular regression/classification)
python scripts/prepare_benchmark.py --openml-suite 99 \
    --registry benchmark/dataset_registry.json \
    --configs-dir benchmark/configs
```

**Ludwig built-in datasets**

```bash
python scripts/prepare_benchmark.py --ludwig-builtins \
    --registry benchmark/dataset_registry.json \
    --configs-dir benchmark/configs
```

**Kaggle datasets** (requires `~/.kaggle/kaggle.json` credentials)

```bash
# Step 1: scrape the Kaggle catalog
python scripts/kaggle_discovery/scrape_kaggle.py \
    --pages 500 --output kaggle_registry.json

# Step 2: filter to open-license, ML-ready datasets
python scripts/kaggle_discovery/filter_kaggle.py \
    --input kaggle_registry.json --output ml_ready.json

# Step 3: download filtered datasets
python scripts/kaggle_discovery/download_kaggle.py \
    --input ml_ready.json --dest data/kaggle/

# Step 4: register them
python -c "
from benchmark.dataset_registry import DatasetRegistry, register_kaggle_filtered
r = DatasetRegistry('benchmark/dataset_registry.json')
register_kaggle_filtered(r, 'ml_ready.json', 'data/kaggle/')
r.save()
"
```

**Local CSV/Parquet**

```python
from benchmark.dataset_registry import DatasetRegistry, DatasetEntry
r = DatasetRegistry("benchmark/dataset_registry.json")
r.add(DatasetEntry(
    name="my_dataset",
    source="path",
    local_path="/abs/path/to/my_dataset.csv",
    target_column="label",
    task_type="binary",
))
r.save()
```

---

### 2. Generate Configs

`scripts/prepare_benchmark.py` handles registration and config generation in one pass. To
regenerate configs independently, or for a single dataset, use `scripts/generate_configs.py`.

```bash
# Regenerate configs for one dataset
python scripts/generate_configs.py \
    --registry benchmark/dataset_registry.json \
    --dataset openml_task_7592 \
    --configs-dir benchmark/configs \
    --n 100 --force

# Generate configs directly from a CSV (no registry entry needed)
python scripts/generate_configs.py \
    --csv mydata.csv --target label \
    --configs-dir benchmark/configs --n 100
```

Configs are written as JSONL files: `benchmark/configs/{dataset_name}/configs.jsonl`, one Ludwig
config dict per line.

Key flags for `prepare_benchmark.py`:

| Flag | Default | Description |
|---|---|---|
| `--n` | 100 | Configs to generate per dataset |
| `--seed` | 42 | Random seed for config sampling |
| `--resume` | off | Skip datasets that already have `configs.jsonl` |
| `--dry-run` | off | Run the pipeline without writing any files |
| `--skip-quality-check` | off | Skip the min-rows/columns sanity check |

---

### 3. Run the Benchmark

The scheduler reads the registry and config files, builds a SQLite job queue
(`benchmark/jobs.db`), and dispatches experiments. Job state persists across restarts; failed jobs
are retried up to 3 times.

```bash
# Sequential (single machine)
python scripts/run_benchmark.py \
    --registry benchmark/dataset_registry.json \
    --configs-dir benchmark/configs \
    --results-dir benchmark/results \
    --time-limit 1800

# Distributed via Ray
python scripts/run_benchmark.py \
    --registry benchmark/dataset_registry.json \
    --configs-dir benchmark/configs \
    --results-dir benchmark/results \
    --ray --max-concurrent 16 --gpus-per-trial 1.0

# Monitor progress while running
python -c "
from benchmark.db import BenchmarkDB
from benchmark.scheduler import BenchmarkScheduler
from benchmark.dashboard import print_progress
db = BenchmarkDB('benchmark/results')
# scheduler is reconstructed from the existing jobs.db
"
```

Each experiment enforces a per-job wall-time limit (default 30 min). Runs that exceed the limit
return `status="timeout"`; out-of-memory failures return `status="oom"`.

---

### 4. View Results

Export the Parquet store to the JSON hierarchy that the dashboard reads, then start the server.

```bash
python -m benchmark.exporter \
    --results-dir benchmark/results \
    --output-dir dashboard \
    --registry benchmark/dataset_registry.json

# Serve the dashboard (static files)
python dashboard/serve.py --data-dir dashboard/data
```

Or query results programmatically:

```python
from benchmark.db import BenchmarkDB

db = BenchmarkDB("benchmark/results")
print(db.progress_summary())
print(db.best_per_dataset())
print(db.combiner_win_rates())
```

---

## Directory Structure

```
ludwig-benchmark/
├── benchmark/                  # Core library
│   ├── dataset_registry.py     # DatasetEntry, DatasetRegistry, register_* helpers
│   ├── runner.py               # run_experiment() — single Ludwig training run
│   ├── scheduler.py            # BenchmarkScheduler — SQLite job queue + dispatch
│   ├── db.py                   # BenchmarkDB — Parquet/DuckDB results store
│   ├── exporter.py             # export_dashboard() — JSON hierarchy for the UI
│   ├── baselines.py            # XGBoost / LightGBM / AutoGluon baseline runners
│   ├── cost_estimator.py       # GPU-hour and cloud cost estimator
│   ├── dashboard.py            # Terminal progress dashboard (Rich)
│   └── configs/                # Generated JSONL config files (one dir per dataset)
├── scripts/
│   ├── prepare_benchmark.py    # End-to-end: register + quality check + generate configs
│   ├── generate_configs.py     # Config generation only (registry or raw CSV)
│   └── kaggle_discovery/
│       ├── scrape_kaggle.py    # Paginate Kaggle API, save metadata JSON
│       ├── filter_kaggle.py    # Filter scraped catalog to open-license ML-ready datasets
│       └── download_kaggle.py  # Download filtered datasets to local disk
├── dashboard/
│   └── serve.py                # Static file server for the benchmark dashboard
├── tests/
│   └── test_exporter.py        # Unit tests for benchmark.exporter
└── data/                       # Generated JSON output for the dashboard (see below)
```

---

## Module Reference

| Module | What it does |
|---|---|
| `benchmark.dataset_registry` | Persists dataset metadata as JSON; helpers to bulk-register OpenML suites, Ludwig builtins, and Kaggle catalogs |
| `benchmark.runner` | Loads a dataset, calls `LudwigModel.train()` + `.evaluate()`, enforces a wall-time timeout, returns a `RunResult` |
| `benchmark.scheduler` | Manages a SQLite job queue; dispatches jobs sequentially or via Ray; retries on failure |
| `benchmark.db` | Writes one Parquet file per run; rebuilds a consolidated index on demand; supports DuckDB analytical queries |
| `benchmark.exporter` | Reads the Parquet store and writes the full `data/` JSON hierarchy consumed by the dashboard |
| `benchmark.baselines` | Trains XGBoost, LightGBM, and (optionally) AutoGluon on the same splits; writes results to `BenchmarkDB` |
| `benchmark.cost_estimator` | Estimates total GPU-hours and AWS spot cost before launching a run |
| `benchmark.dashboard` | Live terminal progress view (Rich); shows scheduler queue state, best-per-dataset table, combiner win rates |
| `scripts.prepare_benchmark` | One-shot CLI: registers datasets, runs quality checks, detects target columns, generates configs |
| `scripts.generate_configs` | Standalone config generation; can operate from registry or directly from a CSV file |
| `scripts.kaggle_discovery.scrape_kaggle` | Paginates Kaggle API, saves raw metadata with checkpointing |
| `scripts.kaggle_discovery.filter_kaggle` | Filters by votes, downloads, license (CC0/CC-BY/ODbL), and file size |

---

## Dataset Sources

| Source | How to register | Notes |
|---|---|---|
| OpenML CC18 | `--openml-suite 99` | 72 classification tasks; requires `pip install openml` |
| OpenML CTR23 | `--openml-suite 353` | Tabular classification + regression benchmark |
| Ludwig builtins | `--ludwig-builtins` | All datasets from `ludwig.datasets.list_datasets()` |
| Kaggle | `scrape_kaggle.py` → `filter_kaggle.py` → `download_kaggle.py` | Requires Kaggle API credentials |
| Local CSV/Parquet | `DatasetEntry(source="path", local_path=...)` | Absolute path; target column must be set manually |

---

## Kaggle Setup

Some datasets (tagged `kaggle_credentials_required` in `dataset_metadata.yaml`) are hosted on
Kaggle competitions and require two things: API credentials and accepted competition rules.

### 1. Get API credentials

1. Go to **https://www.kaggle.com/settings**
2. Scroll to the **API** section and click **"Create New Token"**
3. Save the downloaded `kaggle.json` to `~/.kaggle/kaggle.json`
4. Restrict permissions: `chmod 600 ~/.kaggle/kaggle.json`

Alternatively, export the two values as environment variables:

```bash
export KAGGLE_USERNAME=your_username
export KAGGLE_KEY=your_api_key
```

### 2. Accept competition rules

Each Kaggle competition requires you to read and accept its rules in the browser before the
API allows downloads. The rules page is at:

```
https://www.kaggle.com/competitions/<competition-slug>/rules
```

**The smoke test handles this interactively.** When it encounters a dataset whose competition
rules have not been accepted yet, it will:

1. Print the rules URL in the terminal
2. Wait for you to open the URL, click **"I Understand and Accept"**, and press Enter
3. Retry the download — if it still fails (e.g. Kaggle propagation delay), it will prompt again

Example session:

```
ieee_fraud    ...
  Competition rules not accepted for 'ieee_fraud'.

  1. Open this URL in your browser:
     https://www.kaggle.com/competitions/ieee-fraud-detection/rules
  2. Click 'I Understand and Accept'

  Press Enter once you have accepted the rules...

ieee_fraud    PASS    142.3s
```

### 3. Skip Kaggle datasets

If you want to run only non-Kaggle datasets (e.g. in CI or headless environments), pass
`--skip-kaggle`:

```bash
python scripts/smoke_test.py --metadata-yaml dataset_metadata.yaml --skip-kaggle
```

### Kaggle competition datasets in this benchmark

The following datasets in `dataset_metadata.yaml` require Kaggle credentials + competition
rule acceptance:

| Dataset | Competition |
|---|---|
| `bbcnews` | `learn-ai-bbc` |
| `ames_housing` | `house-prices-advanced-regression-techniques` |
| `allstate_claims_severity` | `allstate-claims-severity` |
| `ieee_fraud` | `ieee-fraud-detection` |
| `otto_group_product` | `otto-group-product-classification-challenge` |
| `porto_seguro_safe_driver` | `porto-seguro-safe-driver-prediction` |
| `santander_customer_satisfaction` | `santander-customer-satisfaction` |
| `santander_customer_transaction` | `santander-customer-transaction-prediction` |
| `santander_value_prediction` | `santander-value-prediction-challenge` |
| `mercedes_benz_greener` | `mercedes-benz-greener-manufacturing` |
| `amazon_employee_access_challenge` | `amazon-employee-access-challenge` |
| `walmart_recruiting` | `walmart-recruiting-trip-type-classification` |
| `bnp_claims_management` | `bnp-paribas-cardif-claims-management` |
| `customer_churn_prediction` | `customer-churn-prediction-2020` |

---

## Smoke Test

`scripts/smoke_test.py` validates that every dataset in `dataset_metadata.yaml` can be loaded
and trained end-to-end (1 epoch, minimal concat model, up to 1000 rows).

```bash
# Run all datasets (Kaggle competitions prompt interactively for rule acceptance)
python scripts/smoke_test.py --metadata-yaml dataset_metadata.yaml --gpu-id 0

# Run only a subset
python scripts/smoke_test.py --metadata-yaml dataset_metadata.yaml \
    --datasets titanic adult_census_income openml_task_37

# Skip all Kaggle competition datasets (headless / CI mode)
python scripts/smoke_test.py --metadata-yaml dataset_metadata.yaml --skip-kaggle

# Auto-discover all Ludwig built-ins (no YAML required)
python scripts/smoke_test.py
```

| Flag | Default | Description |
|---|---|---|
| `--metadata-yaml` | (auto-discover) | Path to `dataset_metadata.yaml`; if omitted, discovers all Ludwig built-ins |
| `--datasets` | all | Space-separated list of dataset names to test |
| `--gpu-id` | CPU | CUDA device index to use |
| `--skip-kaggle` | off | Skip datasets tagged `kaggle_credentials_required` without prompting |
| `--skip-sources` | none | Skip datasets whose `source` field matches any of these values |

**Exit codes:** `0` = all tested datasets passed; `1` = at least one failure.

---

## Output Format

`benchmark.exporter.export_dashboard()` writes the following files under `{output_dir}/data/`:

```
data/
├── summary.json            # Global stats: n_datasets, n_runs, completion rate, top combiners
├── datasets.json           # List of per-dataset summary rows (for the dataset list page)
├── combiners.json          # Per-combiner aggregate stats: win rate, mean rank, score distribution
├── configs.json            # Per-config-hash aggregate stats: n_wins, mean_rank, mean_score
├── datasets/
│   └── {name}.json         # Full ranked run list for one dataset + baseline scores
├── configs/
│   └── {config_hash}.json  # Cross-dataset performance profile for one config
└── runs/
    └── {run_id}.json       # Individual run detail: all metrics, hyperparams, timing
```

The dashboard loads `summary.json` and `datasets.json` on the landing page. Dataset detail loads
`datasets/{name}.json`. Config detail loads `configs/{hash}.json`.

---

## Requirements

**Core**

- Python 3.10+
- `ludwig` (installed from source)
- `pandas`, `numpy`, `pyarrow`
- `duckdb >= 0.10.0`
- `filelock >= 3.12.0`
- `rich >= 13.0.0` (optional — plain-text fallback if absent)

**Dataset sources** (install as needed)

- `openml >= 0.14.0` — OpenML suites
- `kaggle >= 1.6.0` — Kaggle catalog scraping and download

**Baselines** (optional)

- `xgboost` — XGBoost baseline
- `lightgbm` — LightGBM baseline
- `autogluon.tabular` — AutoGluon baseline

**Distributed execution** (optional)

- `ray` — distributed job execution via `--ray` flag
