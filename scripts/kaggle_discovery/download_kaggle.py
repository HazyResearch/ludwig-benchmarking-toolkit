"""Download ML-ready Kaggle datasets to local disk.

Usage:
    python scripts/kaggle_discovery/download_kaggle.py \\
        --input ml_ready.json --output-dir data/kaggle --max-datasets 200

Options:
    --input         Path to ml_ready.json from filter_kaggle.py
    --output-dir    Directory to save downloaded datasets
    --max-datasets  Maximum number to download (default: all)
    --skip-existing Skip datasets already downloaded
"""
from __future__ import annotations

import argparse
import json
import logging
import time
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

_MANIFEST_FILENAME = "downloaded.json"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _safe_dir_name(ref: str) -> str:
    """Convert a Kaggle ref (user/dataset) to a safe directory name."""
    return ref.replace("/", "_").replace("-", "_")


def _find_largest_csv(directory: Path) -> Path | None:
    """Return the largest CSV file in *directory* (recursively), or None."""
    csv_files = list(directory.rglob("*.csv"))
    if not csv_files:
        return None
    return max(csv_files, key=lambda p: p.stat().st_size)


def _load_manifest(output_dir: Path) -> dict[str, str]:
    """Load the download manifest (ref → local_path), returning empty dict if absent."""
    manifest_path = output_dir / _MANIFEST_FILENAME
    if manifest_path.exists():
        with manifest_path.open() as f:
            return json.load(f)
    return {}


def _save_manifest(output_dir: Path, manifest: dict[str, str]) -> None:
    """Write the download manifest to disk."""
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = output_dir / _MANIFEST_FILENAME
    with manifest_path.open("w") as f:
        json.dump(manifest, f, indent=2)


# ---------------------------------------------------------------------------
# Core downloader
# ---------------------------------------------------------------------------

def download_kaggle_datasets(
    ml_ready_json: str | Path,
    output_dir: str | Path,
    max_datasets: int | None = None,
    skip_existing: bool = True,
    sleep_between: float = 0.5,
) -> dict[str, str]:
    """Download ML-ready Kaggle datasets to local disk.

    For each dataset in *ml_ready_json*:
    1. Creates a subdirectory under *output_dir* named after the dataset ref.
    2. Calls kaggle.api.dataset_download_files(ref, path=..., unzip=True).
    3. Finds the largest CSV file in the downloaded contents.
    4. Records the mapping in the download manifest.

    Returns the manifest dict (ref → local CSV path).
    """
    try:
        import kaggle
    except ImportError:
        raise RuntimeError("kaggle package required: pip install kaggle")

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load existing manifest (for skip_existing)
    manifest: dict[str, str] = _load_manifest(output_dir)

    # Load filtered dataset list
    with open(ml_ready_json) as f:
        datasets: list[dict] = json.load(f)

    if max_datasets is not None:
        datasets = datasets[:max_datasets]

    logger.info("Datasets to process: %d", len(datasets))

    api = kaggle.KaggleApi()
    api.authenticate()

    n_success = 0
    n_skip = 0
    n_fail = 0

    for ds in datasets:
        ref = ds.get("ref", "")
        if not ref:
            logger.warning("Skipping entry with missing 'ref': %s", ds)
            n_fail += 1
            continue

        # Skip if already downloaded
        if skip_existing and ref in manifest:
            existing = manifest[ref]
            if existing and Path(existing).exists():
                logger.debug("Skipping already downloaded: %s → %s", ref, existing)
                n_skip += 1
                continue

        dest_dir = output_dir / _safe_dir_name(ref)
        dest_dir.mkdir(parents=True, exist_ok=True)

        logger.info("Downloading: %s → %s", ref, dest_dir)
        try:
            api.dataset_download_files(ref, path=str(dest_dir), unzip=True, quiet=False)
        except Exception as exc:
            logger.error("Download failed for %s: %s", ref, exc)
            n_fail += 1
            # Save manifest on each failure so progress is not lost
            _save_manifest(output_dir, manifest)
            time.sleep(sleep_between)
            continue

        # Find the largest CSV in the downloaded directory
        csv_path = _find_largest_csv(dest_dir)
        if csv_path is None:
            logger.warning("No CSV file found after downloading %s (dir=%s)", ref, dest_dir)
            # Record the directory itself so we can debug later
            manifest[ref] = str(dest_dir)
        else:
            logger.info("  Found CSV: %s (%.1f MB)", csv_path.name, csv_path.stat().st_size / 1e6)
            manifest[ref] = str(csv_path)

        n_success += 1
        _save_manifest(output_dir, manifest)
        time.sleep(sleep_between)

    # Final manifest save
    _save_manifest(output_dir, manifest)

    logger.info(
        "Done. Success=%d, Skipped=%d, Failed=%d. Manifest: %s",
        n_success, n_skip, n_fail, output_dir / _MANIFEST_FILENAME,
    )
    return manifest


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Download ML-ready Kaggle datasets to local disk.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--input",
        required=True,
        help="Path to ml_ready.json from filter_kaggle.py",
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        help="Directory to save downloaded datasets",
    )
    parser.add_argument(
        "--max-datasets",
        type=int,
        default=None,
        help="Maximum number of datasets to download (default: all)",
    )
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        default=True,
        help="Skip datasets that are already in the manifest (default: True)",
    )
    parser.add_argument(
        "--no-skip-existing",
        dest="skip_existing",
        action="store_false",
        help="Re-download datasets even if already present",
    )
    parser.add_argument(
        "--sleep",
        type=float,
        default=0.5,
        help="Seconds to sleep between downloads (default: 0.5)",
    )
    return parser


if __name__ == "__main__":
    parser = _build_parser()
    args = parser.parse_args()

    manifest = download_kaggle_datasets(
        ml_ready_json=args.input,
        output_dir=args.output_dir,
        max_datasets=args.max_datasets,
        skip_existing=args.skip_existing,
        sleep_between=args.sleep,
    )

    print(f"\nManifest ({len(manifest)} entries) written to: {Path(args.output_dir) / _MANIFEST_FILENAME}")
