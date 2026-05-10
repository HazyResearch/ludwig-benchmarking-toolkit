"""Scrapes Kaggle dataset catalog and saves metadata to kaggle_registry.json.

Usage:
    python scrape_kaggle.py --pages 500 --output kaggle_registry.json
    python scrape_kaggle.py --min-size 50000 --max-size 500000000 --tags 14101
"""
from __future__ import annotations

import argparse
import json
import logging
import time
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Core scraper
# ---------------------------------------------------------------------------

def scrape_kaggle_datasets(
    pages: int = 500,
    file_type: str = "csv",
    tags: str = "14101",
    min_size: int = 50_000,
    max_size: int = 500_000_000,
    output_path: str = "kaggle_registry.json",
    sleep_between_pages: float = 1.0,
) -> list[dict]:
    """Paginates through Kaggle API and collects dataset metadata.

    Each entry: {ref, title, size, downloads, votes, updated, license, tags}
    Saves checkpoint every 10 pages to avoid losing progress.
    """
    try:
        import kaggle
    except ImportError:
        raise RuntimeError("kaggle package not installed: pip install kaggle")

    output_path_p = Path(output_path)
    checkpoint_path = output_path_p.with_suffix(".checkpoint.json")

    # Resume from checkpoint if available
    collected: list[dict] = []
    start_page = 1
    if checkpoint_path.exists():
        try:
            with checkpoint_path.open() as f:
                checkpoint = json.load(f)
            collected = checkpoint.get("results", [])
            start_page = checkpoint.get("next_page", 1)
            logger.info("Resuming from checkpoint at page %d (%d results so far)", start_page, len(collected))
        except Exception as exc:
            logger.warning("Could not load checkpoint: %s — starting fresh", exc)
            collected = []
            start_page = 1

    api = kaggle.KaggleApi()
    api.authenticate()

    seen_refs: set[str] = {d["ref"] for d in collected}

    for page in range(start_page, pages + 1):
        logger.info("Fetching page %d / %d ...", page, pages)
        try:
            results = api.dataset_list(
                file_type=file_type,
                tag_ids=tags,
                page=page,
                max_size=max_size,
                min_size=min_size,
                sort_by="votes",
            )
        except Exception as exc:
            logger.error("Error on page %d: %s — retrying once after 5s", page, exc)
            time.sleep(5.0)
            try:
                results = api.dataset_list(
                    file_type=file_type,
                    tag_ids=tags,
                    page=page,
                    max_size=max_size,
                    min_size=min_size,
                    sort_by="votes",
                )
            except Exception as exc2:
                logger.error("Second failure on page %d: %s — skipping", page, exc2)
                continue

        if not results:
            logger.info("Empty page at %d — stopping early", page)
            break

        for ds in results:
            ref = getattr(ds, "ref", None) or str(ds)
            if ref in seen_refs:
                continue
            seen_refs.add(ref)

            # Extract license info
            license_name = ""
            try:
                lic = ds.licenseName if hasattr(ds, "licenseName") else ""
                license_name = str(lic).lower().strip() if lic else ""
            except Exception:
                pass

            # Extract tags
            tag_list: list[str] = []
            try:
                raw_tags = ds.tags if hasattr(ds, "tags") else []
                tag_list = [str(t) for t in (raw_tags or [])]
            except Exception:
                pass

            entry = {
                "ref": ref,
                "title": str(getattr(ds, "title", "") or ""),
                "size": int(getattr(ds, "totalBytes", 0) or 0),
                "downloads": int(getattr(ds, "downloadCount", 0) or 0),
                "votes": int(getattr(ds, "voteCount", 0) or 0),
                "updated": str(getattr(ds, "lastUpdated", "") or ""),
                "license": license_name,
                "tags": tag_list,
                "subtitle": str(getattr(ds, "subtitle", "") or ""),
                "description": str(getattr(ds, "description", "") or "")[:500],
                "url": f"https://www.kaggle.com/datasets/{ref}",
            }
            collected.append(entry)

        # Checkpoint every 10 pages
        if page % 10 == 0:
            _save_checkpoint(checkpoint_path, collected, page + 1)
            logger.info("Checkpoint saved at page %d (%d results)", page, len(collected))

        time.sleep(sleep_between_pages)

    # Final save
    with output_path_p.open("w") as f:
        json.dump(collected, f, indent=2, default=str)
    logger.info("Saved %d datasets to %s", len(collected), output_path)

    # Remove checkpoint on success
    if checkpoint_path.exists():
        checkpoint_path.unlink()

    return collected


def _save_checkpoint(path: Path, results: list[dict], next_page: int) -> None:
    with path.open("w") as f:
        json.dump({"results": results, "next_page": next_page}, f, default=str)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Scrape Kaggle dataset catalog and save metadata to JSON."
    )
    parser.add_argument("--pages", type=int, default=500, help="Number of pages to scrape (default: 500)")
    parser.add_argument("--file-type", default="csv", help="File type filter (default: csv)")
    parser.add_argument("--tags", default="14101", help="Tag IDs to filter (default: 14101 = tabular data)")
    parser.add_argument("--min-size", type=int, default=50_000, help="Min dataset size in bytes (default: 50000)")
    parser.add_argument("--max-size", type=int, default=500_000_000, help="Max dataset size in bytes (default: 500MB)")
    parser.add_argument("--output", default="kaggle_registry.json", help="Output JSON file path")
    parser.add_argument("--sleep", type=float, default=1.0, help="Sleep between pages in seconds (default: 1.0)")
    args = parser.parse_args()

    scrape_kaggle_datasets(
        pages=args.pages,
        file_type=args.file_type,
        tags=args.tags,
        min_size=args.min_size,
        max_size=args.max_size,
        output_path=args.output,
        sleep_between_pages=args.sleep,
    )
