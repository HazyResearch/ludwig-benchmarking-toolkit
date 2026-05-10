"""Filters scraped Kaggle registry to ML-ready tabular datasets.

Usage:
    python filter_kaggle.py --input kaggle_registry.json --output ml_ready.json
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

ALLOWED_LICENSES = {"cc0-1.0", "cc-by-4.0", "odc-odbl", "cc-by-sa-4.0"}

# ---------------------------------------------------------------------------
# Core filter
# ---------------------------------------------------------------------------

def filter_ml_ready(
    registry: list[dict],
    min_votes: int = 5,
    min_downloads: int = 100,
    allowed_licenses: set[str] = ALLOWED_LICENSES,
    min_size: int = 50_000,
    max_size: int = 500_000_000,
) -> list[dict]:
    """Filters datasets to ML-ready candidates.

    Filters applied:
    - min_votes >= 5 (community validation)
    - min_downloads >= 100 (actually used)
    - license in ALLOWED_LICENSES
    - size >= 50KB and <= 500MB
    Returns filtered list sorted by votes descending.
    """
    filtered: list[dict] = []

    for entry in registry:
        votes = int(entry.get("votes", 0) or 0)
        downloads = int(entry.get("downloads", 0) or 0)
        size = int(entry.get("size", 0) or 0)
        license_name = _normalize_license(entry.get("license", "") or "")

        # Community validation
        if votes < min_votes:
            continue

        # Usage filter
        if downloads < min_downloads:
            continue

        # License filter
        if license_name not in allowed_licenses:
            continue

        # Size filter
        if size < min_size or size > max_size:
            continue

        filtered.append(entry)

    # Sort by votes descending (most popular first)
    filtered.sort(key=lambda d: int(d.get("votes", 0) or 0), reverse=True)

    return filtered


def _normalize_license(raw: str) -> str:
    """Normalize license string to a canonical form for comparison."""
    s = raw.lower().strip()
    # Strip common suffixes/variations
    s = s.replace(" ", "-")
    # Map some known aliases
    aliases = {
        "cc0": "cc0-1.0",
        "cc-zero": "cc0-1.0",
        "cc-by": "cc-by-4.0",
        "cc-by-sa": "cc-by-sa-4.0",
        "odbl": "odc-odbl",
        "odc-by": "cc-by-4.0",
    }
    return aliases.get(s, s)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Filter scraped Kaggle registry to ML-ready tabular datasets."
    )
    parser.add_argument("--input", required=True, help="Path to kaggle_registry.json from scrape_kaggle.py")
    parser.add_argument("--output", required=True, help="Output path for filtered ml_ready.json")
    parser.add_argument("--min-votes", type=int, default=5, help="Minimum vote count (default: 5)")
    parser.add_argument("--min-downloads", type=int, default=100, help="Minimum download count (default: 100)")
    parser.add_argument(
        "--allowed-licenses",
        nargs="+",
        default=sorted(ALLOWED_LICENSES),
        help=f"Allowed license identifiers (default: {sorted(ALLOWED_LICENSES)})",
    )
    parser.add_argument("--min-size", type=int, default=50_000, help="Minimum size in bytes (default: 50000)")
    parser.add_argument("--max-size", type=int, default=500_000_000, help="Maximum size in bytes (default: 500MB)")
    args = parser.parse_args()

    input_path = Path(args.input)
    if not input_path.exists():
        raise SystemExit(f"Input file not found: {input_path}")

    with input_path.open() as f:
        registry = json.load(f)

    allowed = set(args.allowed_licenses)
    results = filter_ml_ready(
        registry,
        min_votes=args.min_votes,
        min_downloads=args.min_downloads,
        allowed_licenses=allowed,
        min_size=args.min_size,
        max_size=args.max_size,
    )

    output_path = Path(args.output)
    with output_path.open("w") as f:
        json.dump(results, f, indent=2, default=str)

    print(f"Filtered {len(registry)} → {len(results)} ML-ready datasets → {output_path}")
