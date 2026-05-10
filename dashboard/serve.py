#!/usr/bin/env python3
"""Serve the Ludwig AutoML Dashboard locally.

Usage:
    python serve.py [--port PORT] [--data-dir PATH]

The dashboard expects a `data/` directory alongside index.html. This script
symlinks (or uses) the specified data directory so the browser can fetch
data/summary.json, data/datasets.json, etc.
"""
from __future__ import annotations

import argparse
import os
import sys
import webbrowser
from http.server import HTTPServer, SimpleHTTPRequestHandler
from pathlib import Path


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Serve the Ludwig AutoML Dashboard.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8765,
        help="Port to listen on.",
    )
    parser.add_argument(
        "--data-dir",
        default=None,
        help=(
            "Path to the exported data/ directory "
            "(default: ../data relative to the dashboard directory, "
            "i.e. <repo>/data)."
        ),
    )
    parser.add_argument(
        "--no-browser",
        action="store_true",
        help="Do not auto-open the browser.",
    )
    return parser


def resolve_data_dir(data_dir_arg: str | None, dashboard_dir: Path) -> Path:
    if data_dir_arg:
        p = Path(data_dir_arg).resolve()
    else:
        # Default: one level above dashboard/ — sibling to the dashboard dir
        p = (dashboard_dir.parent / "data").resolve()
    return p


def ensure_data_symlink(data_dir: Path, dashboard_dir: Path) -> None:
    """Create/update a `data` symlink inside dashboard_dir pointing at data_dir."""
    link = dashboard_dir / "data"

    if data_dir == link:
        # data_dir is already the symlink target path, nothing to do
        return

    if link.is_symlink():
        current_target = Path(os.readlink(link))
        if current_target.resolve() == data_dir:
            return  # already correct
        print(f"  Updating symlink: data/ -> {data_dir}")
        link.unlink()
    elif link.exists():
        # It's a real directory — don't clobber it, just use it as-is
        print(f"  Found existing data/ directory at {link} — using it as-is.")
        return
    elif not data_dir.exists():
        print(
            f"\n  WARNING: data directory does not exist: {data_dir}\n"
            "  The dashboard will show an error until you run the exporter:\n\n"
            "    python -m benchmark.exporter \\\n"
            "      --results-dir ./results \\\n"
            "      --output-dir .\n",
            file=sys.stderr,
        )
        return

    link.symlink_to(data_dir)
    print(f"  Symlinked: {link} -> {data_dir}")


class QuietHandler(SimpleHTTPRequestHandler):
    """Like SimpleHTTPRequestHandler but suppresses per-request log lines."""

    def log_message(self, fmt_str: str, *args: object) -> None:  # noqa: A002
        # Only log errors (4xx / 5xx)
        if args and isinstance(args[1], str) and not args[1].startswith(("4", "5")):
            return
        super().log_message(fmt_str, *args)

    def log_error(self, fmt_str: str, *args: object) -> None:
        super().log_message(fmt_str, *args)


def main() -> None:
    parser = build_argument_parser()
    args = parser.parse_args()

    dashboard_dir = Path(__file__).parent.resolve()
    data_dir = resolve_data_dir(args.data_dir, dashboard_dir)

    print(f"\n  Ludwig AutoML Dashboard")
    print(f"  ========================")
    print(f"  Dashboard dir : {dashboard_dir}")
    print(f"  Data dir      : {data_dir}")

    ensure_data_symlink(data_dir, dashboard_dir)

    # Serve from the dashboard directory so index.html is at /
    os.chdir(dashboard_dir)

    url = f"http://localhost:{args.port}/"
    print(f"\n  Serving at    : {url}")
    print(f"  Press Ctrl+C to stop.\n")

    if not args.no_browser:
        webbrowser.open(url)

    server = HTTPServer(("", args.port), QuietHandler)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n  Server stopped.")


if __name__ == "__main__":
    main()
