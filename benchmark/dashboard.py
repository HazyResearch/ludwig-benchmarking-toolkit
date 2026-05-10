"""Live terminal dashboard for benchmark progress."""
from __future__ import annotations

import time
from datetime import datetime, timezone
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from benchmark.db import BenchmarkDB
    from benchmark.scheduler import BenchmarkScheduler

# ---------------------------------------------------------------------------
# Rich availability
# ---------------------------------------------------------------------------

try:
    from rich import box
    from rich.console import Console
    from rich.live import Live
    from rich.panel import Panel
    from rich.table import Table
    from rich.text import Text
    _RICH = True
except ImportError:
    _RICH = False


# ---------------------------------------------------------------------------
# One-shot progress print
# ---------------------------------------------------------------------------

def print_progress(db: "BenchmarkDB", scheduler: "BenchmarkScheduler") -> None:
    """Print a one-shot progress summary using Rich if available, else plain text."""
    sched_prog = scheduler.progress()
    db_prog = db.progress_summary()

    if _RICH:
        _rich_print_progress(sched_prog, db_prog, db)
    else:
        _plain_print_progress(sched_prog, db_prog)


def _plain_print_progress(sched_prog: dict, db_prog: dict) -> None:
    print("\n=== Benchmark Progress ===")
    print(f"  Scheduler  — total={sched_prog['total']}  "
          f"queued={sched_prog['queued']}  "
          f"running={sched_prog['running']}  "
          f"done={sched_prog['done']}  "
          f"failed={sched_prog['failed']}")
    print(f"  Results DB — total={db_prog['total']}  "
          f"queued={db_prog['queued']}  "
          f"running={db_prog['running']}  "
          f"done={db_prog['done']}  "
          f"failed={db_prog['failed']}")
    total = sched_prog["total"]
    done = sched_prog["done"]
    pct = 100 * done / total if total else 0.0
    print(f"  Progress   — {done}/{total} ({pct:.1f}%)")
    print(f"  Timestamp  — {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S')} UTC")
    print()


def _rich_print_progress(
    sched_prog: dict,
    db_prog: dict,
    db: "BenchmarkDB",
    console: "Console | None" = None,
) -> None:
    console = console or Console()
    total = sched_prog["total"]
    done = sched_prog["done"]
    pct = 100 * done / total if total else 0.0

    # Progress table
    prog_table = Table(title="Benchmark Progress", box=box.ROUNDED, show_header=True)
    prog_table.add_column("Source", style="bold cyan")
    prog_table.add_column("Total", justify="right")
    prog_table.add_column("Queued", justify="right", style="yellow")
    prog_table.add_column("Running", justify="right", style="blue")
    prog_table.add_column("Done", justify="right", style="green")
    prog_table.add_column("Failed", justify="right", style="red")

    prog_table.add_row(
        "Scheduler",
        str(sched_prog["total"]),
        str(sched_prog["queued"]),
        str(sched_prog["running"]),
        str(sched_prog["done"]),
        str(sched_prog["failed"]),
    )
    prog_table.add_row(
        "Results DB",
        str(db_prog["total"]),
        str(db_prog["queued"]),
        str(db_prog["running"]),
        str(db_prog["done"]),
        str(db_prog["failed"]),
    )

    console.print(prog_table)
    console.print(
        f"  [bold]Overall:[/bold] {done}/{total} ({pct:.1f}%)  "
        f"| [dim]{datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S')} UTC[/dim]"
    )

    # Best-per-dataset table (top 10)
    try:
        best_df = db.best_per_dataset()
        if not best_df.empty:
            best_table = Table(title="Best Result Per Dataset (top 10)", box=box.SIMPLE, show_header=True)
            best_table.add_column("Dataset", style="cyan")
            best_table.add_column("Combiner", style="magenta")
            best_table.add_column("Metric", style="white")
            best_table.add_column("Value", justify="right", style="green")
            best_table.add_column("Wall (s)", justify="right")

            for _, row in best_df.head(10).iterrows():
                best_table.add_row(
                    str(row.get("dataset_name", "")),
                    str(row.get("combiner", "")),
                    str(row.get("primary_metric", "")),
                    f"{row.get('primary_metric_value', 0.0):.4f}",
                    f"{row.get('wall_seconds', 0.0):.0f}",
                )
            console.print(best_table)
    except Exception:
        pass

    # Combiner win rates
    try:
        win_df = db.combiner_win_rates()
        if not win_df.empty:
            win_table = Table(title="Combiner Win Rates", box=box.SIMPLE)
            win_table.add_column("Combiner", style="magenta")
            win_table.add_column("Wins", justify="right")
            win_table.add_column("Win Rate (%)", justify="right", style="green")
            for _, row in win_df.iterrows():
                win_table.add_row(
                    str(row["combiner"]),
                    str(row["wins"]),
                    f"{row['win_rate']:.1f}",
                )
            console.print(win_table)
    except Exception:
        pass


# ---------------------------------------------------------------------------
# Live dashboard (blocks)
# ---------------------------------------------------------------------------

def run_live_dashboard(
    db: "BenchmarkDB",
    scheduler: "BenchmarkScheduler",
    refresh_seconds: int = 30,
) -> None:
    """Run a live-updating terminal dashboard (blocks until Ctrl-C)."""
    if _RICH:
        _rich_live_dashboard(db, scheduler, refresh_seconds)
    else:
        _plain_live_dashboard(db, scheduler, refresh_seconds)


def _plain_live_dashboard(
    db: "BenchmarkDB",
    scheduler: "BenchmarkScheduler",
    refresh_seconds: int,
) -> None:
    print(f"[dashboard] Refreshing every {refresh_seconds}s. Press Ctrl-C to stop.")
    try:
        while True:
            print_progress(db, scheduler)
            time.sleep(refresh_seconds)
    except KeyboardInterrupt:
        print("\n[dashboard] Stopped.")


def _rich_live_dashboard(
    db: "BenchmarkDB",
    scheduler: "BenchmarkScheduler",
    refresh_seconds: int,
) -> None:
    console = Console()
    console.print(
        f"[bold green]Live Dashboard[/bold green] — refreshing every {refresh_seconds}s. "
        "Press [bold]Ctrl-C[/bold] to stop."
    )

    def _make_renderable() -> Panel:
        from io import StringIO

        from rich.console import Console as _C

        buf = StringIO()
        sub = _C(file=buf, highlight=False)
        sched_prog = scheduler.progress()
        db_prog = db.progress_summary()
        _rich_print_progress(sched_prog, db_prog, db, console=sub)
        total = sched_prog["total"]
        done = sched_prog["done"]
        pct = 100 * done / total if total else 0.0
        title = (
            f"Ludwig Mega-AutoML Benchmark — {done}/{total} ({pct:.1f}%) — "
            f"{datetime.now(timezone.utc).strftime('%H:%M:%S')}"
        )
        return Panel(Text(buf.getvalue()), title=title, border_style="bright_blue")

    # Use Rich's Live for auto-refresh
    try:
        with Live(
            _make_renderable(),
            console=console,
            refresh_per_second=1,
            screen=False,
        ) as live:
            while True:
                time.sleep(refresh_seconds)
                live.update(_make_renderable())
    except KeyboardInterrupt:
        console.print("\n[bold red]Dashboard stopped.[/bold red]")
