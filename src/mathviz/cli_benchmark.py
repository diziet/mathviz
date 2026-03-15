"""CLI benchmark command: run generators through the pipeline and report timings."""

import logging
import os
import platform
import time
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import typer
from rich.console import Console
from rich.table import Table

from mathviz.benchmark_report import generate_html_report
from mathviz.core.container import Container, PlacementPolicy
from mathviz.core.generator import list_generators
from mathviz.pipeline.runner import run

logger = logging.getLogger(__name__)

PIPELINE_STAGES = ["generate", "represent", "transform", "validate"]


@dataclass
class BenchmarkResult:
    """Result of benchmarking a single generator."""

    generator_name: str
    stage_timings: dict[str, float] = field(default_factory=dict)
    total_time: float = 0.0
    error: str | None = None


@dataclass
class BenchmarkSuite:
    """Collection of all benchmark results with system info."""

    results: list[BenchmarkResult] = field(default_factory=list)
    cpu: str = ""
    python_version: str = ""
    date: str = ""
    worker_count: int = 1
    runs_per_generator: int = 1


def _run_single_generator(generator_name: str, num_runs: int) -> BenchmarkResult:
    """Run a single generator through the pipeline N times, return averaged timings."""
    accumulated: dict[str, list[float]] = {}
    total_times: list[float] = []

    for _ in range(num_runs):
        try:
            start = time.monotonic()
            result = run(
                generator=generator_name,
                params=None,
                seed=42,
                container=Container(),
                placement=PlacementPolicy(),
            )
            elapsed_total = time.monotonic() - start

            for stage, duration in result.timings.items():
                accumulated.setdefault(stage, []).append(duration)
            total_times.append(elapsed_total)
        except Exception as exc:
            return BenchmarkResult(
                generator_name=generator_name,
                error=f"{type(exc).__name__}: {exc}",
            )

    mean_timings = {stage: sum(vals) / len(vals) for stage, vals in accumulated.items()}
    mean_total = sum(total_times) / len(total_times)

    return BenchmarkResult(
        generator_name=generator_name,
        stage_timings=mean_timings,
        total_time=mean_total,
    )


def _filter_generators(
    requested: list[str] | None,
) -> list[str]:
    """Return generator names to benchmark, filtering out data-driven ones by default."""
    all_generators = list_generators()

    if requested:
        return list(requested)

    return [
        g.name
        for g in all_generators
        if g.category != "data_driven"
    ]


def _collect_system_info(worker_count: int, runs: int) -> BenchmarkSuite:
    """Create a BenchmarkSuite with system info populated."""
    return BenchmarkSuite(
        cpu=platform.processor() or platform.machine(),
        python_version=platform.python_version(),
        date=datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC"),
        worker_count=worker_count,
        runs_per_generator=runs,
    )


def _print_text_summary(suite: BenchmarkSuite, output_console: Console) -> None:
    """Print a compact text summary table to stdout."""
    table = Table(title="Benchmark Results")
    table.add_column("Generator")
    for stage in PIPELINE_STAGES:
        table.add_column(stage.capitalize(), justify="right")
    table.add_column("Total", justify="right")
    table.add_column("Status")

    for result in sorted(suite.results, key=lambda r: r.total_time):
        if result.error:
            table.add_row(
                result.generator_name,
                *["—"] * len(PIPELINE_STAGES),
                "—",
                "[red]ERROR[/red]",
            )
            continue

        cells = []
        for stage in PIPELINE_STAGES:
            val = result.stage_timings.get(stage, 0.0)
            cells.append(_format_time_rich(val))
        cells.append(_format_time_rich(result.total_time))
        table.add_row(result.generator_name, *cells, "[green]OK[/green]")

    output_console.print(table)
    output_console.print(
        f"\n{len(suite.results)} generators, "
        f"{suite.worker_count} workers, "
        f"{suite.runs_per_generator} run(s) each"
    )


def _format_time_rich(seconds: float) -> str:
    """Format a time value with color coding for rich output."""
    ms = seconds * 1000
    formatted = f"{seconds:.3f}s"
    if ms < 100:
        return f"[green]{formatted}[/green]"
    if ms < 1000:
        return f"[yellow]{formatted}[/yellow]"
    return f"[red]{formatted}[/red]"


def register_benchmark_command(
    app: typer.Typer,
    configure_logging_fn: Any,
    console: Console | None = None,
) -> None:
    """Register the benchmark command on the given Typer app."""
    output_console = console or Console()

    @app.command()
    def benchmark(
        generators: str | None = typer.Option(
            None,
            "--generators",
            help="Comma-separated list of generators to benchmark",
        ),
        workers: int = typer.Option(
            os.cpu_count() or 4,
            "--workers",
            help="Number of parallel workers",
        ),
        output: Path = typer.Option(
            Path("benchmark_report.html"),
            "--output",
            help="Output HTML report file path",
        ),
        runs: int = typer.Option(
            3,
            "--runs",
            help="Number of runs per generator for averaging",
        ),
        verbose: bool = typer.Option(False, "--verbose", help="Enable debug logging"),
        quiet: bool = typer.Option(False, "--quiet", help="Suppress non-error output"),
    ) -> None:
        """Run pipeline benchmarks across generators and produce an HTML report."""
        configure_logging_fn(verbose, quiet)

        gen_names = generators.split(",") if generators else None
        selected = _filter_generators(gen_names)

        if not selected:
            output_console.print("[red]No generators selected for benchmarking.[/red]")
            raise typer.Exit(code=2)

        if not quiet:
            output_console.print(
                f"Benchmarking {len(selected)} generators "
                f"({workers} workers, {runs} run(s) each)..."
            )

        suite = _collect_system_info(workers, runs)

        with ProcessPoolExecutor(max_workers=workers) as executor:
            futures = {
                executor.submit(_run_single_generator, name, runs): name
                for name in selected
            }
            for future in futures:
                bench_result = future.result()
                suite.results.append(bench_result)

        # Generate HTML report
        html = generate_html_report(suite)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(html, encoding="utf-8")

        if not quiet:
            _print_text_summary(suite, output_console)
            output_console.print(f"\nHTML report written to {output}")
