"""CLI render-all command: batch render every generator across multiple views."""

import logging
import os
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import typer
from rich.console import Console

from mathviz.core.container import Container, PlacementPolicy
from mathviz.core.generator import get_generator_meta, list_generators
from mathviz.pipeline.runner import run
from mathviz.preview.renderer import RenderConfig, render_to_png, resolve_view_name

logger = logging.getLogger(__name__)

DEFAULT_VIEWS = ("top", "front", "side", "angle")
DATA_DRIVEN_CATEGORY = "data_driven"


@dataclass
class RenderJob:
    """A single generator+view render task."""

    generator_name: str
    view: str


@dataclass
class RenderResult:
    """Result of a single render job."""

    generator_name: str
    view: str
    output_path: str = ""
    error: str | None = None
    elapsed: float = 0.0


@dataclass
class BatchSummary:
    """Summary of the full batch render run."""

    total: int = 0
    successes: int = 0
    failures: int = 0
    total_time: float = 0.0
    failed_jobs: list[RenderResult] = field(default_factory=list)


def _render_single_job(
    generator_name: str,
    view: str,
    output_dir: str,
    width: int,
    height: int,
    style: str,
) -> RenderResult:
    """Render a single generator+view combination to PNG."""
    start = time.monotonic()
    try:
        result = run(
            generator=generator_name,
            params=None,
            seed=42,
            container=Container(),
            placement=PlacementPolicy(),
        )
        obj = result.math_object

        gen_dir = Path(output_dir) / generator_name
        gen_dir.mkdir(parents=True, exist_ok=True)
        output_path = gen_dir / f"{generator_name}_{view}.png"

        config = RenderConfig(width=width, height=height, style=style)
        render_to_png(obj, output_path, config=config, view=view)

        elapsed = time.monotonic() - start
        return RenderResult(
            generator_name=generator_name,
            view=view,
            output_path=str(output_path),
            elapsed=elapsed,
        )
    except Exception as exc:
        elapsed = time.monotonic() - start
        return RenderResult(
            generator_name=generator_name,
            view=view,
            error=f"{type(exc).__name__}: {exc}",
            elapsed=elapsed,
        )


def _resolve_to_canonical(name: str) -> str | None:
    """Resolve a generator name or alias to its canonical name, or None if unknown."""
    try:
        meta = get_generator_meta(name)
        return meta.name
    except KeyError:
        return None


def _filter_generators(requested: list[str] | None) -> list[str]:
    """Return canonical generator names to render, excluding data-driven by default."""
    if requested:
        resolved: list[str] = []
        for name in requested:
            canonical = _resolve_to_canonical(name)
            if canonical is None:
                logger.warning("Unknown generator name: %s", name)
            else:
                resolved.append(canonical)
        return resolved

    return [
        g.name for g in list_generators() if g.category != DATA_DRIVEN_CATEGORY
    ]


def _validate_views(views: list[str]) -> list[str]:
    """Validate and resolve view names, raising on invalid ones."""
    return [resolve_view_name(view) for view in views]


def _build_jobs(generators: list[str], views: list[str]) -> list[RenderJob]:
    """Build list of render jobs from generators x views."""
    return [
        RenderJob(generator_name=gen, view=view)
        for gen in generators
        for view in views
    ]


def _print_summary(summary: BatchSummary, console: Console) -> None:
    """Print final batch render summary."""
    console.print(f"\n[bold]Batch render complete[/bold]")
    console.print(f"  Total renders: {summary.total}")
    console.print(f"  Successes:     {summary.successes}")
    console.print(f"  Failures:      {summary.failures}")
    console.print(f"  Total time:    {summary.total_time:.1f}s")

    if summary.failed_jobs:
        console.print("\n[red]Failed renders:[/red]")
        for job in summary.failed_jobs:
            console.print(f"  {job.generator_name} {job.view}: {job.error}")


def run_batch_render(
    generators: list[str],
    views: list[str],
    output_dir: str,
    workers: int,
    width: int,
    height: int,
    style: str,
    console: Console,
    quiet: bool = False,
) -> BatchSummary:
    """Execute batch rendering with progress tracking."""
    jobs = _build_jobs(generators, views)
    summary = BatchSummary(total=len(jobs))
    start_time = time.monotonic()

    with ProcessPoolExecutor(max_workers=workers) as executor:
        futures = {
            executor.submit(
                _render_single_job,
                job.generator_name,
                job.view,
                output_dir,
                width,
                height,
                style,
            ): job
            for job in jobs
        }

        completed_count = 0
        for future in as_completed(futures):
            completed_count += 1
            job = futures[future]

            try:
                result = future.result()
            except Exception as exc:
                result = RenderResult(
                    generator_name=job.generator_name,
                    view=job.view,
                    error=f"{type(exc).__name__}: {exc}",
                )

            if result.error:
                summary.failures += 1
                summary.failed_jobs.append(result)
            else:
                summary.successes += 1

            if not quiet:
                status = "[green]OK[/green]" if not result.error else "[red]FAIL[/red]"
                console.print(
                    f"[{completed_count}/{summary.total}] "
                    f"Rendering {job.generator_name} {job.view}... {status}"
                )

    summary.total_time = time.monotonic() - start_time
    return summary


def register_render_all_command(
    app: typer.Typer,
    configure_logging_fn: Any,
    console: Console | None = None,
) -> None:
    """Register the render-all command on the given Typer app."""
    output_console = console or Console()

    @app.command("render-all")
    def render_all(
        output_dir: Path = typer.Option(
            Path("renders"),
            "--output-dir",
            help="Base output directory",
        ),
        views: str = typer.Option(
            ",".join(DEFAULT_VIEWS),
            "--views",
            help="Comma-separated views to render",
        ),
        workers: int = typer.Option(
            os.cpu_count() or 4,
            "--workers",
            min=1,
            help="Number of parallel workers",
        ),
        generators: str | None = typer.Option(
            None,
            "--generators",
            help="Comma-separated list of generators (default: all non-data-driven)",
        ),
        style: str = typer.Option(
            "points",
            "--style",
            help="Render style: shaded, wireframe, points",
        ),
        width: int = typer.Option(1920, "--width", help="Image width in pixels"),
        height: int = typer.Option(1080, "--height", help="Image height in pixels"),
        verbose: bool = typer.Option(False, "--verbose", help="Enable debug logging"),
        quiet: bool = typer.Option(False, "--quiet", help="Suppress non-error output"),
    ) -> None:
        """Render every generator across multiple views in parallel."""
        configure_logging_fn(verbose, quiet)

        # Parse and validate views
        try:
            view_list = _validate_views([v.strip() for v in views.split(",")])
        except ValueError as exc:
            output_console.print(f"[red]{exc}[/red]")
            raise typer.Exit(code=2)

        # Validate style
        try:
            RenderConfig(style=style)
        except ValueError as exc:
            output_console.print(f"[red]{exc}[/red]")
            raise typer.Exit(code=2)

        # Resolve generators
        gen_names = generators.split(",") if generators else None
        selected = _filter_generators(gen_names)

        if not selected:
            output_console.print("[red]No generators selected for rendering.[/red]")
            raise typer.Exit(code=2)

        if not quiet:
            total_jobs = len(selected) * len(view_list)
            output_console.print(
                f"Rendering {len(selected)} generators x "
                f"{len(view_list)} views = {total_jobs} images "
                f"({workers} workers)..."
            )

        output_dir.mkdir(parents=True, exist_ok=True)

        summary = run_batch_render(
            generators=selected,
            views=view_list,
            output_dir=str(output_dir),
            workers=workers,
            width=width,
            height=height,
            style=style,
            console=output_console,
            quiet=quiet,
        )

        if not quiet:
            _print_summary(summary, output_console)

        if summary.failures > 0:
            raise typer.Exit(code=1)
