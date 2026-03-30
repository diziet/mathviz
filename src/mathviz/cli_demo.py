"""CLI export-demo command: build a static demo site from generator output."""

import logging
from collections.abc import Callable
from pathlib import Path

import typer
from rich.console import Console

from mathviz.demo_builder import build_demo

logger = logging.getLogger(__name__)

# Curated list of ~15 visually impressive generators for the default demo
DEFAULT_GENERATORS = ",".join([
    "lorenz",
    "rossler",
    "thomas",
    "halvorsen",
    "mandelbulb",
    "julia3d",
    "gyroid",
    "schwarz_p",
    "torus_knot",
    "figure_eight_knot",
    "hopf_fibration",
    "klein_bottle",
    "enneper_surface",
    "boy_surface",
    "shell_spiral",
])


def register_export_demo_command(
    app: typer.Typer,
    configure_logging_fn: Callable[[bool, bool], None],
    console: Console | None = None,
) -> None:
    """Register the export-demo command on the given Typer app."""
    output_console = console or Console()

    @app.command("export-demo")
    def export_demo(
        generators: str = typer.Option(
            DEFAULT_GENERATORS,
            "--generators",
            help="Comma-separated generator names, or 'all'",
        ),
        output: Path = typer.Option(
            Path("dist"),
            "--output",
            help="Output directory for the demo site",
        ),
        profile: str = typer.Option(
            "preview",
            "--profile",
            help="Sampling profile name (e.g. preview, production)",
        ),
        verbose: bool = typer.Option(False, "--verbose", help="Enable debug logging"),
        quiet: bool = typer.Option(False, "--quiet", help="Suppress non-error output"),
    ) -> None:
        """Build a self-contained static demo site from generator pipeline output."""
        configure_logging_fn(verbose, quiet)

        if not quiet:
            output_console.print(
                f"[bold]Building demo site[/bold] → {output}"
            )

        result = build_demo(generators, output, profile)

        if result.succeeded == 0:
            output_console.print("[red]No generators exported successfully[/red]")
            raise typer.Exit(code=1)

        if result.skipped and not quiet:
            output_console.print(
                f"[yellow]Skipped {len(result.skipped)} generator(s): "
                f"{', '.join(result.skipped)}[/yellow]"
            )

        if not quiet:
            output_console.print(
                f"[green]Demo built: {result.succeeded} generator(s) → {output}[/green]"
            )
