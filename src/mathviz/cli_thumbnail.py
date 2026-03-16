"""CLI command for rendering generator thumbnails in a subprocess-safe context.

Provides `render-thumbnail` which generates WebP thumbnails for generators.
Called directly by users or spawned as a subprocess by the preview server
to avoid VTK/Cocoa main-thread crashes on macOS.
"""

import logging
import sys
from collections.abc import Callable
from typing import Optional

import typer
from rich.console import Console

from mathviz.core.generator import list_generators
from mathviz.preview.thumbnails import (
    DEFAULT_VIEW_MODE,
    VALID_VIEW_MODES,
    generate_thumbnail,
    get_thumbnail_path,
)

logger = logging.getLogger(__name__)


def register_render_thumbnail_command(
    app: typer.Typer,
    configure_logging_fn: Callable[[bool, bool], None],
    console: Console | None = None,
) -> None:
    """Register the render-thumbnail command on the given Typer app."""
    output_console = console or Console()

    @app.command("render-thumbnail")
    def render_thumbnail(
        name: Optional[str] = typer.Argument(None, help="Generator name"),
        view_mode: str = typer.Option(
            DEFAULT_VIEW_MODE, "--view-mode", help="View mode: points/shaded/wireframe",
        ),
        all_generators: bool = typer.Option(
            False, "--all", help="Generate thumbnails for all generators",
        ),
        verbose: bool = typer.Option(False, "--verbose", help="Enable debug logging"),
        quiet: bool = typer.Option(False, "--quiet", help="Suppress non-error output"),
    ) -> None:
        """Generate a WebP thumbnail for a generator (or all generators)."""
        configure_logging_fn(verbose, quiet)

        if view_mode not in VALID_VIEW_MODES:
            output_console.print(
                f"[red]Invalid view_mode {view_mode!r}. "
                f"Must be one of {VALID_VIEW_MODES}.[/red]"
            )
            raise typer.Exit(code=2)

        if all_generators:
            _render_all(view_mode, output_console, quiet)
        elif name is not None:
            _render_one(name, view_mode, output_console, quiet)
        else:
            output_console.print("[red]Provide a generator name or --all.[/red]")
            raise typer.Exit(code=2)

    return None


def _render_one(
    name: str, view_mode: str, console: Console, quiet: bool,
) -> None:
    """Render a single generator's thumbnail."""
    try:
        path = generate_thumbnail(name, view_mode)
    except KeyError:
        console.print(f"[red]Unknown generator: {name!r}[/red]")
        raise typer.Exit(code=2)
    except Exception as exc:
        logger.debug("Thumbnail generation failed for %s", name, exc_info=True)
        console.print(f"[red]Failed to generate thumbnail for {name}: {exc}[/red]")
        raise typer.Exit(code=1)

    if not quiet:
        console.print(f"[green]Thumbnail saved: {path}[/green]")


def _render_all(view_mode: str, console: Console, quiet: bool) -> None:
    """Render thumbnails for all registered generators."""
    generators = list_generators()
    generated = 0
    cached = 0
    failed: list[str] = []

    for meta in generators:
        cached_path = get_thumbnail_path(meta.name, view_mode)
        if cached_path.is_file():
            if not quiet:
                console.print(f"  [dim]{meta.name}: cached[/dim]")
            cached += 1
            continue
        try:
            generate_thumbnail(meta.name, view_mode)
            generated += 1
            if not quiet:
                console.print(f"  [green]{meta.name}: ok[/green]")
        except KeyError:
            failed.append(meta.name)
            console.print(f"  [red]{meta.name}: unknown generator[/red]")
        except Exception as exc:
            logger.debug("Thumbnail generation failed for %s", meta.name, exc_info=True)
            failed.append(meta.name)
            console.print(f"  [red]{meta.name}: {exc}[/red]")

    total = len(generators)
    if not quiet:
        console.print(
            f"\n[bold]Generated {generated}, cached {cached} / {total} thumbnails[/bold]"
        )
    if failed:
        console.print(f"[red]Failed: {', '.join(failed)}[/red]")
        raise typer.Exit(code=1)


def render_thumbnail_main() -> None:
    """Minimal entry point for subprocess invocation.

    Called via `python -m mathviz.cli_thumbnail <name> <view_mode>`.
    Exits 0 on success, non-zero on failure. Avoids importing the full
    CLI/Typer stack for faster subprocess startup.
    """
    if len(sys.argv) < 3:
        print("Usage: python -m mathviz.cli_thumbnail <name> <view_mode>", file=sys.stderr)
        sys.exit(2)

    generator_name = sys.argv[1]
    view_mode = sys.argv[2]

    if view_mode not in VALID_VIEW_MODES:
        print(
            f"Invalid view_mode {view_mode!r}. Must be one of {VALID_VIEW_MODES}",
            file=sys.stderr,
        )
        sys.exit(2)

    logging.basicConfig(level=logging.WARNING, format="%(name)s %(levelname)s: %(message)s")

    try:
        generate_thumbnail(generator_name, view_mode)
    except Exception as exc:
        print(f"Error: {exc}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    render_thumbnail_main()
