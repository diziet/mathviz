"""CLI cache management commands."""

import logging
from collections.abc import Callable

import typer
from rich.console import Console

from mathviz.preview.disk_cache import DiskCache

logger = logging.getLogger(__name__)


def register_cache_commands(
    app: typer.Typer,
    configure_logging_fn: Callable[[bool, bool], None],
    console: Console | None = None,
) -> None:
    """Register cache management commands under 'cache' subcommand."""
    output_console = console or Console()

    cache_app = typer.Typer(
        name="cache",
        help="Manage the generation cache.",
        add_completion=False,
    )

    @cache_app.command("clear")
    def cache_clear(
        verbose: bool = typer.Option(False, "--verbose", help="Enable debug logging"),
        quiet: bool = typer.Option(False, "--quiet", help="Suppress non-error output"),
    ) -> None:
        """Remove all cached generation results from disk."""
        configure_logging_fn(verbose, quiet)
        disk_cache = DiskCache()
        count = disk_cache.clear()
        if not quiet:
            output_console.print(
                f"[green]Cleared {count} cached entries[/green]"
            )

    app.add_typer(cache_app)
