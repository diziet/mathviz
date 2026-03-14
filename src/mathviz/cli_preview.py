"""CLI preview command: start the preview server for a generator or file."""

import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional
from urllib.parse import urlencode

import typer
from rich.console import Console


@dataclass
class PreviewConfig:
    """Configuration for the preview server."""

    target: str
    port: int
    no_open: bool
    quiet: bool
    query_params: dict[str, Any]


def register_preview_command(
    app: typer.Typer,
    parse_params_fn: Any,
    configure_logging_fn: Any,
    console: Console | None = None,
) -> None:
    """Register the preview command on the given Typer app."""
    output_console = console or Console()

    @app.command()
    def preview(
        target: str = typer.Argument(help="Generator name or file path to preview"),
        param: Optional[list[str]] = typer.Option(None, "--param", help="key=value parameter"),
        seed: int = typer.Option(42, "--seed", help="Random seed"),
        port: int = typer.Option(8000, "--port", help="Server port"),
        no_open: bool = typer.Option(False, "--no-open", help="Don't open browser"),
        verbose: bool = typer.Option(False, "--verbose", help="Enable debug logging"),
        quiet: bool = typer.Option(False, "--quiet", help="Suppress non-error output"),
    ) -> None:
        """Start the preview server for a generator or file."""
        configure_logging_fn(verbose, quiet)
        query_params = _build_preview_query(target, param or [], seed, parse_params_fn)
        config = PreviewConfig(
            target=target,
            port=port,
            no_open=no_open,
            quiet=quiet,
            query_params=query_params,
        )
        _run_preview_server(config, output_console)


def _build_preview_query(
    target: str,
    param_list: list[str],
    seed: int,
    parse_params_fn: Any,
) -> dict[str, Any]:
    """Build query parameters dict for the preview URL."""
    target_path = Path(target)
    if target_path.is_file():
        return {"file": str(target_path.resolve())}
    params = parse_params_fn(param_list)
    query: dict[str, Any] = {"generator": target, "seed": seed}
    query.update(params)
    return query


def _run_preview_server(config: PreviewConfig, console: Console) -> None:
    """Start uvicorn and optionally open browser after server is ready."""
    import uvicorn

    from mathviz.preview.server import set_served_file

    target_path = Path(config.target)
    if target_path.is_file():
        set_served_file(str(target_path.resolve()))

    query = urlencode(config.query_params)
    url = f"http://127.0.0.1:{config.port}/?{query}"

    if not config.quiet:
        console.print(f"[bold green]Preview:[/bold green] {url}")
        console.print("Press Ctrl+C to stop the server.")

    if not config.no_open:
        import webbrowser

        timer = threading.Timer(1.0, webbrowser.open, args=[url])
        timer.daemon = True
        timer.start()

    uvicorn.run(
        "mathviz.preview.server:app",
        host="127.0.0.1",
        port=config.port,
        log_level="warning" if config.quiet else "info",
    )
