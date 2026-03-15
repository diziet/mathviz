"""CLI render commands: high-resolution 3D and 2D projection rendering."""

import logging
from pathlib import Path
from typing import Any, Optional

import typer
from rich.console import Console

from mathviz.preview.renderer import (
    PYVISTA_INSTALL_MSG,
    VALID_VIEW_NAMES,
    RenderConfig,
    RenderStyle,
    render_2d_projection,
    render_all_views,
    render_to_png,
    resolve_view_name,
)

logger = logging.getLogger(__name__)


def register_render_commands(
    app: typer.Typer,
    parse_params_fn: Any,
    configure_logging_fn: Any,
    run_pipeline_fn: Any,
    console: Console | None = None,
) -> None:
    """Register render and render-2d commands on the given Typer app."""
    output_console = console or Console()

    @app.command()
    def render(
        generator_name: str = typer.Argument(help="Generator name to render"),
        output: Path = typer.Option(..., "--output", "-o", help="Output PNG file path"),
        view: str = typer.Option("front-right-top", "--view", help="Camera view name or 'all'"),
        param: Optional[list[str]] = typer.Option(None, "--param", help="key=value parameter"),
        seed: Optional[int] = typer.Option(None, "--seed", help="Random seed"),
        width: int = typer.Option(1920, "--width", help="Image width in pixels"),
        height: int = typer.Option(1080, "--height", help="Image height in pixels"),
        style: str = typer.Option("points", "--style", help="Render style: shaded/wireframe/points"),
        point_size: float = typer.Option(3.0, "--point-size", help="Point size for points style"),
        verbose: bool = typer.Option(False, "--verbose", help="Enable debug logging"),
        quiet: bool = typer.Option(False, "--quiet", help="Suppress non-error output"),
    ) -> None:
        """Render a generator to a high-resolution PNG image."""
        configure_logging_fn(verbose, quiet)
        _run_render(
            generator_name=generator_name,
            output=output,
            param_list=param or [],
            seed=seed,
            width=width,
            height=height,
            style=style,
            point_size=point_size,
            view=view,
            use_2d=False,
            parse_params_fn=parse_params_fn,
            run_pipeline_fn=run_pipeline_fn,
            console=output_console,
            quiet=quiet,
        )

    @app.command("render-2d")
    def render_2d(
        generator_name: str = typer.Argument(help="Generator name to render"),
        output: Path = typer.Option(..., "--output", "-o", help="Output PNG file path"),
        view: str = typer.Option("top", "--view", help="Camera view name or 'all'"),
        param: Optional[list[str]] = typer.Option(None, "--param", help="key=value parameter"),
        seed: Optional[int] = typer.Option(None, "--seed", help="Random seed"),
        width: int = typer.Option(1920, "--width", help="Image width in pixels"),
        height: int = typer.Option(1080, "--height", help="Image height in pixels"),
        style: str = typer.Option("points", "--style", help="Render style: shaded/wireframe/points"),
        point_size: float = typer.Option(3.0, "--point-size", help="Point size for points style"),
        verbose: bool = typer.Option(False, "--verbose", help="Enable debug logging"),
        quiet: bool = typer.Option(False, "--quiet", help="Suppress non-error output"),
    ) -> None:
        """Render a 2D projection of a generator to PNG."""
        configure_logging_fn(verbose, quiet)
        _run_render(
            generator_name=generator_name,
            output=output,
            param_list=param or [],
            seed=seed,
            width=width,
            height=height,
            style=style,
            point_size=point_size,
            view=view,
            use_2d=True,
            parse_params_fn=parse_params_fn,
            run_pipeline_fn=run_pipeline_fn,
            console=output_console,
            quiet=quiet,
        )


def _run_render(
    generator_name: str,
    output: Path,
    param_list: list[str],
    seed: int | None,
    width: int,
    height: int,
    style: RenderStyle,
    point_size: float,
    view: str,
    use_2d: bool,
    parse_params_fn: Any,
    run_pipeline_fn: Any,
    console: Console,
    quiet: bool,
) -> None:
    """Shared render logic for render and render-2d commands."""
    # Validate view name early
    if view != "all":
        try:
            resolve_view_name(view)
        except ValueError as exc:
            console.print(f"[red]{exc}[/red]")
            raise typer.Exit(code=2)

    params = parse_params_fn(param_list)
    result = run_pipeline_fn(
        generator_name=generator_name,
        params=params,
        seed=seed,
        json_output=False,
    )

    try:
        config = RenderConfig(width=width, height=height, style=style, point_size=point_size)
    except ValueError as exc:
        console.print(f"[red]{exc}[/red]")
        raise typer.Exit(code=2)

    obj = result.math_object

    try:
        if view == "all":
            paths = render_all_views(obj, output, config=config, use_2d=use_2d)
        elif use_2d:
            paths = [render_2d_projection(obj, output, view=view, config=config)]
        else:
            paths = [render_to_png(obj, output, view=view, config=config)]
    except ImportError:
        console.print(f"[red]{PYVISTA_INSTALL_MSG}[/red]")
        raise typer.Exit(code=2)

    if not quiet:
        for p in paths:
            console.print(f"[green]Rendered to {p}[/green]")
