"""CLI render commands: high-resolution 3D and 2D projection rendering."""

import logging
from pathlib import Path
from typing import Any, Optional

import typer
from rich.console import Console

from mathviz.preview.renderer import (
    PYVISTA_INSTALL_MSG,
    VALID_RENDER_STYLES,
    ProjectionView,
    RenderConfig,
    render_2d_projection,
    render_to_png,
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
        if style not in VALID_RENDER_STYLES:
            output_console.print(
                f"[red]Invalid style: {style!r}. Must be one of {VALID_RENDER_STYLES}[/red]"
            )
            raise typer.Exit(code=2)
        _run_render(
            generator_name=generator_name,
            output=output,
            param_list=param or [],
            seed=seed,
            width=width,
            height=height,
            style=style,
            point_size=point_size,
            view=None,
            parse_params_fn=parse_params_fn,
            run_pipeline_fn=run_pipeline_fn,
            console=output_console,
            quiet=quiet,
        )

    @app.command("render-2d")
    def render_2d(
        generator_name: str = typer.Argument(help="Generator name to render"),
        output: Path = typer.Option(..., "--output", "-o", help="Output PNG file path"),
        view: str = typer.Option("top", "--view", help="Projection view: top/front/side/angle"),
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
        valid_views = ("top", "front", "side", "angle")
        if view not in valid_views:
            output_console.print(f"[red]Invalid view: {view!r}. Must be one of {valid_views}[/red]")
            raise typer.Exit(code=2)
        if style not in VALID_RENDER_STYLES:
            output_console.print(
                f"[red]Invalid style: {style!r}. Must be one of {VALID_RENDER_STYLES}[/red]"
            )
            raise typer.Exit(code=2)

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
    style: str,
    point_size: float,
    view: ProjectionView | None,
    parse_params_fn: Any,
    run_pipeline_fn: Any,
    console: Console,
    quiet: bool,
) -> None:
    """Shared render logic for render and render-2d commands."""
    params = parse_params_fn(param_list)
    result = run_pipeline_fn(
        generator_name=generator_name,
        params=params,
        seed=seed,
        json_output=False,
    )

    config = RenderConfig(width=width, height=height, style=style, point_size=point_size)
    obj = result.math_object

    try:
        if view is not None:
            rendered_path = render_2d_projection(obj, output, view=view, config=config)
        else:
            rendered_path = render_to_png(obj, output, config=config)
    except ImportError:
        console.print(f"[red]{PYVISTA_INSTALL_MSG}[/red]")
        raise typer.Exit(code=2)

    if not quiet:
        console.print(f"[green]Rendered to {rendered_path}[/green]")
