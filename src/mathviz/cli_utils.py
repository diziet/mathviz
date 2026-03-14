"""CLI utility commands: convert, sample, transform, and schema generation."""

import json
import logging
from pathlib import Path
from typing import Optional

import typer
from rich.console import Console

from mathviz.core.container import Container, PlacementPolicy
from mathviz.pipeline.geometry_loader import GeometryLoadError, has_mesh, load_geometry
from mathviz.pipeline.sampler import SamplerConfig

logger = logging.getLogger(__name__)

EXIT_SUCCESS = 0
EXIT_ERROR = 2


def register_util_commands(
    app: typer.Typer,
    configure_logging_fn: object,
    console: Console | None = None,
) -> None:
    """Register convert, sample, transform, and schema commands."""
    output_console = console or Console()

    @app.command()
    def convert(
        input_path: Path = typer.Argument(help="Input geometry file"),
        output_path: Path = typer.Argument(help="Output geometry file"),
        auto_sample: bool = typer.Option(
            False, "--auto-sample", help="Auto-sample mesh to point cloud if needed"
        ),
        fmt: Optional[str] = typer.Option(None, "--format", help="Output format override"),
        num_points: Optional[int] = typer.Option(
            None, "--num-points", help="Point count for auto-sample"
        ),
        verbose: bool = typer.Option(False, "--verbose", help="Enable debug logging"),
        quiet: bool = typer.Option(False, "--quiet", help="Suppress non-error output"),
    ) -> None:
        """Convert geometry between formats (STL, OBJ, PLY)."""
        configure_logging_fn(verbose, quiet)
        _run_convert(
            input_path, output_path, auto_sample, fmt, num_points, output_console, quiet
        )

    @app.command()
    def sample(
        input_path: Path = typer.Argument(help="Input mesh file (STL, OBJ, PLY)"),
        output_path: Path = typer.Argument(help="Output point cloud file (PLY, XYZ)"),
        method: str = typer.Option("uniform_surface", "--method", help="Sampling method"),
        num_points: Optional[int] = typer.Option(None, "--num-points", help="Point count"),
        density: Optional[float] = typer.Option(None, "--density", help="Points/mm²"),
        seed: int = typer.Option(42, "--seed", help="RNG seed"),
        fmt: Optional[str] = typer.Option(None, "--format", help="Output format override"),
        verbose: bool = typer.Option(False, "--verbose", help="Enable debug logging"),
        quiet: bool = typer.Option(False, "--quiet", help="Suppress non-error output"),
    ) -> None:
        """Sample a mesh into a point cloud."""
        configure_logging_fn(verbose, quiet)
        _run_sample(
            input_path, output_path, method, num_points, density, seed, fmt,
            output_console, quiet,
        )

    @app.command()
    def transform(
        input_path: Path = typer.Argument(help="Input geometry file"),
        output_path: Path = typer.Argument(help="Output geometry file"),
        width: float = typer.Option(100.0, "--width", help="Container width (mm)"),
        height: float = typer.Option(100.0, "--height", help="Container height (mm)"),
        depth: float = typer.Option(40.0, "--depth", help="Container depth (mm)"),
        fmt: Optional[str] = typer.Option(None, "--format", help="Output format override"),
        verbose: bool = typer.Option(False, "--verbose", help="Enable debug logging"),
        quiet: bool = typer.Option(False, "--quiet", help="Suppress non-error output"),
    ) -> None:
        """Fit geometry within a container's bounding box."""
        configure_logging_fn(verbose, quiet)
        _run_transform(
            input_path, output_path, width, height, depth, fmt, output_console, quiet
        )

    @app.command()
    def schema(
        output_dir: Path = typer.Argument(help="Directory to write JSON Schema files"),
        verbose: bool = typer.Option(False, "--verbose", help="Enable debug logging"),
        quiet: bool = typer.Option(False, "--quiet", help="Suppress non-error output"),
    ) -> None:
        """Generate JSON Schema files from all config models."""
        configure_logging_fn(verbose, quiet)
        _run_schema(output_dir, output_console, quiet)


def _run_convert(
    input_path: Path,
    output_path: Path,
    auto_sample: bool,
    fmt: str | None,
    num_points: int | None,
    console: Console,
    quiet: bool,
) -> None:
    """Execute the convert command."""
    from mathviz.pipeline.mesh_exporter import export_mesh
    from mathviz.pipeline.point_cloud_exporter import export_point_cloud

    try:
        obj = load_geometry(input_path)
    except GeometryLoadError as exc:
        console.print(f"[red]Error: {exc}[/red]")
        raise typer.Exit(code=EXIT_ERROR)

    output_fmt = _resolve_output_format(output_path, fmt)
    is_cloud_output = output_fmt in {"ply", "xyz", "pcd"}
    is_mesh_output = output_fmt in {"stl", "obj"}

    if is_cloud_output and not has_mesh(obj) and obj.point_cloud is not None:
        export_point_cloud(obj, output_path, fmt=output_fmt)
    elif is_cloud_output and has_mesh(obj) and auto_sample:
        from mathviz.pipeline.sampler import sample as run_sample
        sampler_cfg = SamplerConfig(num_points=num_points) if num_points else SamplerConfig()
        obj = run_sample(obj, sampler_cfg)
        export_point_cloud(obj, output_path, fmt=output_fmt)
    elif is_cloud_output and has_mesh(obj) and not auto_sample:
        if obj.point_cloud is not None:
            export_point_cloud(obj, output_path, fmt=output_fmt)
        else:
            console.print(
                "[red]Error: Input is a mesh but output requires a point cloud. "
                "Use --auto-sample to convert mesh to point cloud.[/red]"
            )
            raise typer.Exit(code=EXIT_ERROR)
    elif is_mesh_output and has_mesh(obj):
        export_mesh(obj, output_path, fmt=output_fmt)
    elif is_mesh_output and not has_mesh(obj):
        console.print(
            "[red]Error: Input has no mesh geometry. Cannot export to mesh format "
            f"'{output_fmt}' without mesh data.[/red]"
        )
        raise typer.Exit(code=EXIT_ERROR)
    elif output_fmt == "ply" and has_mesh(obj):
        export_mesh(obj, output_path, fmt=output_fmt)
    else:
        console.print(f"[red]Error: Unsupported output format '{output_fmt}'[/red]")
        raise typer.Exit(code=EXIT_ERROR)

    if not quiet:
        console.print(f"[green]Converted {input_path} → {output_path}[/green]")


def _run_sample(
    input_path: Path,
    output_path: Path,
    method: str,
    num_points: int | None,
    density: float | None,
    seed: int,
    fmt: str | None,
    console: Console,
    quiet: bool,
) -> None:
    """Execute the sample command."""
    from mathviz.pipeline.point_cloud_exporter import export_point_cloud
    from mathviz.pipeline.sampler import SamplingMethod, sample as run_sample

    try:
        obj = load_geometry(input_path)
    except GeometryLoadError as exc:
        console.print(f"[red]Error: {exc}[/red]")
        raise typer.Exit(code=EXIT_ERROR)

    if not has_mesh(obj):
        console.print("[red]Error: Input file has no mesh geometry. Sampling requires a mesh.[/red]")
        raise typer.Exit(code=EXIT_ERROR)

    try:
        sampling_method = SamplingMethod(method)
    except ValueError:
        valid = [m.value for m in SamplingMethod]
        console.print(f"[red]Error: Unknown method '{method}'. Valid: {valid}[/red]")
        raise typer.Exit(code=EXIT_ERROR)

    config = SamplerConfig(
        method=sampling_method,
        num_points=num_points,
        density=density,
        seed=seed,
        resample=True,
    )
    obj = run_sample(obj, config)
    export_point_cloud(obj, output_path, fmt=fmt)

    if not quiet:
        point_count = len(obj.point_cloud.points) if obj.point_cloud else 0
        console.print(f"[green]Sampled {point_count} points → {output_path}[/green]")


def _run_transform(
    input_path: Path,
    output_path: Path,
    width: float,
    height: float,
    depth_mm: float,
    fmt: str | None,
    console: Console,
    quiet: bool,
) -> None:
    """Execute the transform command."""
    from mathviz.core.math_object import CoordSpace
    from mathviz.pipeline.mesh_exporter import export_mesh
    from mathviz.pipeline.point_cloud_exporter import export_point_cloud
    from mathviz.pipeline.transformer import fit

    try:
        obj = load_geometry(input_path)
    except GeometryLoadError as exc:
        console.print(f"[red]Error: {exc}[/red]")
        raise typer.Exit(code=EXIT_ERROR)

    # Transformer expects ABSTRACT space; loaded files are PHYSICAL
    obj.coord_space = CoordSpace.ABSTRACT

    container = Container(width_mm=width, height_mm=height, depth_mm=depth_mm)
    policy = PlacementPolicy()
    obj = fit(obj, container, policy)

    output_fmt = _resolve_output_format(output_path, fmt)
    if output_fmt in {"stl", "obj"} and has_mesh(obj):
        export_mesh(obj, output_path, fmt=output_fmt)
    elif output_fmt in {"ply", "xyz", "pcd"} and obj.point_cloud is not None:
        export_point_cloud(obj, output_path, fmt=output_fmt)
    elif output_fmt == "ply" and has_mesh(obj):
        export_mesh(obj, output_path, fmt=output_fmt)
    else:
        # Default: export whatever geometry is available
        if has_mesh(obj):
            export_mesh(obj, output_path, fmt=output_fmt)
        elif obj.point_cloud is not None:
            export_point_cloud(obj, output_path, fmt=output_fmt)
        else:
            console.print("[red]Error: No exportable geometry after transform.[/red]")
            raise typer.Exit(code=EXIT_ERROR)

    if not quiet:
        console.print(
            f"[green]Transformed {input_path} → {output_path} "
            f"(container: {width}×{height}×{depth_mm}mm)[/green]"
        )


def _run_schema(output_dir: Path, console: Console, quiet: bool) -> None:
    """Generate JSON Schema files from all Pydantic config models."""
    from mathviz.core.container import Container, PlacementPolicy
    from mathviz.core.engraving import EngravingProfile
    from mathviz.core.generator import list_generators
    from mathviz.core.representation import RepresentationConfig
    from mathviz.pipeline.sampler import SamplerConfig

    output_dir.mkdir(parents=True, exist_ok=True)

    models = {
        "Container": Container,
        "PlacementPolicy": PlacementPolicy,
        "SamplerConfig": SamplerConfig,
        "RepresentationConfig": RepresentationConfig,
        "EngravingProfile": EngravingProfile,
    }

    written: list[str] = []
    for name, model_cls in models.items():
        schema = model_cls.model_json_schema()
        schema_path = output_dir / f"{name}.json"
        schema_path.write_text(json.dumps(schema, indent=2) + "\n", encoding="utf-8")
        written.append(name)

    # Generator parameter schemas
    generators_dir = output_dir / "generators"
    generators_dir.mkdir(parents=True, exist_ok=True)
    for meta in list_generators():
        gen_instance = meta.generator_class.create(resolved_name=meta.name)
        param_schema = gen_instance.get_param_schema()
        if param_schema:
            schema_path = generators_dir / f"{meta.name}.json"
            schema_path.write_text(
                json.dumps(param_schema, indent=2) + "\n", encoding="utf-8"
            )
            written.append(f"generators/{meta.name}")

    if not quiet:
        console.print(f"[green]Generated {len(written)} schema files in {output_dir}[/green]")
        for name in written:
            console.print(f"  {name}.json")


def _resolve_output_format(path: Path, fmt: str | None) -> str:
    """Resolve output format from explicit arg or file suffix."""
    if fmt is not None:
        return fmt.lower().lstrip(".")
    suffix = path.suffix.lower().lstrip(".")
    if not suffix:
        raise typer.Exit(code=EXIT_ERROR)
    return suffix
