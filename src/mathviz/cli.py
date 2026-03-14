"""MathViz CLI: Typer-based command-line interface for the pipeline."""

import json
import logging
from pathlib import Path
from typing import Any, Optional

import typer
from rich.console import Console
from rich.table import Table

from mathviz.core.container import Container, PlacementPolicy
from mathviz.core.generator import (
    GeneratorBase,
    get_generator_meta,
    list_generators,
)
from mathviz.pipeline.runner import ExportConfig, PipelineResult, run

logger = logging.getLogger(__name__)

app = typer.Typer(
    name="mathviz",
    help="Pipeline for generating 3D mathematical forms for laser engraving.",
    add_completion=False,
)

console = Console()

# Exit codes
EXIT_SUCCESS = 0
EXIT_VALIDATION_WARNING = 1
EXIT_ERROR = 2


def _parse_params(param_list: list[str]) -> dict[str, Any]:
    """Parse key=value param strings into a dict, coercing numeric values."""
    params: dict[str, Any] = {}
    for item in param_list:
        if "=" not in item:
            console.print(f"[red]Invalid param format: {item!r} (expected key=value)[/red]")
            raise typer.Exit(code=EXIT_ERROR)
        key, value = item.split("=", 1)
        params[key] = _coerce_value(value)
    return params


def _coerce_value(value: str) -> Any:
    """Coerce a string value to int, float, or leave as string."""
    try:
        return int(value)
    except ValueError:
        pass
    try:
        return float(value)
    except ValueError:
        pass
    return value


def _configure_logging(verbose: bool, quiet: bool) -> None:
    """Configure logging level based on CLI flags."""
    if quiet:
        level = logging.WARNING
    elif verbose:
        level = logging.DEBUG
    else:
        level = logging.INFO
    logging.basicConfig(level=level, format="%(name)s %(levelname)s: %(message)s")


def _exit_code_for_result(result: PipelineResult) -> int:
    """Determine exit code from pipeline result validation."""
    if result.validation.errors:
        return EXIT_ERROR
    if result.validation.warnings:
        return EXIT_VALIDATION_WARNING
    return EXIT_SUCCESS


def _result_to_dict(result: PipelineResult, generator_name: str) -> dict[str, Any]:
    """Convert a PipelineResult to a JSON-serializable dict."""
    obj = result.math_object
    validation_checks = [
        {
            "name": c.name,
            "passed": c.passed,
            "severity": c.severity.value,
            "message": c.message,
        }
        for c in result.validation.checks
    ]
    return {
        "generator": generator_name,
        "seed": obj.seed,
        "parameters": obj.parameters,
        "coord_space": obj.coord_space.value,
        "timings": result.timings,
        "validation": {
            "passed": result.validation.passed,
            "checks": validation_checks,
        },
        "export_path": str(result.export_path) if result.export_path else None,
        "mesh_vertices": len(obj.mesh.vertices) if obj.mesh else None,
        "mesh_faces": len(obj.mesh.faces) if obj.mesh else None,
        "point_count": len(obj.point_cloud.points) if obj.point_cloud else None,
    }


def _print_result_rich(result: PipelineResult, generator_name: str) -> None:
    """Print pipeline result using rich formatting."""
    obj = result.math_object
    console.print(f"[bold green]Generator:[/bold green] {generator_name}")
    console.print(f"[bold]Seed:[/bold] {obj.seed}")
    console.print(f"[bold]Parameters:[/bold] {obj.parameters}")
    if obj.mesh:
        console.print(
            f"[bold]Mesh:[/bold] {len(obj.mesh.vertices)} vertices, "
            f"{len(obj.mesh.faces)} faces"
        )
    if obj.point_cloud:
        console.print(f"[bold]Points:[/bold] {len(obj.point_cloud.points)}")
    _print_timings_rich(result.timings)
    _print_validation_rich(result)


def _print_timings_rich(timings: dict[str, float]) -> None:
    """Print timing table with rich."""
    if not timings:
        return
    table = Table(title="Timings")
    table.add_column("Stage")
    table.add_column("Seconds", justify="right")
    for stage, elapsed in timings.items():
        table.add_row(stage, f"{elapsed:.4f}")
    total = sum(timings.values())
    table.add_row("[bold]total[/bold]", f"[bold]{total:.4f}[/bold]")
    console.print(table)


def _print_validation_rich(result: PipelineResult) -> None:
    """Print validation results with rich."""
    if not result.validation.checks:
        return
    for check in result.validation.checks:
        icon = "[green]PASS[/green]" if check.passed else "[red]FAIL[/red]"
        console.print(f"  {icon} {check.name}: {check.message}")


def _write_report(path: Path, data: dict[str, Any]) -> None:
    """Write a JSON report file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, default=str), encoding="utf-8")
    console.print(f"Report written to {path}")


@app.command()
def generate(
    generator_name: str = typer.Argument(help="Name of the generator to run"),
    param: Optional[list[str]] = typer.Option(None, "--param", help="key=value parameter"),
    seed: int = typer.Option(42, "--seed", help="Random seed for reproducibility"),
    output: Optional[Path] = typer.Option(None, "--output", help="Output file path"),
    fmt: Optional[str] = typer.Option(None, "--format", help="Export format"),
    dry_run: bool = typer.Option(False, "--dry-run", help="Show what would happen"),
    report: Optional[Path] = typer.Option(None, "--report", help="Write JSON report"),
    json_output: bool = typer.Option(False, "--json", help="Output as JSON"),
    verbose: bool = typer.Option(False, "--verbose", help="Enable debug logging"),
    quiet: bool = typer.Option(False, "--quiet", help="Suppress non-error output"),
) -> None:
    """Run a generator through the pipeline."""
    _configure_logging(verbose, quiet)

    try:
        gen_meta = get_generator_meta(generator_name)
    except KeyError:
        _error_exit(f"Unknown generator: {generator_name!r}", json_output)
        return  # unreachable, for type checker

    gen_instance: GeneratorBase = gen_meta.generator_class()
    params = _parse_params(param or [])

    if dry_run:
        _handle_dry_run(gen_instance, generator_name, params, seed, output, json_output)
        return

    merged_params = gen_instance.get_default_params()
    merged_params.update(params)

    container = Container()
    placement = PlacementPolicy()

    export_config = None
    if output is not None:
        export_config = ExportConfig(path=output, fmt=fmt)

    result = run(
        generator=gen_instance,
        params=merged_params,
        seed=seed,
        container=container,
        placement=placement,
        export_config=export_config,
    )

    result_dict = _result_to_dict(result, generator_name)
    exit_code = _exit_code_for_result(result)

    if report is not None:
        _write_report(report, result_dict)

    if json_output:
        typer.echo(json.dumps(result_dict, indent=2, default=str))
    elif not quiet:
        _print_result_rich(result, generator_name)

    raise typer.Exit(code=exit_code)


def _handle_dry_run(
    gen_instance: GeneratorBase,
    generator_name: str,
    params: dict[str, Any],
    seed: int,
    output: Optional[Path],
    json_output: bool,
) -> None:
    """Handle --dry-run: report what would happen without running the pipeline."""
    merged = gen_instance.get_default_params()
    merged.update(params)
    info = {
        "dry_run": True,
        "generator": generator_name,
        "seed": seed,
        "parameters": merged,
        "output": str(output) if output else None,
        "stages": ["generate", "represent", "transform", "validate"],
    }
    if output is not None:
        info["stages"].append("export")

    if json_output:
        typer.echo(json.dumps(info, indent=2, default=str))
    else:
        console.print("[bold yellow]Dry run[/bold yellow] — no files will be written")
        console.print(f"  Generator: {generator_name}")
        console.print(f"  Seed: {seed}")
        console.print(f"  Parameters: {merged}")
        console.print(f"  Output: {output or '(none)'}")
        console.print(f"  Stages: {', '.join(info['stages'])}")

    raise typer.Exit(code=EXIT_SUCCESS)


@app.command("list")
def list_cmd(
    json_output: bool = typer.Option(False, "--json", help="Output as JSON"),
) -> None:
    """List all available generators."""
    generators = list_generators()
    if json_output:
        data = [
            {
                "name": g.name,
                "category": g.category,
                "aliases": g.aliases,
                "description": g.description,
            }
            for g in generators
        ]
        typer.echo(json.dumps(data, indent=2))
    else:
        table = Table(title="Available Generators")
        table.add_column("Name")
        table.add_column("Category")
        table.add_column("Description")
        table.add_column("Aliases")
        for g in generators:
            table.add_row(g.name, g.category, g.description, ", ".join(g.aliases) or "—")
        console.print(table)


@app.command()
def info(
    generator_name: str = typer.Argument(help="Generator name to inspect"),
    json_output: bool = typer.Option(False, "--json", help="Output as JSON"),
) -> None:
    """Show detailed info and parameter schema for a generator."""
    try:
        meta = get_generator_meta(generator_name)
    except KeyError:
        _error_exit(f"Unknown generator: {generator_name!r}", json_output)
        return

    gen_instance: GeneratorBase = meta.generator_class()
    defaults = gen_instance.get_default_params()
    schema = gen_instance.get_param_schema()

    data = {
        "name": meta.name,
        "category": meta.category,
        "aliases": meta.aliases,
        "description": meta.description,
        "resolution_params": meta.resolution_params,
        "default_params": defaults,
        "param_schema": schema,
    }

    if json_output:
        typer.echo(json.dumps(data, indent=2, default=str))
    else:
        console.print(f"[bold]{meta.name}[/bold] ({meta.category})")
        console.print(f"  {meta.description}")
        if meta.aliases:
            console.print(f"  Aliases: {', '.join(meta.aliases)}")
        console.print(f"  Default params: {defaults}")
        if meta.resolution_params:
            console.print(f"  Resolution params: {meta.resolution_params}")


@app.command()
def validate(
    generator_name: str = typer.Argument(help="Generator name to validate"),
    param: Optional[list[str]] = typer.Option(None, "--param", help="key=value parameter"),
    seed: int = typer.Option(42, "--seed", help="Random seed"),
    json_output: bool = typer.Option(False, "--json", help="Output as JSON"),
    verbose: bool = typer.Option(False, "--verbose", help="Enable debug logging"),
    quiet: bool = typer.Option(False, "--quiet", help="Suppress non-error output"),
) -> None:
    """Generate and validate without exporting."""
    _configure_logging(verbose, quiet)

    try:
        get_generator_meta(generator_name)
    except KeyError:
        _error_exit(f"Unknown generator: {generator_name!r}", json_output)
        return

    params = _parse_params(param or [])
    container = Container()
    placement = PlacementPolicy()

    result = run(
        generator=generator_name,
        params=params if params else None,
        seed=seed,
        container=container,
        placement=placement,
    )

    exit_code = _exit_code_for_result(result)
    checks_data = [
        {
            "name": c.name,
            "passed": c.passed,
            "severity": c.severity.value,
            "message": c.message,
        }
        for c in result.validation.checks
    ]

    if json_output:
        data = {"passed": result.validation.passed, "checks": checks_data}
        typer.echo(json.dumps(data, indent=2))
    elif not quiet:
        console.print(f"[bold]Validation for {generator_name}[/bold]")
        _print_validation_rich(result)
        status = "[green]PASSED[/green]" if result.validation.passed else "[red]FAILED[/red]"
        console.print(f"Overall: {status}")

    raise typer.Exit(code=exit_code)


def _error_exit(message: str, json_output: bool) -> None:
    """Print error and exit with code 2."""
    if json_output:
        typer.echo(json.dumps({"error": message}))
    else:
        console.print(f"[red]Error: {message}[/red]", highlight=False)
    raise typer.Exit(code=EXIT_ERROR)


def main() -> None:
    """Entry point for the CLI."""
    app()


if __name__ == "__main__":
    main()
