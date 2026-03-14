"""MathViz CLI: Typer-based command-line interface for the pipeline."""

import json
import logging
from pathlib import Path
from typing import Any, Optional

import typer
from rich.console import Console
from rich.table import Table

from mathviz.cli_output import (
    error_exit,
    print_result_rich,
    print_validation_rich,
    result_to_dict,
    serialize_checks,
    write_report,
)
from mathviz.cli_preview import register_preview_command
from mathviz.core.config import (
    deep_merge,
    load_object_config,
    load_project_config,
    load_sampling_profile,
    resolve_config,
)
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

# Pipeline stages in execution order (matches runner.py chain)
_BASE_STAGES = ["generate", "represent", "transform", "sample", "validate"]


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


def _coerce_value(value: str) -> int | float | str:
    """Coerce a string value to int, float, or leave as string."""
    for caster in (int, float):
        try:
            return caster(value)
        except ValueError:
            continue
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
    """Determine exit code: 0 if passed, 1 if validation errors."""
    if not result.validation.passed:
        return EXIT_VALIDATION_WARNING
    return EXIT_SUCCESS


def _resolve_generator(generator_name: str, json_output: bool) -> GeneratorBase:
    """Resolve a generator name to an instance, exiting on failure."""
    try:
        meta = get_generator_meta(generator_name)
    except KeyError:
        error_exit(f"Unknown generator: {generator_name!r}", json_output)
        raise  # unreachable, for type checker
    return meta.generator_class.create(resolved_name=generator_name)


def _build_cli_overrides(
    params: dict[str, Any],
    seed: int | None = None,
    width: float | None = None,
    height: float | None = None,
    depth: float | None = None,
) -> dict[str, Any]:
    """Build CLI override dict from explicitly provided flag values."""
    overrides: dict[str, Any] = {}
    if params:
        overrides["params"] = params
    if seed is not None:
        overrides["seed"] = seed
    dims = {"width_mm": width, "height_mm": height, "depth_mm": depth}
    container_overrides = {k: v for k, v in dims.items() if v is not None}
    if container_overrides:
        overrides["container"] = container_overrides
    return overrides


def _run_pipeline(
    generator_name: str,
    params: dict[str, Any],
    seed: int | None,
    json_output: bool,
    export_config: ExportConfig | None = None,
    config_path: Path | None = None,
    profile_name: str | None = None,
    container_width: float | None = None,
    container_height: float | None = None,
    container_depth: float | None = None,
) -> PipelineResult:
    """Shared pipeline execution for generate and validate commands."""
    gen_instance = _resolve_generator(generator_name, json_output)

    # Load config layers
    project_cfg = load_project_config()
    object_cfg = _load_safe(load_object_config, json_output, config_path) if config_path else None
    if profile_name:
        profile_cfg = _load_safe(load_sampling_profile, json_output, profile_name)
        object_cfg = deep_merge(object_cfg, profile_cfg) if object_cfg else profile_cfg

    cli_overrides = _build_cli_overrides(
        params,
        seed,
        container_width,
        container_height,
        container_depth,
    )
    resolved = resolve_config(
        project=project_cfg,
        object_config=object_cfg,
        cli_overrides=cli_overrides,
    )

    effective_seed = resolved.seed if resolved.seed is not None else 42

    return run(
        generator=gen_instance,
        params=resolved.params if resolved.params else None,
        seed=effective_seed,
        container=resolved.container,
        placement=resolved.placement,
        sampler_config=resolved.sampler_config,
        export_config=export_config,
    )


def _load_safe(loader: Any, json_output: bool, *args: Any) -> dict[str, Any]:
    """Load a config/profile, calling error_exit on FileNotFoundError."""
    try:
        return loader(*args)
    except FileNotFoundError as exc:
        error_exit(str(exc), json_output)
        raise  # unreachable


@app.command()
def generate(
    generator_name: str = typer.Argument(help="Name of the generator to run"),
    param: Optional[list[str]] = typer.Option(None, "--param", help="key=value parameter"),
    seed: Optional[int] = typer.Option(None, "--seed", help="Random seed (default: 42)"),
    output: Optional[Path] = typer.Option(None, "--output", help="Output file path"),
    fmt: Optional[str] = typer.Option(None, "--format", help="Export format"),
    config: Optional[Path] = typer.Option(None, "--config", help="Per-object TOML config"),
    profile: Optional[str] = typer.Option(None, "--profile", help="Sampling profile name"),
    container_width: Optional[float] = typer.Option(None, "--width", help="Container width (mm)"),
    container_height: Optional[float] = typer.Option(
        None,
        "--height",
        help="Container height (mm)",
    ),
    container_depth: Optional[float] = typer.Option(None, "--depth", help="Container depth (mm)"),
    dry_run: bool = typer.Option(False, "--dry-run", help="Show what would happen"),
    report: Optional[Path] = typer.Option(None, "--report", help="Write JSON report"),
    json_output: bool = typer.Option(False, "--json", help="Output as JSON"),
    verbose: bool = typer.Option(False, "--verbose", help="Enable debug logging"),
    quiet: bool = typer.Option(False, "--quiet", help="Suppress non-error output"),
) -> None:
    """Run a generator through the pipeline."""
    _configure_logging(verbose, quiet)
    params = _parse_params(param or [])

    if dry_run:
        gen_instance = _resolve_generator(generator_name, json_output)
        effective_seed = seed if seed is not None else 42
        _handle_dry_run(gen_instance, generator_name, params, effective_seed, output, json_output)
        return

    export_config = None
    if output is not None:
        export_config = ExportConfig(path=output, fmt=fmt)

    result = _run_pipeline(
        generator_name,
        params,
        seed,
        json_output,
        export_config,
        config_path=config,
        profile_name=profile,
        container_width=container_width,
        container_height=container_height,
        container_depth=container_depth,
    )
    result_dict = result_to_dict(result, generator_name)
    exit_code = _exit_code_for_result(result)

    if report is not None:
        write_report(report, result_dict, silent=json_output)

    if json_output:
        typer.echo(json.dumps(result_dict, indent=2, default=str))
    elif not quiet:
        print_result_rich(result, generator_name)

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
    stages = list(_BASE_STAGES)
    if output is not None:
        stages.append("export")
    info = {
        "dry_run": True,
        "generator": generator_name,
        "seed": seed,
        "parameters": merged,
        "output": str(output) if output else None,
        "stages": stages,
    }
    if json_output:
        typer.echo(json.dumps(info, indent=2, default=str))
    else:
        console.print("[bold yellow]Dry run[/bold yellow] — no files will be written")
        console.print(f"  Generator: {generator_name}")
        console.print(f"  Seed: {seed}")
        console.print(f"  Parameters: {merged}")
        console.print(f"  Output: {output or '(none)'}")
        console.print(f"  Stages: {', '.join(stages)}")

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
    gen_instance = _resolve_generator(generator_name, json_output)
    meta = get_generator_meta(generator_name)
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
    seed: Optional[int] = typer.Option(None, "--seed", help="Random seed (default: 42)"),
    config: Optional[Path] = typer.Option(None, "--config", help="Per-object TOML config"),
    profile: Optional[str] = typer.Option(None, "--profile", help="Sampling profile name"),
    container_width: Optional[float] = typer.Option(None, "--width", help="Container width (mm)"),
    container_height: Optional[float] = typer.Option(
        None,
        "--height",
        help="Container height (mm)",
    ),
    container_depth: Optional[float] = typer.Option(None, "--depth", help="Container depth (mm)"),
    json_output: bool = typer.Option(False, "--json", help="Output as JSON"),
    verbose: bool = typer.Option(False, "--verbose", help="Enable debug logging"),
    quiet: bool = typer.Option(False, "--quiet", help="Suppress non-error output"),
) -> None:
    """Generate and validate without exporting."""
    _configure_logging(verbose, quiet)
    params = _parse_params(param or [])

    result = _run_pipeline(
        generator_name,
        params,
        seed,
        json_output,
        config_path=config,
        profile_name=profile,
        container_width=container_width,
        container_height=container_height,
        container_depth=container_depth,
    )
    exit_code = _exit_code_for_result(result)

    if json_output:
        data = {
            "passed": result.validation.passed,
            "checks": serialize_checks(result.validation.checks),
        }
        typer.echo(json.dumps(data, indent=2))
    elif not quiet:
        console.print(f"[bold]Validation for {generator_name}[/bold]")
        print_validation_rich(result)
        status = "[green]PASSED[/green]" if result.validation.passed else "[red]FAILED[/red]"
        console.print(f"Overall: {status}")

    raise typer.Exit(code=exit_code)


register_preview_command(app, _parse_params, _configure_logging, console)


def main() -> None:
    """Entry point for the CLI."""
    app()


if __name__ == "__main__":
    main()
