"""MathViz CLI: Typer-based command-line interface for the pipeline."""

import json
import logging
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional
from urllib.parse import urlencode

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
    """Determine exit code from pipeline result validation.

    Uses ValidationResult.passed which returns True when there are zero errors
    (warnings are acceptable). Exit code 1 only when there are actual errors.
    Exit code 2 is reserved for CLI-level errors (unknown generator, bad args).
    """
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


def _run_pipeline(
    generator_name: str,
    params: dict[str, Any],
    seed: int,
    json_output: bool,
    export_config: ExportConfig | None = None,
) -> PipelineResult:
    """Shared pipeline execution for generate and validate commands."""
    gen_instance = _resolve_generator(generator_name, json_output)
    container = Container()
    placement = PlacementPolicy()

    return run(
        generator=gen_instance,
        params=params if params else None,
        seed=seed,
        container=container,
        placement=placement,
        export_config=export_config,
    )


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
    params = _parse_params(param or [])

    if dry_run:
        gen_instance = _resolve_generator(generator_name, json_output)
        _handle_dry_run(gen_instance, generator_name, params, seed, output, json_output)
        return

    export_config = None
    if output is not None:
        export_config = ExportConfig(path=output, fmt=fmt)

    result = _run_pipeline(generator_name, params, seed, json_output, export_config)
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
    seed: int = typer.Option(42, "--seed", help="Random seed"),
    json_output: bool = typer.Option(False, "--json", help="Output as JSON"),
    verbose: bool = typer.Option(False, "--verbose", help="Enable debug logging"),
    quiet: bool = typer.Option(False, "--quiet", help="Suppress non-error output"),
) -> None:
    """Generate and validate without exporting."""
    _configure_logging(verbose, quiet)
    params = _parse_params(param or [])

    result = _run_pipeline(generator_name, params, seed, json_output)
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


@dataclass
class PreviewConfig:
    """Configuration for the preview server."""

    target: str
    port: int
    no_open: bool
    quiet: bool
    query_params: dict[str, Any]


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
    _configure_logging(verbose, quiet)
    query_params = _build_preview_query(target, param or [], seed)
    config = PreviewConfig(
        target=target, port=port, no_open=no_open, quiet=quiet, query_params=query_params,
    )
    _run_preview_server(config)


def _build_preview_query(target: str, param_list: list[str], seed: int) -> dict[str, Any]:
    """Build query parameters dict for the preview URL."""
    target_path = Path(target)
    if target_path.is_file():
        return {"file": str(target_path.resolve())}
    params = _parse_params(param_list)
    query: dict[str, Any] = {"generator": target, "seed": seed}
    query.update(params)
    return query


def _run_preview_server(config: PreviewConfig) -> None:
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


def main() -> None:
    """Entry point for the CLI."""
    app()


if __name__ == "__main__":
    main()
