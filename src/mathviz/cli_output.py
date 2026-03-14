"""CLI output helpers: JSON serialization and rich terminal formatting."""

import json
from pathlib import Path
from typing import Any

import typer
from rich.console import Console
from rich.table import Table

from mathviz.core.validator import CheckResult
from mathviz.pipeline.runner import PipelineResult

console = Console()


def serialize_checks(checks: list[CheckResult]) -> list[dict[str, Any]]:
    """Convert validation checks to a JSON-serializable list of dicts."""
    return [
        {
            "name": c.name,
            "passed": c.passed,
            "severity": c.severity.value,
            "message": c.message,
        }
        for c in checks
    ]


def result_to_dict(result: PipelineResult, generator_name: str) -> dict[str, Any]:
    """Convert a PipelineResult to a JSON-serializable dict."""
    obj = result.math_object
    return {
        "generator": generator_name,
        "seed": obj.seed,
        "parameters": obj.parameters,
        "coord_space": obj.coord_space.value,
        "timings": result.timings,
        "validation": {
            "passed": result.validation.passed,
            "checks": serialize_checks(result.validation.checks),
        },
        "export_path": str(result.export_path) if result.export_path else None,
        "mesh_vertices": len(obj.mesh.vertices) if obj.mesh else None,
        "mesh_faces": len(obj.mesh.faces) if obj.mesh else None,
        "point_count": len(obj.point_cloud.points) if obj.point_cloud else None,
    }


def print_result_rich(result: PipelineResult, generator_name: str) -> None:
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
    print_timings_rich(result.timings)
    print_validation_rich(result)


def print_timings_rich(timings: dict[str, float]) -> None:
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


def print_validation_rich(result: PipelineResult) -> None:
    """Print validation results with rich."""
    if not result.validation.checks:
        return
    for check in result.validation.checks:
        icon = "[green]PASS[/green]" if check.passed else "[red]FAIL[/red]"
        console.print(f"  {icon} {check.name}: {check.message}")


def write_report(path: Path, data: dict[str, Any], *, silent: bool = False) -> None:
    """Write a JSON report file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, default=str), encoding="utf-8")
    if not silent:
        console.print(f"Report written to {path}")


def error_exit(message: str, json_output: bool) -> None:
    """Print error and exit with code 2."""
    from mathviz.cli import EXIT_ERROR

    if json_output:
        typer.echo(json.dumps({"error": message}))
    else:
        console.print(f"[red]Error: {message}[/red]", highlight=False)
    raise typer.Exit(code=EXIT_ERROR)
