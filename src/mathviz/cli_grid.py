"""CLI grid commands: manage the installation grid manifest."""

import json
import logging
from pathlib import Path
from typing import Any, Optional

import typer
from rich.console import Console
from rich.table import Table

from mathviz.core.grid import BlockStatus, GridBlock, GridManifest

logger = logging.getLogger(__name__)

grid_app = typer.Typer(
    name="grid",
    help="Manage the installation grid manifest.",
    add_completion=False,
)

console = Console()

EXIT_SUCCESS = 0
EXIT_ERROR = 2


def _exit_error(message: str, json_output: bool) -> None:
    """Print an error message and exit with code 2."""
    if json_output:
        typer.echo(json.dumps({"error": message}))
    else:
        console.print(f"[red]Error: {message}[/red]")
    raise typer.Exit(code=EXIT_ERROR)


def _load_manifest(path: Path, json_output: bool = False) -> GridManifest:
    """Load a grid manifest or exit with an error if not found."""
    try:
        return GridManifest.load(path)
    except FileNotFoundError:
        _exit_error(f"Grid manifest not found: {path}", json_output)
        raise  # unreachable, for type checker


@grid_app.command("init")
def init_grid(
    rows: int = typer.Argument(help="Number of rows"),
    cols: int = typer.Argument(help="Number of columns"),
    path: Path = typer.Option(Path("grid.toml"), "--path", help="Manifest file path"),
    json_output: bool = typer.Option(False, "--json", help="Output as JSON"),
) -> None:
    """Create a new grid manifest."""
    manifest = GridManifest.create(rows, cols, path)
    manifest.save()
    if json_output:
        typer.echo(json.dumps({"rows": rows, "cols": cols, "path": str(path)}))
    else:
        console.print(f"Created {rows}x{cols} grid manifest at {path}")


@grid_app.command("show")
def show_grid(
    path: Path = typer.Option(Path("grid.toml"), "--path", help="Manifest file path"),
    json_output: bool = typer.Option(False, "--json", help="Output as JSON"),
) -> None:
    """Display the grid as an ASCII table or JSON."""
    manifest = _load_manifest(path, json_output)
    if json_output:
        typer.echo(json.dumps(manifest.to_dict(), indent=2))
        return
    _print_grid_table(manifest)


def _print_grid_table(manifest: GridManifest) -> None:
    """Render the grid as a Rich table."""
    table = Table(title=f"Grid ({manifest.rows}x{manifest.cols})")
    table.add_column("", justify="right", style="dim")
    for c in range(manifest.cols):
        table.add_column(str(c), justify="center")
    for r in range(manifest.rows):
        cells: list[str] = []
        for c in range(manifest.cols):
            block = manifest.get_block(r, c)
            cells.append(_format_cell(block.preset, block.status))
        table.add_row(str(r), *cells)
    console.print(table)


def _format_cell(preset: str | None, status: BlockStatus) -> str:
    """Format a single grid cell for display."""
    if preset is None:
        return "[dim]·[/dim]"
    style = _STATUS_STYLES.get(status, "")
    label = preset[:8]
    return f"[{style}]{label}[/{style}]" if style else label


_STATUS_STYLES: dict[BlockStatus, str] = {
    BlockStatus.EMPTY: "dim",
    BlockStatus.ASSIGNED: "yellow",
    BlockStatus.EXPORTED: "green",
    BlockStatus.ERROR: "red",
}


@grid_app.command("assign")
def assign_block(
    row: int = typer.Argument(help="Row position (0-indexed)"),
    col: int = typer.Argument(help="Column position (0-indexed)"),
    preset: str = typer.Argument(help="Generator preset name"),
    config: Optional[str] = typer.Option(None, "--config", help="Per-block config path"),
    path: Path = typer.Option(Path("grid.toml"), "--path", help="Manifest file path"),
    json_output: bool = typer.Option(False, "--json", help="Output as JSON"),
) -> None:
    """Assign a preset to a grid position."""
    manifest = _load_manifest(path, json_output)
    try:
        block = manifest.assign(row, col, preset, config_path=config)
    except ValueError as exc:
        _exit_error(str(exc), json_output)
    manifest.save()
    if json_output:
        typer.echo(json.dumps(block.to_dict()))
    else:
        console.print(
            f"Assigned [bold]{preset}[/bold] to position ({row}, {col})"
        )


@grid_app.command("status")
def block_status(
    row: int = typer.Argument(help="Row position"),
    col: int = typer.Argument(help="Column position"),
    set_status: Optional[str] = typer.Option(
        None, "--set", help="Set status (assigned/exported/error/empty)"
    ),
    path: Path = typer.Option(Path("grid.toml"), "--path", help="Manifest file path"),
    json_output: bool = typer.Option(False, "--json", help="Output as JSON"),
) -> None:
    """Show or update the status of a grid block."""
    manifest = _load_manifest(path, json_output)
    if set_status is not None:
        try:
            new_status = BlockStatus(set_status)
        except ValueError:
            valid = [s.value for s in BlockStatus]
            _exit_error(f"Invalid status {set_status!r}. Valid: {valid}", json_output)
        try:
            block = manifest.set_status(row, col, new_status)
        except ValueError as exc:
            _exit_error(str(exc), json_output)
        manifest.save()
    else:
        try:
            block = manifest.get_block(row, col)
        except ValueError as exc:
            _exit_error(str(exc), json_output)
    if json_output:
        typer.echo(json.dumps(block.to_dict()))
    else:
        console.print(
            f"Block ({row}, {col}): "
            f"preset={block.preset or '(none)'}, "
            f"status={block.status.value}"
        )


@grid_app.command("neighbors")
def show_neighbors(
    row: int = typer.Argument(help="Row position"),
    col: int = typer.Argument(help="Column position"),
    path: Path = typer.Option(Path("grid.toml"), "--path", help="Manifest file path"),
    json_output: bool = typer.Option(False, "--json", help="Output as JSON"),
) -> None:
    """Show the 8 surrounding blocks for a position."""
    manifest = _load_manifest(path, json_output)
    try:
        nbrs = manifest.neighbors(row, col)
    except ValueError as exc:
        _exit_error(str(exc), json_output)
    if json_output:
        typer.echo(json.dumps([b.to_dict() for b in nbrs], indent=2))
    else:
        console.print(f"[bold]Neighbors of ({row}, {col}):[/bold] {len(nbrs)} blocks")
        for b in nbrs:
            console.print(
                f"  ({b.row}, {b.col}): "
                f"preset={b.preset or '(none)'}, status={b.status.value}"
            )


@grid_app.command("summary")
def grid_summary(
    path: Path = typer.Option(Path("grid.toml"), "--path", help="Manifest file path"),
    json_output: bool = typer.Option(False, "--json", help="Output as JSON"),
) -> None:
    """Show counts of blocks by status."""
    manifest = _load_manifest(path, json_output)
    counts = manifest.summary()
    total = manifest.rows * manifest.cols
    if json_output:
        typer.echo(json.dumps({"total": total, **counts}))
    else:
        console.print(f"[bold]Grid Summary ({manifest.rows}x{manifest.cols})[/bold]")
        for status, count in counts.items():
            console.print(f"  {status}: {count}")
        console.print(f"  [bold]total: {total}[/bold]")


@grid_app.command("export-all")
def export_all(
    path: Path = typer.Option(Path("grid.toml"), "--path", help="Manifest file path"),
    output_dir: Path = typer.Option(
        Path("export"), "--output-dir", help="Output directory"
    ),
    fmt: Optional[str] = typer.Option(None, "--format", help="Export format"),
    json_output: bool = typer.Option(False, "--json", help="Output as JSON"),
) -> None:
    """Batch export all assigned blocks sequentially."""
    from mathviz.core.config import load_project_config

    manifest = _load_manifest(path, json_output)
    assigned = [
        b for b in manifest.blocks.values()
        if b.preset is not None and b.status in (BlockStatus.ASSIGNED, BlockStatus.ERROR)
    ]
    if not assigned:
        if json_output:
            typer.echo(json.dumps({"exported": 0, "results": []}))
        else:
            console.print("No assigned blocks to export.")
        return

    output_dir.mkdir(parents=True, exist_ok=True)
    project_cfg = load_project_config()
    results: list[dict[str, Any]] = []

    for block in sorted(assigned, key=lambda b: (b.row, b.col)):
        result = _export_single_block(
            manifest, block, output_dir, fmt, json_output, project_cfg,
        )
        results.append(result)

    manifest.save()

    if json_output:
        typer.echo(json.dumps({"exported": len(results), "results": results}, indent=2))
    else:
        exported = sum(1 for r in results if r["success"])
        failed = len(results) - exported
        console.print(f"Exported {exported} blocks, {failed} errors.")


def _export_single_block(
    manifest: GridManifest,
    block: GridBlock,
    output_dir: Path,
    fmt: str | None,
    json_output: bool,
    project_cfg: dict[str, Any],
) -> dict[str, Any]:
    """Export a single block through the pipeline."""
    from mathviz.core.config import load_object_config, resolve_config
    from mathviz.pipeline.runner import ExportConfig, run

    suffix = f".{fmt}" if fmt else ".ply"
    out_path = output_dir / f"block_{block.row}_{block.col}{suffix}"
    result_info: dict[str, Any] = {
        "row": block.row, "col": block.col,
        "preset": block.preset, "success": False,
    }

    try:
        object_cfg = (
            load_object_config(Path(block.config_path))
            if block.config_path else None
        )
        resolved = resolve_config(project=project_cfg, object_config=object_cfg)

        export_config = ExportConfig(path=out_path, fmt=fmt)
        seed = resolved.seed if resolved.seed is not None else 42
        params = resolved.params if resolved.params is not None else None
        run(
            generator=block.preset,
            params=params,
            seed=seed,
            container=resolved.container,
            placement=resolved.placement,
            sampler_config=resolved.sampler_config,
            export_config=export_config,
        )
        manifest.set_status(block.row, block.col, BlockStatus.EXPORTED)
        result_info["success"] = True
        result_info["path"] = str(out_path)
        if not json_output:
            console.print(
                f"  [green]OK[/green] ({block.row},{block.col}) "
                f"{block.preset} -> {out_path}"
            )
    except (ValueError, FileNotFoundError, KeyError, OSError) as exc:
        manifest.set_status(block.row, block.col, BlockStatus.ERROR)
        result_info["error"] = str(exc)
        if not json_output:
            console.print(
                f"  [red]FAIL[/red] ({block.row},{block.col}) "
                f"{block.preset}: {exc}"
            )
        logger.error("Export failed for (%d,%d): %s", block.row, block.col, exc)

    return result_info
