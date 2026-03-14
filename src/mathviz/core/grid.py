"""Grid manifest: tracks preset/config assignments for installation grid blocks."""

import enum
import logging
import tomllib
from pathlib import Path
from typing import Any

from pydantic import BaseModel, field_validator

logger = logging.getLogger(__name__)

DEFAULT_GRID_PATH = "grid.toml"


class BlockStatus(str, enum.Enum):
    """Status of a grid block in the installation."""

    EMPTY = "empty"
    ASSIGNED = "assigned"
    EXPORTED = "exported"
    ERROR = "error"


class GridBlock(BaseModel):
    """A single block in the installation grid."""

    row: int
    col: int
    preset: str | None = None
    config_path: str | None = None
    status: BlockStatus = BlockStatus.EMPTY


class GridManifest(BaseModel):
    """Manifest tracking all blocks in the installation grid."""

    rows: int
    cols: int
    blocks: dict[str, GridBlock] = {}
    path: Path = Path(DEFAULT_GRID_PATH)

    @field_validator("rows", "cols")
    @classmethod
    def _positive_dimensions(cls, v: int) -> int:
        """Validate grid dimensions are positive."""
        if v < 1:
            raise ValueError(f"Grid dimension must be >= 1, got {v}")
        return v

    @staticmethod
    def _key(row: int, col: int) -> str:
        """Create a dict key from row/col coordinates."""
        return f"{row},{col}"

    def _validate_bounds(self, row: int, col: int) -> None:
        """Raise ValueError if position is out of bounds."""
        if row < 0 or row >= self.rows or col < 0 or col >= self.cols:
            raise ValueError(
                f"Position ({row}, {col}) is out of bounds "
                f"for grid {self.rows}x{self.cols}"
            )

    def assign(
        self,
        row: int,
        col: int,
        preset: str,
        config_path: str | None = None,
    ) -> GridBlock:
        """Assign a preset to a grid position."""
        self._validate_bounds(row, col)
        block = GridBlock(
            row=row,
            col=col,
            preset=preset,
            config_path=config_path,
            status=BlockStatus.ASSIGNED,
        )
        self.blocks[self._key(row, col)] = block
        return block

    def get_block(self, row: int, col: int) -> GridBlock:
        """Get the block at a position, returning empty block if unassigned."""
        self._validate_bounds(row, col)
        key = self._key(row, col)
        if key in self.blocks:
            return self.blocks[key]
        return GridBlock(row=row, col=col)

    def set_status(self, row: int, col: int, status: BlockStatus) -> GridBlock:
        """Update the status of a block."""
        self._validate_bounds(row, col)
        key = self._key(row, col)
        if key not in self.blocks:
            self.blocks[key] = GridBlock(row=row, col=col)
        self.blocks[key].status = status
        return self.blocks[key]

    def neighbors(self, row: int, col: int) -> list[GridBlock]:
        """Return up to 8 surrounding blocks for a position."""
        self._validate_bounds(row, col)
        result: list[GridBlock] = []
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                if dr == 0 and dc == 0:
                    continue
                nr, nc = row + dr, col + dc
                if 0 <= nr < self.rows and 0 <= nc < self.cols:
                    result.append(self.get_block(nr, nc))
        return result

    def summary(self) -> dict[str, int]:
        """Count blocks by status, including unassigned empties."""
        counts: dict[str, int] = {s.value: 0 for s in BlockStatus}
        for r in range(self.rows):
            for c in range(self.cols):
                block = self.get_block(r, c)
                counts[block.status.value] += 1
        return counts

    def save(self, path: Path | None = None) -> Path:
        """Save manifest to a TOML file."""
        target = path or self.path
        lines = [
            f"rows = {self.rows}",
            f"cols = {self.cols}",
            "",
        ]
        for key in sorted(self.blocks):
            block = self.blocks[key]
            lines.append(f"[blocks.\"{key}\"]")
            lines.append(f"row = {block.row}")
            lines.append(f"col = {block.col}")
            if block.preset is not None:
                lines.append(f'preset = "{block.preset}"')
            if block.config_path is not None:
                lines.append(f'config_path = "{block.config_path}"')
            lines.append(f'status = "{block.status.value}"')
            lines.append("")

        target.write_text("\n".join(lines), encoding="utf-8")
        logger.info("Saved grid manifest to %s", target)
        return target

    @classmethod
    def load(cls, path: Path) -> "GridManifest":
        """Load a manifest from a TOML file."""
        if not path.exists():
            raise FileNotFoundError(f"Grid manifest not found: {path}")
        with open(path, "rb") as f:
            data = tomllib.load(f)
        rows = data["rows"]
        cols = data["cols"]
        blocks: dict[str, GridBlock] = {}
        for key, block_data in data.get("blocks", {}).items():
            blocks[key] = GridBlock(**block_data)
        return cls(rows=rows, cols=cols, blocks=blocks, path=path)

    @classmethod
    def create(cls, rows: int, cols: int, path: Path | None = None) -> "GridManifest":
        """Create a new empty grid manifest."""
        manifest_path = path or Path(DEFAULT_GRID_PATH)
        return cls(rows=rows, cols=cols, path=manifest_path)

    def assigned_blocks(self) -> list[GridBlock]:
        """Return all blocks that have been assigned a preset."""
        return [
            b for b in self.blocks.values()
            if b.status != BlockStatus.EMPTY
        ]

    def to_dict(self) -> dict[str, Any]:
        """Convert manifest to a JSON-serializable dict."""
        return {
            "rows": self.rows,
            "cols": self.cols,
            "blocks": {
                k: {
                    "row": b.row,
                    "col": b.col,
                    "preset": b.preset,
                    "config_path": b.config_path,
                    "status": b.status.value,
                }
                for k, b in self.blocks.items()
            },
        }
