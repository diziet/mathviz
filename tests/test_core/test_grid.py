"""Tests for the grid manifest and grid block models."""

import json
from pathlib import Path

import pytest

from mathviz.core.grid import BlockStatus, GridBlock, GridManifest


class TestGridBlock:
    """Tests for the GridBlock model."""

    def test_default_block_is_empty(self) -> None:
        """A new block defaults to EMPTY status with no preset."""
        block = GridBlock(row=0, col=0)
        assert block.status == BlockStatus.EMPTY
        assert block.preset is None
        assert block.config_path is None

    def test_block_with_preset(self) -> None:
        """A block can be created with a preset and config."""
        block = GridBlock(
            row=1, col=2, preset="lorenz", config_path="configs/a.toml",
            status=BlockStatus.ASSIGNED,
        )
        assert block.preset == "lorenz"
        assert block.config_path == "configs/a.toml"
        assert block.status == BlockStatus.ASSIGNED


class TestGridManifestAssign:
    """Tests for assigning blocks and retrieving them."""

    def test_assign_and_retrieve(self) -> None:
        """Assigning a block and retrieving it returns correct preset/config."""
        manifest = GridManifest.create(4, 4)
        manifest.assign(1, 2, "lorenz", config_path="my.toml")
        block = manifest.get_block(1, 2)
        assert block.preset == "lorenz"
        assert block.config_path == "my.toml"
        assert block.status == BlockStatus.ASSIGNED

    def test_unassigned_block_is_empty(self) -> None:
        """Getting an unassigned position returns an empty block."""
        manifest = GridManifest.create(4, 4)
        block = manifest.get_block(0, 0)
        assert block.status == BlockStatus.EMPTY
        assert block.preset is None

    def test_assign_overwrites_previous(self) -> None:
        """Assigning to an already-assigned position overwrites."""
        manifest = GridManifest.create(4, 4)
        manifest.assign(0, 0, "lorenz")
        manifest.assign(0, 0, "torus")
        block = manifest.get_block(0, 0)
        assert block.preset == "torus"

    def test_out_of_bounds_raises(self) -> None:
        """Assigning to an out-of-bounds position raises an error."""
        manifest = GridManifest.create(4, 4)
        with pytest.raises(ValueError, match="out of bounds"):
            manifest.assign(5, 0, "lorenz")

    def test_negative_position_raises(self) -> None:
        """Assigning to a negative position raises an error."""
        manifest = GridManifest.create(4, 4)
        with pytest.raises(ValueError, match="out of bounds"):
            manifest.assign(-1, 0, "lorenz")

    def test_out_of_bounds_get_raises(self) -> None:
        """Getting an out-of-bounds position raises an error."""
        manifest = GridManifest.create(4, 4)
        with pytest.raises(ValueError, match="out of bounds"):
            manifest.get_block(4, 0)


class TestGridSummary:
    """Tests for grid summary counts."""

    def test_summary_all_empty(self) -> None:
        """An empty grid has all blocks counted as empty."""
        manifest = GridManifest.create(3, 3)
        counts = manifest.summary()
        assert counts["empty"] == 9
        assert counts["assigned"] == 0
        assert counts["exported"] == 0

    def test_summary_counts_statuses_correctly(self) -> None:
        """Summary counts statuses correctly, including empties."""
        manifest = GridManifest.create(3, 3)
        manifest.assign(0, 0, "lorenz")
        manifest.assign(0, 1, "torus")
        manifest.set_status(0, 0, BlockStatus.EXPORTED)
        counts = manifest.summary()
        assert counts["empty"] == 7
        assert counts["assigned"] == 1
        assert counts["exported"] == 1

    def test_summary_with_error_status(self) -> None:
        """Summary includes error counts."""
        manifest = GridManifest.create(2, 2)
        manifest.assign(0, 0, "lorenz")
        manifest.set_status(0, 0, BlockStatus.ERROR)
        counts = manifest.summary()
        assert counts["error"] == 1
        assert counts["empty"] == 3


class TestGridNeighbors:
    """Tests for neighbor lookup."""

    def test_corner_returns_3_neighbors(self) -> None:
        """Neighbors at corner position returns only 3 neighbors."""
        manifest = GridManifest.create(4, 4)
        nbrs = manifest.neighbors(0, 0)
        assert len(nbrs) == 3
        positions = {(b.row, b.col) for b in nbrs}
        assert positions == {(0, 1), (1, 0), (1, 1)}

    def test_bottom_right_corner_returns_3(self) -> None:
        """Bottom-right corner also returns 3 neighbors."""
        manifest = GridManifest.create(4, 4)
        nbrs = manifest.neighbors(3, 3)
        assert len(nbrs) == 3
        positions = {(b.row, b.col) for b in nbrs}
        assert positions == {(2, 2), (2, 3), (3, 2)}

    def test_edge_returns_5_neighbors(self) -> None:
        """A position on an edge (not corner) returns 5 neighbors."""
        manifest = GridManifest.create(4, 4)
        nbrs = manifest.neighbors(0, 1)
        assert len(nbrs) == 5

    def test_center_returns_8_neighbors(self) -> None:
        """A center position returns all 8 neighbors."""
        manifest = GridManifest.create(4, 4)
        nbrs = manifest.neighbors(2, 2)
        assert len(nbrs) == 8

    def test_neighbors_out_of_bounds_raises(self) -> None:
        """Neighbors for out-of-bounds position raises."""
        manifest = GridManifest.create(4, 4)
        with pytest.raises(ValueError, match="out of bounds"):
            manifest.neighbors(4, 4)


class TestGridDimensions:
    """Tests for non-square and various grid dimensions."""

    def test_non_square_grid(self) -> None:
        """Grid with non-square dimensions (8x12) works correctly."""
        manifest = GridManifest.create(8, 12)
        assert manifest.rows == 8
        assert manifest.cols == 12
        # Can assign to all corners
        manifest.assign(0, 0, "a")
        manifest.assign(0, 11, "b")
        manifest.assign(7, 0, "c")
        manifest.assign(7, 11, "d")
        assert manifest.get_block(7, 11).preset == "d"
        counts = manifest.summary()
        assert counts["assigned"] == 4
        assert counts["empty"] == 92

    def test_single_cell_grid(self) -> None:
        """A 1x1 grid works."""
        manifest = GridManifest.create(1, 1)
        manifest.assign(0, 0, "lorenz")
        assert manifest.get_block(0, 0).preset == "lorenz"
        nbrs = manifest.neighbors(0, 0)
        assert len(nbrs) == 0

    def test_wide_grid(self) -> None:
        """A 1xN grid has only left/right neighbors."""
        manifest = GridManifest.create(1, 5)
        nbrs = manifest.neighbors(0, 2)
        assert len(nbrs) == 2
        positions = {(b.row, b.col) for b in nbrs}
        assert positions == {(0, 1), (0, 3)}

    def test_zero_dimension_raises(self) -> None:
        """Grid with zero rows or cols raises."""
        with pytest.raises(ValueError, match="must be >= 1"):
            GridManifest.create(0, 4)

    def test_negative_dimension_raises(self) -> None:
        """Grid with negative dimensions raises."""
        with pytest.raises(ValueError, match="must be >= 1"):
            GridManifest.create(-1, 4)


class TestGridPersistence:
    """Tests for saving and loading grid manifests."""

    def test_save_and_load_roundtrip(self, tmp_path: Path) -> None:
        """Saving and loading a manifest preserves data."""
        manifest_path = tmp_path / "grid.toml"
        manifest = GridManifest.create(4, 4, manifest_path)
        manifest.assign(0, 0, "lorenz", config_path="a.toml")
        manifest.assign(1, 2, "torus")
        manifest.set_status(0, 0, BlockStatus.EXPORTED)
        manifest.save()

        loaded = GridManifest.load(manifest_path)
        assert loaded.rows == 4
        assert loaded.cols == 4
        b00 = loaded.get_block(0, 0)
        assert b00.preset == "lorenz"
        assert b00.config_path == "a.toml"
        assert b00.status == BlockStatus.EXPORTED
        b12 = loaded.get_block(1, 2)
        assert b12.preset == "torus"
        assert b12.status == BlockStatus.ASSIGNED

    def test_status_transition_updates_disk(self, tmp_path: Path) -> None:
        """Status transitions update the manifest file on disk."""
        manifest_path = tmp_path / "grid.toml"
        manifest = GridManifest.create(2, 2, manifest_path)
        manifest.assign(0, 0, "lorenz")
        manifest.save()

        # Transition to exported
        manifest.set_status(0, 0, BlockStatus.EXPORTED)
        manifest.save()

        loaded = GridManifest.load(manifest_path)
        assert loaded.get_block(0, 0).status == BlockStatus.EXPORTED

        # Transition to error
        loaded.set_status(0, 0, BlockStatus.ERROR)
        loaded.save()

        reloaded = GridManifest.load(manifest_path)
        assert reloaded.get_block(0, 0).status == BlockStatus.ERROR

    def test_load_nonexistent_raises(self, tmp_path: Path) -> None:
        """Loading a nonexistent manifest raises FileNotFoundError."""
        with pytest.raises(FileNotFoundError):
            GridManifest.load(tmp_path / "nope.toml")

    def test_empty_grid_save_load(self, tmp_path: Path) -> None:
        """An empty grid saves and loads correctly."""
        manifest_path = tmp_path / "grid.toml"
        manifest = GridManifest.create(3, 5, manifest_path)
        manifest.save()

        loaded = GridManifest.load(manifest_path)
        assert loaded.rows == 3
        assert loaded.cols == 5
        assert len(loaded.blocks) == 0


class TestGridToDict:
    """Tests for JSON serialization."""

    def test_to_dict_structure(self) -> None:
        """to_dict returns expected structure."""
        manifest = GridManifest.create(2, 2)
        manifest.assign(0, 0, "lorenz")
        d = manifest.to_dict()
        assert d["rows"] == 2
        assert d["cols"] == 2
        assert "0,0" in d["blocks"]
        assert d["blocks"]["0,0"]["preset"] == "lorenz"
        assert d["blocks"]["0,0"]["status"] == "assigned"
