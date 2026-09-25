"""Tests that CLI commands apply the [representation] config section."""

import json
from pathlib import Path

import pytest
from typer.testing import CliRunner

from mathviz.cli import app
from mathviz.core.container import Container, PlacementPolicy
from mathviz.core.representation import RepresentationConfig, RepresentationType
from mathviz.pipeline.geometry_loader import has_mesh, load_geometry
from mathviz.pipeline.runner import run

runner = CliRunner()

# trefoil is a curve generator whose default representation is a 16-sided tube.
CURVE_GENERATOR = "trefoil"

TUBE_5_SIDES_TOML = '[representation]\ntype = "tube"\ntube_radius = 0.15\ntube_sides = 5\n'
RAW_POINT_CLOUD_TOML = '[representation]\ntype = "raw_point_cloud"\n'
INVALID_TUBE_SIDES_TOML = '[representation]\ntype = "tube"\ntube_sides = 0\n'
TUBE_5_SIDES = RepresentationConfig(type=RepresentationType.TUBE, tube_radius=0.15, tube_sides=5)


@pytest.fixture(autouse=True)
def isolated_cwd(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Run each test in an empty directory so no stray mathviz.toml is discovered."""
    monkeypatch.chdir(tmp_path)
    return tmp_path


def _write(path: Path, text: str) -> Path:
    """Write text to path and return the path."""
    path.write_text(text)
    return path


def _generate_json(*args: str) -> dict:
    """Run `mathviz generate trefoil --json` with extra args and return the parsed output."""
    result = runner.invoke(app, ["generate", CURVE_GENERATOR, "--json", *args])
    assert result.exit_code in (0, 1), result.output
    return json.loads(result.output)


def _tube_vertex_count(representation_config: RepresentationConfig | None) -> int:
    """Return the mesh vertex count that runner.run produces for trefoil."""
    result = run(
        CURVE_GENERATOR,
        container=Container(),
        placement=PlacementPolicy(),
        representation_config=representation_config,
    )
    assert result.math_object.mesh is not None
    return len(result.math_object.mesh.vertices)


class TestGenerateRepresentationConfig:
    """Test that generate applies [representation] from project and per-object configs."""

    def test_config_tube_sides_sets_mesh_vertex_count(self, tmp_path: Path) -> None:
        """--config with tube_sides = 5 gives the same mesh as runner.run with that config."""
        config = _write(tmp_path / "tube.toml", TUBE_5_SIDES_TOML)
        data = _generate_json("--config", str(config))
        assert data["mesh_vertices"] == _tube_vertex_count(TUBE_5_SIDES)

    def test_config_raw_point_cloud_replaces_default_tube(self, tmp_path: Path) -> None:
        """--config with type = raw_point_cloud gives a point cloud and no mesh."""
        config = _write(tmp_path / "raw.toml", RAW_POINT_CLOUD_TOML)
        data = _generate_json("--config", str(config))
        assert data["mesh_vertices"] is None
        assert data["point_count"] is not None and data["point_count"] > 0

    def test_project_config_representation_applies(self, isolated_cwd: Path) -> None:
        """A [representation] section in mathviz.toml in the working directory applies."""
        _write(isolated_cwd / "mathviz.toml", RAW_POINT_CLOUD_TOML)
        data = _generate_json()
        assert data["mesh_vertices"] is None
        assert data["point_count"] is not None and data["point_count"] > 0

    def test_object_config_representation_overrides_project(
        self, isolated_cwd: Path
    ) -> None:
        """--config [representation] overrides the project [representation]."""
        _write(isolated_cwd / "mathviz.toml", RAW_POINT_CLOUD_TOML)
        config = _write(isolated_cwd / "tube.toml", TUBE_5_SIDES_TOML)
        data = _generate_json("--config", str(config))
        assert data["point_count"] is None
        assert data["mesh_vertices"] == _tube_vertex_count(TUBE_5_SIDES)

    def test_config_without_representation_keeps_generator_default(
        self, tmp_path: Path
    ) -> None:
        """--config without [representation] gives the generator's default tube."""
        config = _write(tmp_path / "seed.toml", "seed = 42\n")
        data = _generate_json("--config", str(config))
        assert data["mesh_vertices"] == _tube_vertex_count(None)

    def test_invalid_representation_exits_2_with_json_error(self, tmp_path: Path) -> None:
        """An invalid [representation] exits 2 with a JSON error that names the field."""
        config = _write(tmp_path / "bad.toml", INVALID_TUBE_SIDES_TOML)
        result = runner.invoke(
            app, ["generate", CURVE_GENERATOR, "--json", "--config", str(config)]
        )
        assert result.exit_code == 2
        error = json.loads(result.output)["error"]
        assert "Invalid representation config: tube_sides" in error

    def test_invalid_representation_rich_output_names_field(self, tmp_path: Path) -> None:
        """An invalid [representation] without --json exits 2 and prints the full error."""
        config = _write(tmp_path / "bad.toml", INVALID_TUBE_SIDES_TOML)
        result = runner.invoke(app, ["generate", CURVE_GENERATOR, "--config", str(config)])
        assert result.exit_code == 2
        assert "Invalid representation config: tube_sides" in result.output


class TestValidateRepresentationConfig:
    """Test that validate applies [representation] from a per-object config."""

    def test_config_raw_point_cloud_skips_mesh_checks(self, tmp_path: Path) -> None:
        """validate --config with raw_point_cloud runs no mesh checks, since there is no mesh."""
        config = _write(tmp_path / "raw.toml", RAW_POINT_CLOUD_TOML)
        result = runner.invoke(
            app, ["validate", CURVE_GENERATOR, "--json", "--config", str(config)]
        )
        assert result.exit_code == 0, result.output
        assert json.loads(result.output)["checks"] == []

    def test_invalid_representation_exits_2(self, tmp_path: Path) -> None:
        """validate --config with an invalid [representation] exits 2 with a JSON error."""
        config = _write(tmp_path / "bad.toml", INVALID_TUBE_SIDES_TOML)
        result = runner.invoke(
            app, ["validate", CURVE_GENERATOR, "--json", "--config", str(config)]
        )
        assert result.exit_code == 2
        assert "tube_sides" in json.loads(result.output)["error"]


def _export_single_block(tmp_path: Path, config_text: str) -> dict:
    """Assign trefoil with a per-block config to a 1x1 grid, run export-all, return its result."""
    manifest = tmp_path / "grid.toml"
    config = _write(tmp_path / "block.toml", config_text)
    for args in (
        ["grid", "init", "1", "1", "--path", str(manifest)],
        ["grid", "assign", "0", "0", CURVE_GENERATOR, "--config", str(config),
         "--path", str(manifest)],
    ):
        assert runner.invoke(app, args).exit_code == 0
    result = runner.invoke(
        app,
        ["grid", "export-all", "--path", str(manifest),
         "--output-dir", str(tmp_path / "export"), "--json"],
    )
    assert result.exit_code == 0, result.output
    return json.loads(result.output)["results"][0]


class TestGridExportRepresentationConfig:
    """Test that grid export-all applies [representation] from a per-block config."""

    def test_block_config_raw_point_cloud_exports_point_cloud(self, tmp_path: Path) -> None:
        """A block config with raw_point_cloud exports a point cloud instead of a tube mesh."""
        block = _export_single_block(tmp_path, RAW_POINT_CLOUD_TOML)
        assert block["success"] is True, block
        exported = load_geometry(Path(block["path"]))
        assert not has_mesh(exported)
        assert exported.point_cloud is not None

    def test_invalid_block_representation_marks_block_error(self, tmp_path: Path) -> None:
        """A block config with an invalid [representation] fails with an error naming the field."""
        block = _export_single_block(tmp_path, INVALID_TUBE_SIDES_TOML)
        assert block["success"] is False
        assert "tube_sides" in block["error"]
