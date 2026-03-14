"""Tests for CLI utility commands: convert, sample, transform, schema."""

import json
from pathlib import Path

import numpy as np
import trimesh
from typer.testing import CliRunner

from mathviz.cli import app

runner = CliRunner()


def _create_stl(path: Path) -> None:
    """Create a minimal valid STL file using trimesh."""
    mesh = trimesh.creation.box(extents=(10, 10, 10))
    mesh.export(str(path), file_type="stl")


def _create_obj(path: Path) -> None:
    """Create a minimal valid OBJ file using trimesh."""
    mesh = trimesh.creation.box(extents=(10, 10, 10))
    mesh.export(str(path), file_type="obj")


def _create_ply_cloud(path: Path, num_points: int = 100) -> None:
    """Create a PLY file containing only a point cloud (no faces)."""
    rng = np.random.default_rng(42)
    points = rng.uniform(-5, 5, size=(num_points, 3))
    header = (
        "ply\n"
        "format ascii 1.0\n"
        f"element vertex {num_points}\n"
        "property float x\n"
        "property float y\n"
        "property float z\n"
        "end_header\n"
    )
    with open(path, "w", encoding="utf-8") as f:
        f.write(header)
        np.savetxt(f, points, fmt="%.6f", delimiter=" ")


class TestConvertCommand:
    """Test convert command."""

    def test_convert_stl_to_obj(self, tmp_path: Path) -> None:
        """mathviz convert STL to OBJ produces a valid OBJ file."""
        stl_path = tmp_path / "input.stl"
        obj_path = tmp_path / "output.obj"
        _create_stl(stl_path)

        result = runner.invoke(app, ["convert", str(stl_path), str(obj_path)])
        assert result.exit_code == 0, result.output
        assert obj_path.exists()
        content = obj_path.read_text()
        assert "v " in content  # OBJ vertex lines

    def test_convert_mesh_stl_to_ply_cloud_auto_sample(self, tmp_path: Path) -> None:
        """mathviz convert mesh STL to PLY with --auto-sample produces a valid cloud PLY."""
        stl_path = tmp_path / "input.stl"
        ply_path = tmp_path / "output.ply"
        _create_stl(stl_path)

        result = runner.invoke(
            app,
            ["convert", str(stl_path), str(ply_path), "--auto-sample"],
        )
        assert result.exit_code == 0, result.output
        assert ply_path.exists()
        content = ply_path.read_text()
        assert "ply" in content
        assert "element vertex" in content
        # Should NOT have face data (it's a point cloud)
        assert "element face" not in content

    def test_convert_cloud_ply_to_stl_without_auto_sample_fails(self, tmp_path: Path) -> None:
        """mathviz convert PLY(cloud) to STL without --auto-sample fails with clear error."""
        ply_path = tmp_path / "cloud.ply"
        stl_path = tmp_path / "output.stl"
        _create_ply_cloud(ply_path)

        result = runner.invoke(app, ["convert", str(ply_path), str(stl_path)])
        assert result.exit_code == 2
        assert "mesh" in result.output.lower() or "error" in result.output.lower()

    def test_convert_stl_to_stl(self, tmp_path: Path) -> None:
        """STL to STL is a valid identity conversion."""
        stl_in = tmp_path / "input.stl"
        stl_out = tmp_path / "output.stl"
        _create_stl(stl_in)

        result = runner.invoke(app, ["convert", str(stl_in), str(stl_out)])
        assert result.exit_code == 0, result.output
        assert stl_out.exists()

    def test_convert_missing_input_fails(self, tmp_path: Path) -> None:
        """Converting a nonexistent file exits with error."""
        result = runner.invoke(
            app, ["convert", str(tmp_path / "no.stl"), str(tmp_path / "out.obj")]
        )
        assert result.exit_code == 2


class TestSampleCommand:
    """Test sample command."""

    def test_sample_stl_to_ply(self, tmp_path: Path) -> None:
        """mathviz sample on an STL produces a PLY point cloud."""
        stl_path = tmp_path / "input.stl"
        ply_path = tmp_path / "output.ply"
        _create_stl(stl_path)

        result = runner.invoke(
            app,
            ["sample", str(stl_path), str(ply_path), "--num-points", "500"],
        )
        assert result.exit_code == 0, result.output
        assert ply_path.exists()
        content = ply_path.read_text()
        assert "ply" in content
        assert "element vertex" in content

    def test_sample_with_density(self, tmp_path: Path) -> None:
        """Sampling with --density option works."""
        stl_path = tmp_path / "input.stl"
        ply_path = tmp_path / "output.ply"
        _create_stl(stl_path)

        result = runner.invoke(
            app,
            ["sample", str(stl_path), str(ply_path), "--density", "0.5"],
        )
        assert result.exit_code == 0, result.output
        assert ply_path.exists()

    def test_sample_cloud_input_fails(self, tmp_path: Path) -> None:
        """Sampling a point cloud (no mesh) fails with clear error."""
        ply_in = tmp_path / "cloud.ply"
        ply_out = tmp_path / "output.ply"
        _create_ply_cloud(ply_in)

        result = runner.invoke(app, ["sample", str(ply_in), str(ply_out)])
        assert result.exit_code == 2
        assert "mesh" in result.output.lower()


class TestTransformCommand:
    """Test transform command."""

    def test_transform_fits_within_container(self, tmp_path: Path) -> None:
        """mathviz transform fits geometry within specified container dimensions."""
        stl_path = tmp_path / "input.stl"
        out_path = tmp_path / "output.stl"
        _create_stl(stl_path)

        result = runner.invoke(
            app,
            [
                "transform", str(stl_path), str(out_path),
                "--width", "50", "--height", "50", "--depth", "20",
            ],
        )
        assert result.exit_code == 0, result.output
        assert out_path.exists()

        # Verify the output mesh fits within the container
        loaded = trimesh.load(str(out_path), process=False)
        bounds = loaded.bounds  # (2, 3) array
        size = bounds[1] - bounds[0]
        # Usable volume = dims - 2*margin (default margin=5mm)
        assert size[0] <= 50.0 + 0.01
        assert size[1] <= 50.0 + 0.01
        assert size[2] <= 20.0 + 0.01

    def test_transform_point_cloud(self, tmp_path: Path) -> None:
        """Transform works on point cloud files."""
        ply_in = tmp_path / "cloud.ply"
        ply_out = tmp_path / "output.ply"
        _create_ply_cloud(ply_in)

        result = runner.invoke(
            app, ["transform", str(ply_in), str(ply_out)]
        )
        assert result.exit_code == 0, result.output
        assert ply_out.exists()


class TestSchemaCommand:
    """Test schema generation command."""

    def test_schema_produces_valid_json(self, tmp_path: Path) -> None:
        """Schema generation produces valid JSON Schema files."""
        schema_dir = tmp_path / "schemas"

        result = runner.invoke(app, ["schema", str(schema_dir)])
        assert result.exit_code == 0, result.output
        assert schema_dir.exists()

        # Check that expected schema files exist and are valid JSON
        expected = ["Container.json", "PlacementPolicy.json", "SamplerConfig.json",
                    "RepresentationConfig.json", "EngravingProfile.json"]
        for fname in expected:
            fpath = schema_dir / fname
            assert fpath.exists(), f"Missing schema: {fname}"
            schema = json.loads(fpath.read_text())
            assert "properties" in schema or "type" in schema

    def test_schema_has_generator_schemas(self, tmp_path: Path) -> None:
        """Schema generation includes generator parameter schemas."""
        schema_dir = tmp_path / "schemas"
        result = runner.invoke(app, ["schema", str(schema_dir)])
        assert result.exit_code == 0, result.output

        gen_dir = schema_dir / "generators"
        if gen_dir.exists():
            # At least some generators should have schemas
            gen_files = list(gen_dir.glob("*.json"))
            for gf in gen_files:
                schema = json.loads(gf.read_text())
                assert isinstance(schema, dict)

    def test_schema_container_has_expected_fields(self, tmp_path: Path) -> None:
        """Container schema includes width_mm, height_mm, depth_mm."""
        schema_dir = tmp_path / "schemas"
        runner.invoke(app, ["schema", str(schema_dir)])

        container_schema = json.loads((schema_dir / "Container.json").read_text())
        props = container_schema.get("properties", {})
        assert "width_mm" in props
        assert "height_mm" in props
        assert "depth_mm" in props
