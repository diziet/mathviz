"""Tests for export routing auto-detection (Task 38)."""

from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest
from fastapi.testclient import TestClient

from mathviz.core.math_object import Curve, MathObject, Mesh, PointCloud
from mathviz.pipeline.mesh_exporter import MeshExportError
from mathviz.pipeline.point_cloud_exporter import PointCloudExportError
from mathviz.pipeline.runner import ExportConfig, _detect_export_type, _run_export


# --- Helpers ---


def _get_executor() -> "GenerationExecutor":  # noqa: F821
    """Return the module-level executor used by the preview server."""
    from mathviz.preview.server import get_executor
    return get_executor()


def _cube_mesh() -> Mesh:
    """Create a minimal cube mesh."""
    vertices = np.array([
        [-1, -1, -1], [1, -1, -1], [1, 1, -1], [-1, 1, -1],
        [-1, -1, 1], [1, -1, 1], [1, 1, 1], [-1, 1, 1],
    ], dtype=np.float64)
    faces = np.array([
        [0, 1, 2], [0, 2, 3], [4, 5, 6], [4, 6, 7],
        [0, 1, 5], [0, 5, 4], [2, 3, 7], [2, 7, 6],
        [0, 3, 7], [0, 7, 4], [1, 2, 6], [1, 6, 5],
    ], dtype=np.int64)
    return Mesh(vertices=vertices, faces=faces)


def _small_cloud() -> PointCloud:
    """Create a minimal point cloud."""
    points = np.array([
        [0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1], [1, 1, 1],
    ], dtype=np.float64)
    return PointCloud(points=points)


def _mesh_only_obj() -> MathObject:
    """MathObject with mesh only."""
    return MathObject(mesh=_cube_mesh(), generator_name="test_mesh")


def _cloud_only_obj() -> MathObject:
    """MathObject with point cloud only."""
    return MathObject(point_cloud=_small_cloud(), generator_name="test_cloud")


def _both_obj() -> MathObject:
    """MathObject with both mesh and point cloud."""
    return MathObject(
        mesh=_cube_mesh(),
        point_cloud=_small_cloud(),
        generator_name="test_both",
    )


# --- ExportConfig validation ---


class TestExportConfigValidation:
    """ExportConfig accepts 'auto', 'mesh', 'point_cloud' and rejects others."""

    def test_auto_default(self) -> None:
        """Default export_type is 'auto'."""
        config = ExportConfig(path=Path("out.ply"))
        assert config.export_type == "auto"

    def test_explicit_mesh(self) -> None:
        """Explicit 'mesh' is accepted."""
        config = ExportConfig(path=Path("out.stl"), export_type="mesh")
        assert config.export_type == "mesh"

    def test_explicit_point_cloud(self) -> None:
        """Explicit 'point_cloud' is accepted."""
        config = ExportConfig(path=Path("out.xyz"), export_type="point_cloud")
        assert config.export_type == "point_cloud"

    def test_invalid_export_type_raises(self) -> None:
        """Invalid export_type raises ValueError."""
        with pytest.raises(ValueError, match="export_type must be one of"):
            ExportConfig(path=Path("out.stl"), export_type="invalid")


# --- Auto-detection logic ---


class TestDetectExportType:
    """_detect_export_type picks the right exporter based on geometry and extension."""

    def test_mesh_only_returns_mesh(self) -> None:
        """Mesh-only object always routes to mesh."""
        assert _detect_export_type(_mesh_only_obj(), Path("out.ply")) == "mesh"

    def test_cloud_only_returns_cloud(self) -> None:
        """Cloud-only object always routes to point_cloud."""
        assert _detect_export_type(_cloud_only_obj(), Path("out.ply")) == "point_cloud"

    def test_both_stl_returns_mesh(self) -> None:
        """Both geometries + .stl extension routes to mesh."""
        assert _detect_export_type(_both_obj(), Path("out.stl")) == "mesh"

    def test_both_obj_returns_mesh(self) -> None:
        """Both geometries + .obj extension routes to mesh."""
        assert _detect_export_type(_both_obj(), Path("out.obj")) == "mesh"

    def test_both_xyz_returns_cloud(self) -> None:
        """Both geometries + .xyz extension routes to point_cloud."""
        assert _detect_export_type(_both_obj(), Path("out.xyz")) == "point_cloud"

    def test_both_pcd_returns_cloud(self) -> None:
        """Both geometries + .pcd extension routes to point_cloud."""
        assert _detect_export_type(_both_obj(), Path("out.pcd")) == "point_cloud"

    def test_both_ply_prefers_mesh(self) -> None:
        """Both geometries + .ply (ambiguous) prefers mesh."""
        assert _detect_export_type(_both_obj(), Path("out.ply")) == "mesh"

    def test_no_geometry_raises_value_error(self) -> None:
        """Curve-only object with no mesh or cloud raises ValueError."""
        curve_only = MathObject(
            curves=[Curve(points=np.array([[0, 0, 0], [1, 1, 1]], dtype=np.float64))],
            generator_name="test_curves",
        )
        with pytest.raises(ValueError, match="has no mesh or point_cloud to export"):
            _detect_export_type(curve_only, Path("out.ply"))


# --- End-to-end export routing ---


class TestAutoExportMeshOnly:
    """ExportConfig(export_type='auto') with mesh-only MathObject routes to mesh."""

    def test_auto_mesh_only_stl(self, tmp_path: Path) -> None:
        """Auto-detect exports mesh-only object to STL successfully."""
        config = ExportConfig(path=tmp_path / "out.stl")
        result = _run_export(_mesh_only_obj(), config)
        assert result.exists()
        assert result.stat().st_size > 0

    def test_auto_mesh_only_ply(self, tmp_path: Path) -> None:
        """Auto-detect exports mesh-only object to PLY successfully."""
        config = ExportConfig(path=tmp_path / "out.ply")
        result = _run_export(_mesh_only_obj(), config)
        assert result.exists()


class TestAutoExportCloudOnly:
    """ExportConfig(export_type='auto') with cloud-only MathObject routes to cloud."""

    def test_auto_cloud_only_ply(self, tmp_path: Path) -> None:
        """Auto-detect exports cloud-only object to PLY successfully."""
        config = ExportConfig(path=tmp_path / "out.ply")
        result = _run_export(_cloud_only_obj(), config)
        assert result.exists()

    def test_auto_cloud_only_xyz(self, tmp_path: Path) -> None:
        """Auto-detect exports cloud-only object to XYZ successfully."""
        config = ExportConfig(path=tmp_path / "out.xyz")
        result = _run_export(_cloud_only_obj(), config)
        assert result.exists()
        assert result.stat().st_size > 0


class TestExplicitOverride:
    """Explicit export_type='mesh' or 'point_cloud' overrides auto-detection."""

    def test_explicit_mesh_forces_mesh(self, tmp_path: Path) -> None:
        """export_type='mesh' forces mesh export even with cloud present."""
        obj = _both_obj()
        config = ExportConfig(path=tmp_path / "out.stl", export_type="mesh")
        result = _run_export(obj, config)
        assert result.exists()

    def test_explicit_cloud_forces_cloud(self, tmp_path: Path) -> None:
        """export_type='point_cloud' forces cloud export even with mesh present."""
        obj = _both_obj()
        config = ExportConfig(path=tmp_path / "out.xyz", export_type="point_cloud")
        result = _run_export(obj, config)
        assert result.exists()

    def test_explicit_mesh_no_mesh_raises(self, tmp_path: Path) -> None:
        """export_type='mesh' with cloud-only object raises MeshExportError."""
        config = ExportConfig(path=tmp_path / "out.stl", export_type="mesh")
        with pytest.raises(MeshExportError):
            _run_export(_cloud_only_obj(), config)

    def test_explicit_cloud_no_cloud_raises(self, tmp_path: Path) -> None:
        """export_type='point_cloud' with mesh-only object raises PointCloudExportError."""
        config = ExportConfig(path=tmp_path / "out.ply", export_type="point_cloud")
        with pytest.raises(PointCloudExportError):
            _run_export(_mesh_only_obj(), config)


# --- Preview server tests ---


class TestPreviewServerGeometryRouting:
    """Preview server returns 200 for mesh-only and cloud-only generators."""

    @pytest.fixture(autouse=True)
    def _setup_generators(self) -> None:
        """Ensure generators are registered."""
        import mathviz.core.generator as gen_mod
        from mathviz.generators.parametric.torus import TorusGenerator

        if "torus" not in gen_mod._alias_map:
            gen_mod._discovered = True
            gen_mod.register(TorusGenerator)

    @pytest.fixture(autouse=True)
    def _clean_cache(self) -> None:
        """Reset preview cache between tests."""
        from mathviz.preview.server import reset_cache
        reset_cache()

    @pytest.fixture
    def client(self) -> TestClient:
        """Return a FastAPI test client."""
        from mathviz.preview.server import app
        return TestClient(app)

    def test_torus_mesh_only_returns_200(self, client: TestClient) -> None:
        """POST /api/generate for torus (mesh via SURFACE_SHELL) returns 200."""
        resp = client.post(
            "/api/generate",
            json={
                "generator": "torus",
                "seed": 42,
                "resolution": {"grid_resolution": 16},
            },
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["mesh_url"] is not None

    def test_torus_mesh_url_serves_glb(self, client: TestClient) -> None:
        """Mesh URL for torus returns valid GLB binary."""
        resp = client.post(
            "/api/generate",
            json={
                "generator": "torus",
                "seed": 42,
                "resolution": {"grid_resolution": 16},
            },
        )
        mesh_url = resp.json()["mesh_url"]
        mesh_resp = client.get(f"{mesh_url}?lod=preview")
        assert mesh_resp.status_code == 200
        assert mesh_resp.content[:4] == b"glTF"

    def test_cloud_only_generator_returns_200(self, client: TestClient) -> None:
        """POST /api/generate for a cloud-only generator returns 200 with cloud_url."""
        from mathviz.core.validator import ValidationResult
        from mathviz.pipeline.runner import PipelineResult

        cloud_obj = _cloud_only_obj()
        mock_result = PipelineResult(
            math_object=cloud_obj,
            validation=ValidationResult(),
        )
        with patch.object(
            _get_executor(), "submit", return_value=mock_result
        ):
            resp = client.post(
                "/api/generate",
                json={"generator": "torus", "seed": 99},
            )
        assert resp.status_code == 200
        data = resp.json()
        assert data["cloud_url"] is not None
        assert data["mesh_url"] is None

    def test_cloud_only_cloud_url_serves_ply(self, client: TestClient) -> None:
        """Cloud URL for cloud-only generator returns valid PLY data."""
        from mathviz.core.validator import ValidationResult
        from mathviz.pipeline.runner import PipelineResult

        cloud_obj = _cloud_only_obj()
        mock_result = PipelineResult(
            math_object=cloud_obj,
            validation=ValidationResult(),
        )
        with patch.object(
            _get_executor(), "submit", return_value=mock_result
        ):
            resp = client.post(
                "/api/generate",
                json={"generator": "torus", "seed": 99},
            )
        cloud_url = resp.json()["cloud_url"]
        cloud_resp = client.get(f"{cloud_url}?lod=preview")
        assert cloud_resp.status_code == 200
