"""Tests for the exporter pipeline stages."""

import json
import sys
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest
import trimesh

from mathviz.core.math_object import MathObject, Mesh, PointCloud
from mathviz.pipeline.mesh_exporter import MeshExportError, export_mesh
from mathviz.pipeline.point_cloud_exporter import PointCloudExportError, export_point_cloud


def _make_cube_mesh() -> Mesh:
    """Create a simple cube mesh for testing."""
    vertices = np.array([
        [-1, -1, -1], [1, -1, -1], [1, 1, -1], [-1, 1, -1],
        [-1, -1, 1], [1, -1, 1], [1, 1, 1], [-1, 1, 1],
    ], dtype=np.float64)
    faces = np.array([
        [0, 1, 2], [0, 2, 3],
        [4, 5, 6], [4, 6, 7],
        [0, 1, 5], [0, 5, 4],
        [2, 3, 7], [2, 7, 6],
        [0, 3, 7], [0, 7, 4],
        [1, 2, 6], [1, 6, 5],
    ], dtype=np.int64)
    return Mesh(vertices=vertices, faces=faces)


def _make_mesh_obj() -> MathObject:
    """Create a MathObject with mesh geometry."""
    return MathObject(
        mesh=_make_cube_mesh(),
        generator_name="test_cube",
        parameters={"size": 2.0},
        seed=123,
    )


def _make_cloud_obj() -> MathObject:
    """Create a MathObject with point cloud geometry."""
    points = np.array([
        [0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1], [1, 1, 1],
    ], dtype=np.float64)
    return MathObject(
        point_cloud=PointCloud(points=points),
        generator_name="test_cloud",
        parameters={"count": 5},
        seed=456,
    )


class TestMeshExportNoMesh:
    """STL export of a MathObject with no mesh raises a clear error."""

    def test_stl_export_no_mesh_raises(self, tmp_path: Path) -> None:
        """Exporting STL from a MathObject without mesh raises MeshExportError."""
        obj = MathObject(
            point_cloud=PointCloud(
                points=np.array([[0, 0, 0]], dtype=np.float64)
            ),
            generator_name="no_mesh",
        )
        with pytest.raises(MeshExportError, match="has no mesh geometry"):
            export_mesh(obj, tmp_path / "out.stl")


class TestPointCloudExportNoCloud:
    """PLY cloud export of a MathObject with no point_cloud raises a clear error."""

    def test_ply_cloud_export_no_cloud_raises(self, tmp_path: Path) -> None:
        """Exporting PLY cloud from a MathObject without point_cloud raises error."""
        obj = MathObject(
            mesh=_make_cube_mesh(),
            generator_name="no_cloud",
        )
        with pytest.raises(PointCloudExportError, match="has no point_cloud"):
            export_point_cloud(obj, tmp_path / "out.ply")


class TestStlRoundTrip:
    """STL round-trip: export then reimport via trimesh, vertex count matches."""

    def test_stl_round_trip_vertex_count(self, tmp_path: Path) -> None:
        """Export STL, reimport with trimesh, verify vertex count matches."""
        obj = _make_mesh_obj()
        out_path = tmp_path / "cube.stl"
        export_mesh(obj, out_path)

        loaded = trimesh.load(str(out_path), file_type="stl")
        # STL may split shared vertices; compare face count instead for exact match
        assert loaded.faces.shape[0] == obj.mesh.faces.shape[0]
        # Vertex count should be >= original (STL duplicates per-face)
        assert loaded.vertices.shape[0] >= obj.mesh.vertices.shape[0]


class TestSidecarAutoWrite:
    """Sidecar .meta.json is auto-written alongside every geometry export."""

    def test_mesh_export_writes_sidecar(self, tmp_path: Path) -> None:
        """Mesh export auto-writes a .meta.json sidecar file."""
        obj = _make_mesh_obj()
        out_path = tmp_path / "cube.stl"
        export_mesh(obj, out_path)

        meta_path = out_path.with_suffix(".stl.meta.json")
        assert meta_path.exists(), f"Expected sidecar at {meta_path}"

    def test_cloud_export_writes_sidecar(self, tmp_path: Path) -> None:
        """Point cloud export auto-writes a .meta.json sidecar file."""
        obj = _make_cloud_obj()
        out_path = tmp_path / "cloud.xyz"
        export_point_cloud(obj, out_path)

        meta_path = out_path.with_suffix(".xyz.meta.json")
        assert meta_path.exists(), f"Expected sidecar at {meta_path}"

    def test_obj_export_writes_sidecar(self, tmp_path: Path) -> None:
        """OBJ mesh export auto-writes sidecar."""
        obj = _make_mesh_obj()
        out_path = tmp_path / "cube.obj"
        export_mesh(obj, out_path)

        meta_path = out_path.with_suffix(".obj.meta.json")
        assert meta_path.exists()


class TestSidecarContents:
    """Sidecar contains generator_name, params, seed, and timestamp."""

    def test_sidecar_required_fields(self, tmp_path: Path) -> None:
        """Sidecar JSON contains all required reproducibility fields."""
        obj = _make_mesh_obj()
        out_path = tmp_path / "cube.stl"
        export_mesh(obj, out_path)

        meta_path = out_path.with_suffix(".stl.meta.json")
        meta = json.loads(meta_path.read_text(encoding="utf-8"))

        assert meta["generator_name"] == "test_cube"
        assert meta["params"] == {"size": 2.0}
        assert meta["seed"] == 123
        assert "export_timestamp" in meta
        assert "version" in meta

    def test_cloud_sidecar_fields(self, tmp_path: Path) -> None:
        """Point cloud sidecar also contains required fields."""
        obj = _make_cloud_obj()
        out_path = tmp_path / "cloud.ply"
        export_point_cloud(obj, out_path)

        meta_path = out_path.with_suffix(".ply.meta.json")
        meta = json.loads(meta_path.read_text(encoding="utf-8"))

        assert meta["generator_name"] == "test_cloud"
        assert meta["params"] == {"count": 5}
        assert meta["seed"] == 456
        assert "export_timestamp" in meta


class TestPcdImportError:
    """PCD export without open3d installed raises ImportError with install instructions."""

    def test_pcd_raises_import_error(self, tmp_path: Path) -> None:
        """PCD export raises ImportError directing user to install open3d."""
        obj = _make_cloud_obj()
        with patch.dict(sys.modules, {"open3d": None}):
            with pytest.raises(ImportError, match="pip install mathviz\\[open3d\\]"):
                export_point_cloud(obj, tmp_path / "cloud.pcd")


class TestPlyCloudRoundTrip:
    """PLY cloud export produces valid point data."""

    def test_ply_cloud_point_count(self, tmp_path: Path) -> None:
        """PLY cloud export contains correct number of points."""
        obj = _make_cloud_obj()
        out_path = tmp_path / "cloud.ply"
        export_point_cloud(obj, out_path)

        content = out_path.read_text(encoding="utf-8")
        assert "element vertex 5" in content
        # Count data lines after header
        lines = content.strip().split("\n")
        header_end = lines.index("end_header")
        data_lines = lines[header_end + 1:]
        assert len(data_lines) == 5


class TestXyzExport:
    """XYZ export produces valid point data."""

    def test_xyz_point_count(self, tmp_path: Path) -> None:
        """XYZ export produces correct number of lines."""
        obj = _make_cloud_obj()
        out_path = tmp_path / "cloud.xyz"
        export_point_cloud(obj, out_path)

        lines = out_path.read_text(encoding="utf-8").strip().split("\n")
        assert len(lines) == 5
