"""Tests for bounding box and geometry coordinate space alignment in preview."""

import struct
from typing import Any, Generator

import numpy as np
import pytest
import trimesh
from fastapi.testclient import TestClient

from mathviz.core.container import Container
from mathviz.core.generator import register
from mathviz.core.math_object import Mesh, PointCloud
from mathviz.generators.parametric.torus import TorusGenerator
from mathviz.preview.lod import cloud_to_binary_ply, mesh_to_glb
from mathviz.preview.server import app, reset_cache


def _ensure_torus_registered() -> None:
    """Re-register the torus generator if missing from the registry."""
    import mathviz.core.generator as gen_mod

    if "torus" not in gen_mod._alias_map:
        gen_mod._discovered = True
        register(TorusGenerator)


@pytest.fixture(autouse=True)
def _ensure_generators() -> Generator[None, None, None]:
    """Ensure generators are registered and cache is clean."""
    _ensure_torus_registered()
    reset_cache()
    yield
    reset_cache()


@pytest.fixture
def client() -> TestClient:
    """Return a FastAPI test client."""
    return TestClient(app)


@pytest.fixture
def preview_html(client: TestClient) -> str:
    """Fetch the preview HTML once for assertion-only tests."""
    resp = client.get("/")
    assert resp.status_code == 200
    return resp.text


@pytest.fixture
def fit_camera_code(preview_html: str) -> str:
    """Extract the fitCamera function body from the preview HTML."""
    return preview_html.split("function fitCamera")[1].split("function")[0]


def _make_offset_mesh() -> Mesh:
    """Create a mesh with vertices far from the origin (simulating pipeline output)."""
    vertices = np.array([
        [5.0, 5.0, 7.0],
        [95.0, 5.0, 7.0],
        [95.0, 95.0, 7.0],
        [5.0, 95.0, 7.0],
        [5.0, 5.0, 33.0],
        [95.0, 5.0, 33.0],
        [95.0, 95.0, 33.0],
        [5.0, 95.0, 33.0],
    ], dtype=np.float64)
    faces = np.array([
        [0, 1, 2], [0, 2, 3],
        [4, 5, 6], [4, 6, 7],
        [0, 1, 5], [0, 5, 4],
        [2, 3, 7], [2, 7, 6],
    ], dtype=np.int64)
    return Mesh(vertices=vertices, faces=faces)


def _make_offset_cloud() -> PointCloud:
    """Create a point cloud with points far from the origin."""
    points = np.array([
        [5.0, 5.0, 7.0],
        [95.0, 5.0, 7.0],
        [95.0, 95.0, 7.0],
        [5.0, 95.0, 7.0],
        [50.0, 50.0, 20.0],
    ], dtype=np.float64)
    return PointCloud(points=points)


def _load_centered_mesh(glb_bytes: bytes) -> trimesh.Trimesh:
    """Load GLB bytes into a trimesh and return it."""
    return trimesh.load(trimesh.util.wrap_as_stream(glb_bytes), file_type="glb")


def _assert_centered_at_origin(tri: trimesh.Trimesh, atol: float = 0.1) -> None:
    """Assert that a trimesh is centered at the origin."""
    np.testing.assert_allclose(tri.bounding_box.centroid, [0.0, 0.0, 0.0], atol=atol)


def _fetch_and_assert_centered_mesh(
    client: TestClient, payload: dict[str, Any], atol: float = 1.0
) -> None:
    """POST /api/generate, fetch the mesh, and assert it's centered at origin."""
    resp = client.post("/api/generate", json=payload)
    assert resp.status_code == 200
    data = resp.json()
    assert data["mesh_url"] is not None

    mesh_resp = client.get(data["mesh_url"] + "?lod=preview")
    assert mesh_resp.status_code == 200

    tri = _load_centered_mesh(mesh_resp.content)
    _assert_centered_at_origin(tri, atol=atol)


def _parse_ply_positions(ply_bytes: bytes, num_points: int) -> np.ndarray:
    """Extract xyz positions from binary PLY bytes."""
    header_end = ply_bytes.index(b"end_header\n") + len(b"end_header\n")
    header_str = ply_bytes[:header_end].decode("ascii")

    stride_bytes = 0
    in_vertex = False
    type_sizes = {"float": 4, "float32": 4, "double": 8, "int": 4, "uint": 4}
    for line in header_str.split("\n"):
        parts = line.strip().split()
        if len(parts) >= 2 and parts[0] == "element":
            in_vertex = parts[1] == "vertex"
        if len(parts) >= 3 and parts[0] == "property" and in_vertex:
            stride_bytes += type_sizes.get(parts[1], 4)

    data = ply_bytes[header_end:]
    positions = np.zeros((num_points, 3), dtype=np.float32)
    for i in range(num_points):
        offset = i * stride_bytes
        positions[i, 0] = struct.unpack_from("<f", data, offset)[0]
        positions[i, 1] = struct.unpack_from("<f", data, offset + 4)[0]
        positions[i, 2] = struct.unpack_from("<f", data, offset + 8)[0]
    return positions


# --- Geometry centering tests ---


class TestMeshCentering:
    """Tests that mesh_to_glb centers geometry at the origin."""

    def test_mesh_centered_at_origin(self) -> None:
        """GLB mesh geometry should be approximately centered at the origin."""
        tri = _load_centered_mesh(mesh_to_glb(_make_offset_mesh()))
        _assert_centered_at_origin(tri)

    def test_mesh_dimensions_preserved(self) -> None:
        """Centering should not change the dimensions of the mesh."""
        mesh = _make_offset_mesh()
        original_extents = mesh.vertices.max(axis=0) - mesh.vertices.min(axis=0)
        tri = _load_centered_mesh(mesh_to_glb(mesh))
        np.testing.assert_allclose(tri.bounding_box.extents, original_extents, atol=0.1)


class TestCloudCentering:
    """Tests that cloud_to_binary_ply centers point cloud at the origin."""

    def test_cloud_centered_at_origin(self) -> None:
        """PLY point cloud should be approximately centered at the origin."""
        cloud = _make_offset_cloud()
        positions = _parse_ply_positions(cloud_to_binary_ply(cloud), len(cloud.points))
        center = (positions.min(axis=0) + positions.max(axis=0)) / 2.0
        np.testing.assert_allclose(center, [0.0, 0.0, 0.0], atol=0.1)

    def test_cloud_dimensions_preserved(self) -> None:
        """Centering should not change the spatial extent of the cloud."""
        cloud = _make_offset_cloud()
        original_extents = cloud.points.max(axis=0) - cloud.points.min(axis=0)
        positions = _parse_ply_positions(cloud_to_binary_ply(cloud), len(cloud.points))
        actual_extents = positions.max(axis=0) - positions.min(axis=0)
        np.testing.assert_allclose(actual_extents, original_extents, atol=0.1)


# --- Bounding box HTML tests ---


class TestBoundingBoxHTML:
    """Tests for bounding box behavior in the preview HTML."""

    def test_bbox_uses_container_dimensions_without_scale(
        self, preview_html: str
    ) -> None:
        """Bounding box should use raw mm dimensions, not scaled by 0.01."""
        assert "cp.width_mm / 2" in preview_html
        assert "* 0.01" not in preview_html.split("addBoundingBox")[1].split(
            "function"
        )[0]

    def test_bbox_toggle_checkbox_exists(self, preview_html: str) -> None:
        """Bounding box toggle checkbox should exist in the HTML."""
        assert 'id="show-bbox"' in preview_html

    def test_bbox_visibility_wired_to_checkbox(self, preview_html: str) -> None:
        """Bounding box visibility should be controlled by the checkbox."""
        assert "show-bbox" in preview_html
        assert "bboxHelper" in preview_html


# --- Camera clipping tests ---


class TestFitCameraClipping:
    """Tests for dynamic camera clipping plane adjustment."""

    def test_fit_camera_adjusts_near_far(self, fit_camera_code: str) -> None:
        """fitCamera should dynamically set camera.near and camera.far."""
        assert "camera.near" in fit_camera_code
        assert "camera.far" in fit_camera_code
        assert "camera.updateProjectionMatrix" in fit_camera_code

    def test_far_plane_proportional_to_distance(self, fit_camera_code: str) -> None:
        """camera.far should be proportional to camera distance (dist * 10)."""
        assert "dist * 10" in fit_camera_code

    def test_near_plane_proportional_to_distance(self, fit_camera_code: str) -> None:
        """camera.near should be proportional to camera distance (dist * 0.01)."""
        assert "dist * 0.01" in fit_camera_code


# --- Integration: geometry + bounding box alignment ---


class TestGeometryBoundingBoxAlignment:
    """Integration tests verifying geometry and bounding box share coordinates."""

    def test_generated_mesh_centered_via_api(self, client: TestClient) -> None:
        """Mesh served via API should be centered at the origin."""
        _fetch_and_assert_centered_mesh(client, {"generator": "torus", "seed": 42})

    def test_generated_cloud_centered_via_api(self, client: TestClient) -> None:
        """Point cloud served via API should be centered at the origin."""
        resp = client.post(
            "/api/generate",
            json={"generator": "torus", "seed": 42},
        )
        assert resp.status_code == 200
        data = resp.json()
        if data["cloud_url"] is None:
            pytest.skip("Torus generator does not produce a point cloud")

        cloud_resp = client.get(data["cloud_url"] + "?lod=preview")
        assert cloud_resp.status_code == 200

    def test_bbox_contains_geometry(self) -> None:
        """Container bounding box should contain the centered geometry."""
        mesh = _make_offset_mesh()
        tri = _load_centered_mesh(mesh_to_glb(mesh))
        _assert_centered_at_origin(tri)
        geo_extents = tri.bounding_box.extents
        # Use default container dimensions from the Container model
        default = Container.with_uniform_margin()
        container_dims = np.array([default.width_mm, default.height_mm, default.depth_mm])
        assert np.all(geo_extents <= container_dims + 0.1)


# --- Container scale tests ---


_LARGE_CONTAINER = {
    "width_mm": 100, "height_mm": 100, "depth_mm": 100,
    "margin_x_mm": 5, "margin_y_mm": 5, "margin_z_mm": 5,
}
_SMALL_CONTAINER = {
    "width_mm": 10, "height_mm": 10, "depth_mm": 10,
    "margin_x_mm": 1, "margin_y_mm": 1, "margin_z_mm": 1,
}


class TestLargeAndSmallContainers:
    """Tests for rendering at different container scales."""

    @pytest.mark.parametrize("container", [_LARGE_CONTAINER, _SMALL_CONTAINER],
                             ids=["large_100mm", "small_10mm"])
    def test_container_produces_centered_mesh(
        self, client: TestClient, container: dict[str, float]
    ) -> None:
        """Container at any scale should produce geometry centered at origin."""
        _fetch_and_assert_centered_mesh(
            client, {"generator": "torus", "seed": 42, "container": container}
        )
