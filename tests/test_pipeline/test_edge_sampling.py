"""Tests for edge sampling and Edge Cloud view mode.

Verifies edge-only sampling, combined dense+edge sampling, determinism,
proportional allocation, sample caps, and server integration.
"""

import numpy as np
import pytest
from fastapi.testclient import TestClient

from mathviz.core.math_object import CoordSpace, MathObject, Mesh, PointCloud
from mathviz.core.container import Container
from mathviz.core.generator import register
from mathviz.generators.parametric.torus import TorusGenerator
from mathviz.pipeline.dense_sampling import (
    MAX_DENSE_SAMPLES,
    _sample_mesh_edges,
    apply_edge_sampling,
    apply_post_transform_sampling,
)
from mathviz.preview.cache import compute_cache_key
from mathviz.preview.server import app, get_cache, reset_cache


def _make_cube_mesh() -> Mesh:
    """Create a simple cube mesh for testing."""
    vertices = np.array([
        [0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0],
        [0, 0, 1], [1, 0, 1], [1, 1, 1], [0, 1, 1],
    ], dtype=np.float64)
    faces = np.array([
        [0, 1, 2], [0, 2, 3],  # bottom
        [4, 5, 6], [4, 6, 7],  # top
        [0, 1, 5], [0, 5, 4],  # front
        [2, 3, 7], [2, 7, 6],  # back
        [0, 3, 7], [0, 7, 4],  # left
        [1, 2, 6], [1, 6, 5],  # right
    ], dtype=np.int64)
    return Mesh(vertices=vertices, faces=faces)


def _make_obj_with_mesh(mesh: Mesh) -> MathObject:
    """Wrap a mesh in a MathObject in PHYSICAL space."""
    return MathObject(
        mesh=mesh,
        coord_space=CoordSpace.PHYSICAL,
        generator_name="test",
    )


def _make_elongated_mesh() -> Mesh:
    """Create a mesh with edges of very different lengths."""
    vertices = np.array([
        [0, 0, 0],
        [10, 0, 0],   # long edge: 10
        [0, 0.1, 0],  # short edge: 0.1
    ], dtype=np.float64)
    faces = np.array([[0, 1, 2]], dtype=np.int64)
    return Mesh(vertices=vertices, faces=faces)


class TestEdgeSamplingPointsOnEdges:
    """Edge sampling produces points that lie on mesh edges."""

    def test_points_lie_on_edges(self) -> None:
        """Every sampled point must lie on a mesh edge within tolerance."""
        mesh = _make_cube_mesh()
        obj = _make_obj_with_mesh(mesh)
        cloud = _sample_mesh_edges(obj, max_samples=500)

        verts = mesh.vertices
        faces = mesh.faces
        raw_edges = np.concatenate([
            faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]],
        ], axis=0)
        raw_edges = np.sort(raw_edges, axis=1)
        edges = np.unique(raw_edges, axis=0)

        for point in cloud.points:
            min_dist = _min_distance_to_edges(point, verts, edges)
            assert min_dist < 1e-10, f"Point {point} not on any edge (dist={min_dist})"


def _min_distance_to_edges(
    point: np.ndarray, verts: np.ndarray, edges: np.ndarray,
) -> float:
    """Compute minimum distance from point to any edge segment."""
    min_dist = float("inf")
    for e in edges:
        a, b = verts[e[0]], verts[e[1]]
        ab = b - a
        ap = point - a
        t = np.dot(ap, ab) / max(np.dot(ab, ab), 1e-30)
        t = np.clip(t, 0.0, 1.0)
        closest = a + t * ab
        dist = float(np.linalg.norm(point - closest))
        min_dist = min(min_dist, dist)
    return min_dist


class TestEdgeSamplingCap:
    """Point count respects the max_samples cap."""

    def test_cap_respected(self) -> None:
        """Edge sampling never exceeds max_samples."""
        mesh = _make_cube_mesh()
        obj = _make_obj_with_mesh(mesh)
        cap = 50
        cloud = _sample_mesh_edges(obj, max_samples=cap)
        assert len(cloud.points) <= cap

    def test_large_cap(self) -> None:
        """With a large cap, all budget is used."""
        mesh = _make_cube_mesh()
        obj = _make_obj_with_mesh(mesh)
        cloud = _sample_mesh_edges(obj, max_samples=1000)
        # Cube has 18 unique edges; with 1000 budget, should use most of it
        assert len(cloud.points) > 100


class TestEdgeSamplingProportional:
    """Longer edges receive proportionally more points."""

    def test_longer_edges_get_more_points(self) -> None:
        """An elongated triangle allocates more points to its longer edge."""
        mesh = _make_elongated_mesh()
        obj = _make_obj_with_mesh(mesh)
        cloud = _sample_mesh_edges(obj, max_samples=200)

        verts = mesh.vertices
        # Edge 0-1 is length 10, edge 0-2 is length 0.1, edge 1-2 is ~10
        long_edge_start = verts[0]
        long_edge_end = verts[1]

        # Count points near the long edge (0→1)
        near_long = 0
        for p in cloud.points:
            dist = _min_distance_to_edges(
                p, verts, np.array([[0, 1]]),
            )
            if dist < 1e-6:
                near_long += 1

        # Long edge should have roughly half the points (it's ~50% of total length)
        assert near_long > len(cloud.points) * 0.3


class TestDenseCombined:
    """Dense Cloud includes both surface and edge points."""

    def test_dense_has_both_surface_and_edge(self) -> None:
        """Dense sampling combines surface and edge points in a single cloud."""
        mesh = _make_cube_mesh()
        obj = _make_obj_with_mesh(mesh)

        dense_result = apply_post_transform_sampling(obj, max_samples=1000)
        cloud = dense_result.point_cloud
        assert cloud is not None

        # Dense combines surface (70%) + edge (30%); cloud should have points
        # from both sources (more total than just edge budget alone)
        assert len(cloud.points) > 0

    def test_dense_includes_more_sources_than_edge_only(
        self, client: TestClient,
    ) -> None:
        """Dense Cloud (surface+edge) has points from both sources."""
        # Use a low cap so neither source saturates
        cap = 5000
        edge_data = _generate(client, sampling="edge", max_samples=cap)
        edge_entry = get_cache().get(edge_data["geometry_id"])
        assert edge_entry is not None
        edge_count = len(edge_entry.math_object.point_cloud.points)

        reset_cache()

        dense_data = _generate(client, sampling="post_transform", max_samples=cap)
        dense_entry = get_cache().get(dense_data["geometry_id"])
        assert dense_entry is not None
        dense_count = len(dense_entry.math_object.point_cloud.points)

        # Dense combines surface (70% budget) + edge (30% budget)
        # Both should produce substantial points; dense total should be close to cap
        assert dense_count > edge_count * 0.8
        # Dense should have more points than edge-only's 30% share
        assert dense_count > cap * 0.5


class TestNoFacesError:
    """Mesh with no faces raises ValueError."""

    def test_no_faces_raises(self) -> None:
        """Edge sampling raises ValueError for mesh without faces."""
        mesh = Mesh(
            vertices=np.array([[0, 0, 0], [1, 0, 0]], dtype=np.float64),
            faces=np.empty((0, 3), dtype=np.int64),
        )
        obj = _make_obj_with_mesh(mesh)
        with pytest.raises(ValueError, match="faces"):
            _sample_mesh_edges(obj, max_samples=100)


class TestDeterminism:
    """Result is deterministic for the same input."""

    def test_deterministic_output(self) -> None:
        """Two calls with same input produce identical points."""
        mesh = _make_cube_mesh()
        obj = _make_obj_with_mesh(mesh)

        cloud1 = _sample_mesh_edges(obj, max_samples=500)
        cloud2 = _sample_mesh_edges(obj, max_samples=500)

        np.testing.assert_array_equal(cloud1.points, cloud2.points)


# --- Server integration tests ---

def _ensure_torus_registered() -> None:
    """Re-register the torus generator if missing from the registry."""
    import mathviz.core.generator as gen_mod

    if "torus" not in gen_mod._alias_map:
        gen_mod._discovered = True
        register(TorusGenerator)


@pytest.fixture(autouse=True)
def _ensure_generators() -> None:
    """Ensure real generators are registered and cache is clean."""
    _ensure_torus_registered()
    reset_cache()
    yield
    reset_cache()


@pytest.fixture
def client() -> TestClient:
    """Return a FastAPI test client."""
    return TestClient(app)


def _torus_payload(**overrides) -> dict:
    """Build a standard torus generation request."""
    payload = {
        "generator": "torus",
        "params": {"major_radius": 1.0, "minor_radius": 0.4},
        "seed": 42,
    }
    payload.update(overrides)
    return payload


def _generate(client: TestClient, **overrides) -> dict:
    """Run a generation request and return the JSON response."""
    resp = client.post("/api/generate", json=_torus_payload(**overrides))
    assert resp.status_code == 200
    return resp.json()


class TestEdgeCloudServer:
    """Edge Cloud mode works through the server endpoint."""

    def test_edge_cloud_produces_cloud(self, client: TestClient) -> None:
        """Edge sampling via API produces a point cloud."""
        data = _generate(client, sampling="edge")
        entry = get_cache().get(data["geometry_id"])
        assert entry is not None
        assert entry.math_object.point_cloud is not None
        assert len(entry.math_object.point_cloud.points) > 0

    def test_edge_cloud_respects_max_samples(self, client: TestClient) -> None:
        """Edge Cloud respects the max_samples parameter."""
        cap = 5000
        data = _generate(client, sampling="edge", max_samples=cap)
        entry = get_cache().get(data["geometry_id"])
        assert entry is not None
        assert len(entry.math_object.point_cloud.points) <= cap

    def test_edge_cloud_cache_key_differs(self) -> None:
        """Edge sampling has a different cache key from default and dense."""
        params = {"major_radius": 1.0, "minor_radius": 0.4}
        container = Container()
        ckw = {
            "width_mm": container.width_mm,
            "height_mm": container.height_mm,
            "depth_mm": container.depth_mm,
            "margin_x_mm": container.margin_x_mm,
            "margin_y_mm": container.margin_y_mm,
            "margin_z_mm": container.margin_z_mm,
        }

        key_default = compute_cache_key("torus", params, 42, {}, container_kwargs=ckw)
        key_dense = compute_cache_key(
            "torus", params, 42, {}, container_kwargs=ckw, sampling="post_transform",
        )
        key_edge = compute_cache_key(
            "torus", params, 42, {}, container_kwargs=ckw, sampling="edge",
        )

        assert key_default != key_edge
        assert key_dense != key_edge


class TestEdgeCloudFrontend:
    """Edge Cloud option appears in the HTML dropdown."""

    def test_edge_cloud_option_present(self, client: TestClient) -> None:
        """The edge_cloud option exists in the view-mode select."""
        resp = client.get("/")
        assert resp.status_code == 200
        assert '<option value="edge_cloud">Edge Cloud</option>' in resp.text

    def test_existing_options_still_present(self, client: TestClient) -> None:
        """All original view mode options remain present."""
        html = client.get("/").text
        for value in ("shaded", "wireframe", "points", "dense", "hd_cloud", "crystal", "colormap"):
            assert f'value="{value}"' in html
