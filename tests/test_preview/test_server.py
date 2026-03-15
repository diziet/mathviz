"""Tests for the FastAPI preview server."""

import io
from typing import Any
from unittest.mock import patch

import numpy as np
import pytest
import trimesh
from fastapi.testclient import TestClient

from mathviz.core.generator import register
from mathviz.core.math_object import MathObject, Mesh
from mathviz.generators.parametric.torus import TorusGenerator
from mathviz.preview.cache import GeometryCache, compute_cache_key
from mathviz.preview.lod import PREVIEW_MAX_FACES
from mathviz.preview.server import app, get_cache, reset_cache


def _ensure_torus_registered() -> None:
    """Re-register the torus generator if missing from the registry."""
    import mathviz.core.generator as gen_mod

    if "torus" not in gen_mod._alias_map:
        gen_mod._discovered = True
        register(TorusGenerator)


# --- Fixtures ---


@pytest.fixture(autouse=True)
def _ensure_generators() -> None:
    """Ensure real generators are registered and cache is clean."""
    _ensure_torus_registered()
    reset_cache()


@pytest.fixture
def client() -> TestClient:
    """Return a FastAPI test client."""
    return TestClient(app)


# --- GET /api/generators ---


class TestListGenerators:
    """Tests for GET /api/generators."""

    def test_returns_json_list_including_torus(self, client: TestClient) -> None:
        """GET /api/generators returns JSON list including the torus."""
        resp = client.get("/api/generators")
        assert resp.status_code == 200
        data = resp.json()
        assert isinstance(data, list)
        names = [g["name"] for g in data]
        assert "torus" in names

    def test_generator_entries_have_expected_fields(self, client: TestClient) -> None:
        """Each generator entry has name, category, aliases, description."""
        resp = client.get("/api/generators")
        data = resp.json()
        for entry in data:
            assert "name" in entry
            assert "category" in entry
            assert "aliases" in entry
            assert "description" in entry


# --- GET /api/generators/{name} ---


class TestGetGeneratorDetails:
    """Tests for GET /api/generators/{name}."""

    def test_known_generator_returns_details(self, client: TestClient) -> None:
        """Known generator returns its metadata."""
        resp = client.get("/api/generators/torus")
        assert resp.status_code == 200
        data = resp.json()
        assert data["name"] == "torus"
        assert data["category"] == "parametric"

    def test_unknown_generator_returns_404(self, client: TestClient) -> None:
        """Unknown generator returns 404 with helpful error."""
        resp = client.get("/api/generators/nonexistent_xyz")
        assert resp.status_code == 404
        assert "not found" in resp.json()["detail"].lower()


# --- POST /api/generate ---


class TestGenerate:
    """Tests for POST /api/generate."""

    def test_torus_returns_geometry_urls(self, client: TestClient) -> None:
        """POST /api/generate with torus returns geometry URLs."""
        resp = client.post(
            "/api/generate",
            json={
                "generator": "torus",
                "params": {"major_radius": 1.0, "minor_radius": 0.4},
                "seed": 42,
                "resolution": {"grid_resolution": 16},
            },
        )
        assert resp.status_code == 200
        data = resp.json()
        assert "geometry_id" in data
        assert data["mesh_url"] is not None
        assert "/mesh" in data["mesh_url"]

    def test_unknown_generator_returns_404(self, client: TestClient) -> None:
        """Unknown generator returns 404 with helpful error."""
        resp = client.post(
            "/api/generate",
            json={"generator": "nonexistent_xyz"},
        )
        assert resp.status_code == 404
        assert "not found" in resp.json()["detail"].lower()

    def test_cache_hit_on_second_call(self, client: TestClient) -> None:
        """Same generation request hits cache on second call (no re-generation)."""
        payload = {
            "generator": "torus",
            "params": {"major_radius": 1.0, "minor_radius": 0.4},
            "seed": 42,
            "resolution": {"grid_resolution": 16},
        }

        resp1 = client.post("/api/generate", json=payload)
        assert resp1.status_code == 200
        assert get_cache().size == 1

        with patch("mathviz.preview.executor.run_pipeline", wraps=None) as mock_run:
            resp2 = client.post("/api/generate", json=payload)
            assert resp2.status_code == 200
            mock_run.assert_not_called()

        assert resp1.json()["geometry_id"] == resp2.json()["geometry_id"]
        assert get_cache().size == 1


# --- GET /api/geometry/{id}/mesh ---


class TestGetMesh:
    """Tests for GET /api/geometry/{id}/mesh."""

    def _generate_torus(self, client: TestClient) -> str:
        """Helper: generate a torus and return the geometry_id."""
        resp = client.post(
            "/api/generate",
            json={
                "generator": "torus",
                "seed": 42,
                "resolution": {"grid_resolution": 16},
            },
        )
        return resp.json()["geometry_id"]

    def test_preview_mesh_returns_valid_binary(self, client: TestClient) -> None:
        """GET /api/geometry/{id}/mesh?lod=preview returns valid binary data."""
        gid = self._generate_torus(client)
        resp = client.get(f"/api/geometry/{gid}/mesh?lod=preview")
        assert resp.status_code == 200
        assert resp.headers["content-type"] == "model/gltf-binary"
        # GLB starts with magic bytes 0x46546C67 ("glTF")
        assert resp.content[:4] == b"glTF"

    def test_full_mesh_returns_valid_binary(self, client: TestClient) -> None:
        """GET /api/geometry/{id}/mesh?lod=full returns valid binary data."""
        gid = self._generate_torus(client)
        resp = client.get(f"/api/geometry/{gid}/mesh?lod=full")
        assert resp.status_code == 200
        assert resp.content[:4] == b"glTF"

    def test_missing_geometry_returns_404(self, client: TestClient) -> None:
        """Unknown geometry ID returns 404."""
        resp = client.get("/api/geometry/does_not_exist/mesh")
        assert resp.status_code == 404


# --- LOD constraints ---


class TestLodConstraints:
    """Tests for LOD decimation limits."""

    def test_preview_mesh_within_face_limit(self, client: TestClient) -> None:
        """LOD preview mesh has <= 100K faces."""
        # Use a high-res torus that exceeds the limit
        resp = client.post(
            "/api/generate",
            json={
                "generator": "torus",
                "seed": 42,
                "resolution": {"grid_resolution": 400},
            },
        )
        gid = resp.json()["geometry_id"]
        resp = client.get(f"/api/geometry/{gid}/mesh?lod=preview")
        assert resp.status_code == 200
        # Verify via trimesh that the result has <= 100K faces
        scene_or_mesh = trimesh.load(io.BytesIO(resp.content), file_type="glb")
        if isinstance(scene_or_mesh, trimesh.Scene):
            total_faces = sum(
                len(g.faces) for g in scene_or_mesh.geometry.values()
            )
        else:
            total_faces = len(scene_or_mesh.faces)
        assert total_faces <= PREVIEW_MAX_FACES


# --- Cache unit tests ---


class TestGeometryCache:
    """Tests for the GeometryCache."""

    def test_cache_key_deterministic(self) -> None:
        """Same inputs produce the same cache key."""
        key1 = compute_cache_key("torus", {"r": 1.0}, 42, {})
        key2 = compute_cache_key("torus", {"r": 1.0}, 42, {})
        assert key1 == key2

    def test_cache_key_varies_with_params(self) -> None:
        """Different params produce different cache keys."""
        key1 = compute_cache_key("torus", {"r": 1.0}, 42, {})
        key2 = compute_cache_key("torus", {"r": 2.0}, 42, {})
        assert key1 != key2

    def test_lru_eviction(self) -> None:
        """Cache evicts oldest entry when at capacity."""
        cache = GeometryCache(max_entries=2)
        obj = MathObject(
            mesh=Mesh(
                vertices=np.zeros((3, 3), dtype=np.float64),
                faces=np.array([[0, 1, 2]], dtype=np.int64),
            ),
            generator_name="test",
        )
        for i in range(3):
            entry = _make_entry(obj, f"gen_{i}")
            cache.put(f"key_{i}", entry)

        assert cache.size == 2
        assert cache.get("key_0") is None
        assert cache.get("key_1") is not None
        assert cache.get("key_2") is not None


def _make_entry(obj: MathObject, name: str = "test") -> Any:
    """Create a CacheEntry for testing."""
    from mathviz.preview.cache import CacheEntry
    return CacheEntry(
        math_object=obj,
        generator_name=name,
        params={},
        seed=42,
        resolution_kwargs={},
    )
