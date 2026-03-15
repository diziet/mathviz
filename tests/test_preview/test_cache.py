"""Tests for disk-based generation cache with UI indicator and invalidation."""

import time
from pathlib import Path
from typing import Any

import pytest
from fastapi.testclient import TestClient

from mathviz.core.generator import register
from mathviz.generators.parametric.torus import TorusGenerator
from mathviz.preview.disk_cache import DiskCache
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


@pytest.fixture
def tmp_cache_dir(tmp_path: Path) -> Path:
    """Return a temporary cache directory."""
    cache_dir = tmp_path / "test_cache"
    cache_dir.mkdir()
    return cache_dir


@pytest.fixture
def disk_cache(tmp_cache_dir: Path) -> DiskCache:
    """Return a DiskCache using a temp directory."""
    return DiskCache(cache_dir=tmp_cache_dir)


def _torus_payload(**overrides: Any) -> dict[str, Any]:
    """Build a standard torus generation request."""
    payload: dict[str, Any] = {
        "generator": "torus",
        "params": {"major_radius": 1.0, "minor_radius": 0.4},
        "seed": 42,
        "resolution": {"grid_resolution": 16},
    }
    payload.update(overrides)
    return payload


# --- Same request twice: second call serves from cache ---


class TestCacheHit:
    """Same request twice should serve from cache on second call."""

    def test_second_call_is_cache_hit(self, client: TestClient) -> None:
        """Second identical request returns X-Cache: HIT."""
        payload = _torus_payload()
        resp1 = client.post("/api/generate", json=payload)
        assert resp1.status_code == 200
        assert resp1.headers["x-cache"] == "MISS"

        resp2 = client.post("/api/generate", json=payload)
        assert resp2.status_code == 200
        assert resp2.headers["x-cache"] == "HIT"
        assert resp1.json()["geometry_id"] == resp2.json()["geometry_id"]


# --- Different params produce cache miss ---


class TestCacheMiss:
    """Different parameters should produce a cache miss."""

    def test_different_params_produce_miss(self, client: TestClient) -> None:
        """Changing params results in X-Cache: MISS."""
        resp1 = client.post("/api/generate", json=_torus_payload())
        assert resp1.status_code == 200
        assert resp1.headers["x-cache"] == "MISS"

        resp2 = client.post(
            "/api/generate",
            json=_torus_payload(params={"major_radius": 2.0, "minor_radius": 0.3}),
        )
        assert resp2.status_code == 200
        assert resp2.headers["x-cache"] == "MISS"
        assert resp1.json()["geometry_id"] != resp2.json()["geometry_id"]


# --- force: true bypasses cache ---


class TestForceRegenerate:
    """force: true should bypass cache and overwrite the entry."""

    def test_force_bypasses_cache(self, client: TestClient) -> None:
        """Request with force=true returns MISS even on second call."""
        payload = _torus_payload()
        resp1 = client.post("/api/generate", json=payload)
        assert resp1.status_code == 200
        assert resp1.headers["x-cache"] == "MISS"

        force_payload = _torus_payload(force=True)
        resp2 = client.post("/api/generate", json=force_payload)
        assert resp2.status_code == 200
        assert resp2.headers["x-cache"] == "MISS"


# --- POST /api/cache/clear ---


class TestCacheClear:
    """POST /api/cache/clear should remove all cached files."""

    def test_clear_removes_entries(self, client: TestClient) -> None:
        """Clearing cache removes all entries."""
        client.post("/api/generate", json=_torus_payload())
        assert get_cache().size == 1

        resp = client.post("/api/cache/clear")
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "ok"
        assert get_cache().size == 0


# --- X-Cache header present ---


class TestXCacheHeader:
    """X-Cache header should be present in all generate responses."""

    def test_xcache_header_on_miss(self, client: TestClient) -> None:
        """First request has X-Cache: MISS."""
        resp = client.post("/api/generate", json=_torus_payload())
        assert "x-cache" in resp.headers
        assert resp.headers["x-cache"] == "MISS"

    def test_xcache_header_on_hit(self, client: TestClient) -> None:
        """Second identical request has X-Cache: HIT."""
        payload = _torus_payload()
        client.post("/api/generate", json=payload)
        resp = client.post("/api/generate", json=payload)
        assert "x-cache" in resp.headers
        assert resp.headers["x-cache"] == "HIT"


# --- Cache directory created automatically ---


class TestDiskCacheAutoCreate:
    """Cache directory should be created on first use."""

    def test_cache_dir_created_on_put(self, tmp_path: Path) -> None:
        """Cache directory is created when first entry is stored."""
        cache_dir = tmp_path / "new_cache"
        assert not cache_dir.exists()
        dc = DiskCache(cache_dir=cache_dir)
        dc.put(
            "testkey",
            generator_name="torus",
            params={},
            seed=42,
            resolution_kwargs={},
            container_kwargs={},
            mesh_data=b"fake-mesh-data",
        )
        assert cache_dir.is_dir()
        assert (cache_dir / "testkey" / "metadata.json").is_file()


# --- Cache size limit and eviction ---


class TestCacheSizeLimit:
    """Cache should respect max size and evict oldest entries."""

    def test_evicts_oldest_when_full(self, tmp_cache_dir: Path) -> None:
        """Oldest entry is evicted when cache exceeds size limit."""
        # Create cache with tiny 500-byte limit
        dc = DiskCache(cache_dir=tmp_cache_dir, max_size_bytes=500)

        # Store first entry (~200 bytes)
        dc.put(
            "key1",
            generator_name="gen1",
            params={},
            seed=1,
            resolution_kwargs={},
            container_kwargs={},
            mesh_data=b"x" * 100,
        )
        assert dc.get("key1") is not None

        # Give key1 an older timestamp
        time.sleep(0.05)

        # Store second entry (~200 bytes)
        dc.put(
            "key2",
            generator_name="gen2",
            params={},
            seed=2,
            resolution_kwargs={},
            container_kwargs={},
            mesh_data=b"y" * 100,
        )

        time.sleep(0.05)

        # Store third large entry that exceeds limit
        dc.put(
            "key3",
            generator_name="gen3",
            params={},
            seed=3,
            resolution_kwargs={},
            container_kwargs={},
            mesh_data=b"z" * 300,
        )

        # key1 (oldest) should have been evicted
        assert dc.get("key1") is None
        assert dc.get("key3") is not None


# --- Preview HTML contains force-regenerate button ---


class TestRegenerateButton:
    """Preview HTML should contain a force-regenerate button."""

    def test_regenerate_button_in_html(self, client: TestClient) -> None:
        """The viewer HTML contains a regenerate button."""
        resp = client.get("/")
        assert resp.status_code == 200
        assert 'id="regenerate-btn"' in resp.text


# --- UI shows Cached badge ---


class TestCachedBadge:
    """UI should show a Cached badge when result is from cache."""

    def test_cache_badge_in_html(self, client: TestClient) -> None:
        """The viewer HTML contains a cache badge element."""
        resp = client.get("/")
        assert resp.status_code == 200
        assert 'id="cache-badge"' in resp.text
        assert "Cached" in resp.text


# --- DiskCache unit tests ---


class TestDiskCacheUnit:
    """Unit tests for DiskCache operations."""

    def test_get_nonexistent_returns_none(self, disk_cache: DiskCache) -> None:
        """Getting a nonexistent key returns None."""
        assert disk_cache.get("nonexistent") is None

    def test_put_and_get(self, disk_cache: DiskCache) -> None:
        """Stored entry can be retrieved."""
        disk_cache.put(
            "abc123",
            generator_name="torus",
            params={"r": 1.0},
            seed=42,
            resolution_kwargs={},
            container_kwargs={},
            mesh_data=b"mesh-data",
            cloud_data=b"cloud-data",
        )
        entry = disk_cache.get("abc123")
        assert entry is not None
        assert entry.generator_name == "torus"
        assert entry.seed == 42
        assert entry.has_mesh is True
        assert entry.has_cloud is True

    def test_mesh_and_cloud_paths(self, disk_cache: DiskCache) -> None:
        """Mesh and cloud file paths are correct."""
        disk_cache.put(
            "pathtest",
            generator_name="torus",
            params={},
            seed=1,
            resolution_kwargs={},
            container_kwargs={},
            mesh_data=b"glb-content",
            cloud_data=b"ply-content",
        )
        mesh_path = disk_cache.get_mesh_path("pathtest")
        cloud_path = disk_cache.get_cloud_path("pathtest")
        assert mesh_path is not None
        assert mesh_path.read_bytes() == b"glb-content"
        assert cloud_path is not None
        assert cloud_path.read_bytes() == b"ply-content"

    def test_clear_removes_all(self, disk_cache: DiskCache) -> None:
        """Clear removes all cached entries."""
        for i in range(3):
            disk_cache.put(
                f"key{i}",
                generator_name=f"gen{i}",
                params={},
                seed=i,
                resolution_kwargs={},
                container_kwargs={},
            )
        count = disk_cache.clear()
        assert count == 3
        assert disk_cache.get("key0") is None

    def test_total_size_bytes(self, disk_cache: DiskCache) -> None:
        """Total size reflects stored data."""
        assert disk_cache.total_size_bytes() == 0
        disk_cache.put(
            "sizetest",
            generator_name="t",
            params={},
            seed=1,
            resolution_kwargs={},
            container_kwargs={},
            mesh_data=b"x" * 1000,
        )
        assert disk_cache.total_size_bytes() > 1000
