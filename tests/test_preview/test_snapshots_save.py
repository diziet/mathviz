"""Tests for the snapshot save feature (POST /api/snapshots)."""

import base64
import json
import os
import struct
import zlib
from pathlib import Path
from typing import Any
from unittest.mock import patch

import numpy as np
import pytest
from fastapi.testclient import TestClient

from mathviz.core.generator import register
from mathviz.core.math_object import MathObject, Mesh, PointCloud
from mathviz.generators.parametric.torus import TorusGenerator
from mathviz.preview.cache import CacheEntry
from mathviz.preview.server import app, get_cache, reset_cache
from mathviz.preview.snapshots import (
    DEFAULT_SNAPSHOTS_DIR,
    SNAPSHOTS_DIR_ENV_VAR,
    get_snapshots_dir,
)


def _make_tiny_png() -> bytes:
    """Create a minimal valid 1x1 red PNG."""

    def _chunk(chunk_type: bytes, data: bytes) -> bytes:
        crc = zlib.crc32(chunk_type + data) & 0xFFFFFFFF
        return struct.pack(">I", len(data)) + chunk_type + data + struct.pack(">I", crc)

    sig = b"\x89PNG\r\n\x1a\n"
    ihdr = struct.pack(">IIBBBBB", 1, 1, 8, 2, 0, 0, 0)
    raw_row = b"\x00\xff\x00\x00"
    idat = zlib.compress(raw_row)
    return sig + _chunk(b"IHDR", ihdr) + _chunk(b"IDAT", idat) + _chunk(b"IEND", b"")


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


@pytest.fixture
def client() -> TestClient:
    """Return a FastAPI test client."""
    return TestClient(app)


def _generate_torus(client: TestClient) -> str:
    """Generate a torus and return the geometry_id."""
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
    return resp.json()["geometry_id"]


from tests.test_preview.conftest import make_snapshot_request as _make_snapshot_request


@pytest.fixture
def torus_snapshot(
    client: TestClient, tmp_path: Path
) -> tuple[dict[str, Any], Path]:
    """Generate torus, save snapshot, return (response_data, snapshot_dir)."""
    gid = _generate_torus(client)
    with patch.dict(os.environ, {SNAPSHOTS_DIR_ENV_VAR: str(tmp_path)}):
        resp = client.post("/api/snapshots", json=_make_snapshot_request(gid))
    assert resp.status_code == 200
    data = resp.json()
    snapshot_dir = tmp_path / data["snapshot_id"]
    return data, snapshot_dir


class TestSnapshotCreation:
    """Tests for POST /api/snapshots creating snapshot directories."""

    def test_creates_snapshot_directory_with_metadata(
        self, torus_snapshot: tuple[dict[str, Any], Path]
    ) -> None:
        """POST /api/snapshots creates a snapshot directory with metadata.json."""
        _data, snapshot_dir = torus_snapshot
        assert snapshot_dir.is_dir()
        assert (snapshot_dir / "metadata.json").is_file()

    def test_metadata_contains_required_fields(
        self, torus_snapshot: tuple[dict[str, Any], Path]
    ) -> None:
        """metadata.json contains generator, params, seed, container, created_at."""
        _data, snapshot_dir = torus_snapshot
        metadata = json.loads((snapshot_dir / "metadata.json").read_text())

        assert metadata["generator"] == "torus"
        assert metadata["params"] == {"major_radius": 1.0, "minor_radius": 0.4}
        assert metadata["seed"] == 42
        assert "container" in metadata
        assert metadata["container"]["width_mm"] == 100.0
        assert "created_at" in metadata
        assert "geometry_id" in metadata

    def test_geometry_files_copied_to_snapshot_dir(
        self, torus_snapshot: tuple[dict[str, Any], Path]
    ) -> None:
        """Geometry files (mesh.glb and/or cloud.ply) are copied to snapshot dir."""
        _data, snapshot_dir = torus_snapshot
        mesh_file = snapshot_dir / "mesh.glb"
        assert mesh_file.is_file()
        assert mesh_file.read_bytes()[:4] == b"glTF"

    def test_snapshot_id_is_timestamp_based(
        self, torus_snapshot: tuple[dict[str, Any], Path]
    ) -> None:
        """Snapshot ID is timestamp-based (YYYYMMDD-HHMMSS-ffffff format)."""
        data, _snapshot_dir = torus_snapshot
        snapshot_id = data["snapshot_id"]
        parts = snapshot_id.split("-")
        assert len(parts) >= 3
        assert len(parts[0]) == 8  # YYYYMMDD
        assert parts[0].isdigit()
        assert len(parts[1]) == 6  # HHMMSS
        assert parts[1].isdigit()
        assert len(parts[2]) == 6  # microseconds
        assert parts[2].isdigit()

    def test_snapshot_ids_are_unique(
        self, client: TestClient, tmp_path: Path
    ) -> None:
        """Multiple snapshots get unique IDs."""
        gid = _generate_torus(client)
        ids = set()
        with patch.dict(os.environ, {SNAPSHOTS_DIR_ENV_VAR: str(tmp_path)}):
            for _ in range(3):
                resp = client.post(
                    "/api/snapshots", json=_make_snapshot_request(gid)
                )
                assert resp.status_code == 200
                ids.add(resp.json()["snapshot_id"])

        assert len(ids) == 3

    def test_response_does_not_expose_filesystem_path(
        self, torus_snapshot: tuple[dict[str, Any], Path]
    ) -> None:
        """Response contains snapshot_id but not full filesystem path."""
        data, _snapshot_dir = torus_snapshot
        assert "snapshot_id" in data
        assert "path" not in data


class TestSnapshotUI:
    """Tests for the Save button in preview HTML."""

    def test_preview_html_contains_save_button(self, client: TestClient) -> None:
        """Preview HTML contains a Save button."""
        resp = client.get("/")
        assert resp.status_code == 200
        html = resp.text
        assert 'id="save-btn"' in html
        assert ">Save<" in html

    def test_preview_html_contains_save_toast(self, client: TestClient) -> None:
        """Preview HTML contains a save toast notification."""
        resp = client.get("/")
        html = resp.text
        assert 'id="save-toast"' in html
        assert "Saved!" in html


class TestSnapshotErrorHandling:
    """Tests for error cases in snapshot creation."""

    def test_missing_geometry_returns_400(self, client: TestClient) -> None:
        """Saving without a generated geometry returns 400."""
        resp = client.post(
            "/api/snapshots",
            json={
                "generator": "torus",
                "params": {},
                "seed": 42,
                "geometry_id": "nonexistent_id_12345",
            },
        )
        assert resp.status_code == 400
        assert "geometry" in resp.json()["detail"].lower()


class TestSnapshotConfigurableDir:
    """Tests for configurable snapshot directory."""

    def test_default_snapshots_dir(self) -> None:
        """Default snapshots directory is ~/.mathviz/snapshots."""
        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop(SNAPSHOTS_DIR_ENV_VAR, None)
            result = get_snapshots_dir()
        assert result == DEFAULT_SNAPSHOTS_DIR

    def test_custom_snapshots_dir_via_env(self, tmp_path: Path) -> None:
        """Snapshot directory is configurable via env var."""
        custom_dir = tmp_path / "custom_snapshots"
        with patch.dict(os.environ, {SNAPSHOTS_DIR_ENV_VAR: str(custom_dir)}):
            result = get_snapshots_dir()
        assert result == custom_dir


class TestSnapshotPointCloud:
    """Tests for snapshots with point cloud geometry."""

    def test_cloud_ply_saved_when_present(
        self, client: TestClient, tmp_path: Path
    ) -> None:
        """cloud.ply is saved when the geometry has a point cloud."""
        obj = MathObject(
            mesh=Mesh(
                vertices=np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]], dtype=np.float64),
                faces=np.array([[0, 1, 2]], dtype=np.int64),
            ),
            point_cloud=PointCloud(
                points=np.random.default_rng(0).random((100, 3)).astype(np.float64),
            ),
            generator_name="test",
        )
        entry = CacheEntry(
            math_object=obj,
            generator_name="test",
            params={},
            seed=0,
            resolution_kwargs={},
        )
        cache_key = "test_cloud_key_123456789012"
        get_cache().put(cache_key, entry)

        req = {
            "generator": "test",
            "params": {},
            "seed": 0,
            "geometry_id": cache_key,
        }
        with patch.dict(os.environ, {SNAPSHOTS_DIR_ENV_VAR: str(tmp_path)}):
            resp = client.post("/api/snapshots", json=req)

        assert resp.status_code == 200
        snapshot_dir = tmp_path / resp.json()["snapshot_id"]
        assert (snapshot_dir / "mesh.glb").is_file()
        assert (snapshot_dir / "cloud.ply").is_file()


class TestSnapshotNoPyvista:
    """Tests that snapshot save does not import or call PyVista."""

    def test_save_does_not_import_pyvista(
        self, client: TestClient, tmp_path: Path
    ) -> None:
        """POST /api/snapshots does not import or call PyVista."""
        import sys

        gid = _generate_torus(client)
        # Remove pyvista from sys.modules so we can detect a fresh import
        pyvista_modules = [k for k in sys.modules if k.startswith("pyvista")]
        saved = {k: sys.modules.pop(k) for k in pyvista_modules}
        try:
            # Patch __import__ to detect pyvista imports
            original_import = __builtins__.__import__
            pyvista_imported = []

            def _tracking_import(name: str, *args: Any, **kwargs: Any) -> Any:
                if name.startswith("pyvista"):
                    pyvista_imported.append(name)
                return original_import(name, *args, **kwargs)

            __builtins__.__import__ = _tracking_import
            try:
                req = _make_snapshot_request(gid)
                with patch.dict(os.environ, {SNAPSHOTS_DIR_ENV_VAR: str(tmp_path)}):
                    resp = client.post("/api/snapshots", json=req)
                assert resp.status_code == 200
                assert pyvista_imported == [], (
                    f"PyVista was imported during save: {pyvista_imported}"
                )
            finally:
                __builtins__.__import__ = original_import
        finally:
            sys.modules.update(saved)


class TestSnapshotThumbnailBase64:
    """Tests for base64 thumbnail support in snapshot save."""

    def test_base64_thumbnail_creates_file(
        self, client: TestClient, tmp_path: Path
    ) -> None:
        """Saving with a base64 thumbnail field creates thumbnail.png on disk."""
        gid = _generate_torus(client)
        png_bytes = _make_tiny_png()
        b64 = base64.b64encode(png_bytes).decode()
        req = _make_snapshot_request(gid)
        req["thumbnail"] = b64
        with patch.dict(os.environ, {SNAPSHOTS_DIR_ENV_VAR: str(tmp_path)}):
            resp = client.post("/api/snapshots", json=req)
        assert resp.status_code == 200
        snapshot_dir = tmp_path / resp.json()["snapshot_id"]
        thumb_path = snapshot_dir / "thumbnail.png"
        assert thumb_path.is_file()
        assert thumb_path.read_bytes() == png_bytes

    def test_no_thumbnail_field_succeeds(
        self, client: TestClient, tmp_path: Path
    ) -> None:
        """Saving without a thumbnail field succeeds (no thumbnail.png created)."""
        gid = _generate_torus(client)
        req = _make_snapshot_request(gid)
        assert "thumbnail" not in req
        with patch.dict(os.environ, {SNAPSHOTS_DIR_ENV_VAR: str(tmp_path)}):
            resp = client.post("/api/snapshots", json=req)
        assert resp.status_code == 200
        snapshot_dir = tmp_path / resp.json()["snapshot_id"]
        assert not (snapshot_dir / "thumbnail.png").exists()

    def test_invalid_base64_returns_400(self, client: TestClient) -> None:
        """Invalid base64 thumbnail returns 400, does not crash the server."""
        gid = _generate_torus(client)
        req = _make_snapshot_request(gid)
        req["thumbnail"] = "!!!not-valid-base64!!!"
        resp = client.post("/api/snapshots", json=req)
        assert resp.status_code == 400
        assert "thumbnail" in resp.json()["detail"].lower()

    def test_server_responsive_after_save(
        self, client: TestClient, tmp_path: Path
    ) -> None:
        """Server remains responsive after a save operation."""
        gid = _generate_torus(client)
        req = _make_snapshot_request(gid)
        with patch.dict(os.environ, {SNAPSHOTS_DIR_ENV_VAR: str(tmp_path)}):
            resp = client.post("/api/snapshots", json=req)
        assert resp.status_code == 200
        # Verify server still responds to other requests
        health = client.get("/api/generators")
        assert health.status_code == 200
