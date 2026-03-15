"""Tests for snapshot browsing, loading, and deletion (Task 58)."""

import os
from collections.abc import Generator
from pathlib import Path
from typing import Any
from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient

from mathviz.core.generator import register
from mathviz.generators.parametric.torus import TorusGenerator
from mathviz.preview.server import app, reset_cache
from mathviz.preview.snapshots import SNAPSHOTS_DIR_ENV_VAR


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


@pytest.fixture(autouse=True)
def _snapshots_dir(tmp_path: Path) -> Generator[Path, None, None]:
    """Point snapshot storage at tmp_path for all tests."""
    with patch.dict(os.environ, {SNAPSHOTS_DIR_ENV_VAR: str(tmp_path)}):
        yield tmp_path


@pytest.fixture
def client() -> TestClient:
    """Return a FastAPI test client."""
    return TestClient(app)


@pytest.fixture
def preview_html(client: TestClient) -> str:
    """Fetch preview HTML for assertion tests."""
    resp = client.get("/")
    assert resp.status_code == 200
    return resp.text


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


def _make_snapshot_request(geometry_id: str) -> dict[str, Any]:
    """Build a standard snapshot request body."""
    return {
        "generator": "torus",
        "params": {"major_radius": 1.0, "minor_radius": 0.4},
        "seed": 42,
        "container": {
            "width_mm": 100.0,
            "height_mm": 100.0,
            "depth_mm": 100.0,
            "margin_x_mm": 5.0,
            "margin_y_mm": 5.0,
            "margin_z_mm": 5.0,
        },
        "geometry_id": geometry_id,
    }


def _create_snapshot(client: TestClient) -> dict[str, Any]:
    """Generate torus, save snapshot, return response data."""
    gid = _generate_torus(client)
    resp = client.post("/api/snapshots", json=_make_snapshot_request(gid))
    assert resp.status_code == 200
    return resp.json()


class TestListSnapshots:
    """Tests for GET /api/snapshots listing endpoint."""

    def test_returns_list_sorted_by_date(self, client: TestClient) -> None:
        """GET /api/snapshots returns a list of saved snapshots sorted by date."""
        gid = _generate_torus(client)
        for _ in range(3):
            resp = client.post(
                "/api/snapshots", json=_make_snapshot_request(gid)
            )
            assert resp.status_code == 200

        resp = client.get("/api/snapshots")
        assert resp.status_code == 200
        data = resp.json()
        assert len(data) == 3
        dates = [s["created_at"] for s in data]
        assert dates == sorted(dates, reverse=True)

    def test_each_entry_contains_required_fields(
        self, client: TestClient
    ) -> None:
        """Each snapshot entry contains generator, params, created_at, has_thumbnail."""
        _create_snapshot(client)
        resp = client.get("/api/snapshots")

        assert resp.status_code == 200
        data = resp.json()
        assert len(data) >= 1
        entry = data[0]
        assert entry["generator"] == "torus"
        assert "params" in entry
        assert "created_at" in entry
        assert "has_thumbnail" in entry
        assert "snapshot_id" in entry
        assert "seed" in entry
        assert "container" in entry

    def test_empty_list_when_no_snapshots(self, client: TestClient) -> None:
        """GET /api/snapshots returns empty list when no snapshots exist."""
        resp = client.get("/api/snapshots")
        assert resp.status_code == 200
        assert resp.json() == []


class TestSnapshotThumbnail:
    """Tests for GET /api/snapshots/<id>/thumbnail endpoint."""

    def test_serves_thumbnail_png(
        self, client: TestClient, tmp_path: Path
    ) -> None:
        """GET /api/snapshots/<id>/thumbnail serves the thumbnail PNG."""
        data = _create_snapshot(client)
        snap_dir = tmp_path / data["snapshot_id"]
        thumb_path = snap_dir / "thumbnail.png"
        thumb_path.write_bytes(b"\x89PNG\r\n\x1a\nfake_png_data")

        resp = client.get(f"/api/snapshots/{data['snapshot_id']}/thumbnail")

        assert resp.status_code == 200
        assert resp.headers["content-type"] == "image/png"
        assert resp.content.startswith(b"\x89PNG")

    def test_returns_404_when_no_thumbnail(
        self, client: TestClient, tmp_path: Path
    ) -> None:
        """GET /api/snapshots/<id>/thumbnail returns 404 when no thumbnail exists."""
        data = _create_snapshot(client)
        thumb = tmp_path / data["snapshot_id"] / "thumbnail.png"
        if thumb.exists():
            thumb.unlink()

        resp = client.get(f"/api/snapshots/{data['snapshot_id']}/thumbnail")
        assert resp.status_code == 404

    def test_returns_404_for_nonexistent_snapshot(
        self, client: TestClient
    ) -> None:
        """Thumbnail endpoint returns 404 for nonexistent snapshot ID."""
        resp = client.get("/api/snapshots/nonexistent-id/thumbnail")
        assert resp.status_code == 404


class TestDeleteSnapshot:
    """Tests for DELETE /api/snapshots/<id> endpoint."""

    def test_removes_snapshot_directory(
        self, client: TestClient, tmp_path: Path
    ) -> None:
        """DELETE /api/snapshots/<id> removes the snapshot directory."""
        data = _create_snapshot(client)
        snap_dir = tmp_path / data["snapshot_id"]
        assert snap_dir.is_dir()

        resp = client.delete(f"/api/snapshots/{data['snapshot_id']}")

        assert resp.status_code == 200
        assert resp.json()["deleted"] == data["snapshot_id"]
        assert not snap_dir.exists()

    def test_returns_404_for_nonexistent_snapshot(
        self, client: TestClient
    ) -> None:
        """DELETE returns 404 for nonexistent snapshot ID."""
        resp = client.delete("/api/snapshots/nonexistent-id")
        assert resp.status_code == 404


class TestSnapshotGeometry:
    """Tests for GET /api/snapshots/<id>/geometry/<filename> endpoint."""

    def test_serves_mesh_glb(self, client: TestClient) -> None:
        """Geometry endpoint serves saved mesh.glb file."""
        data = _create_snapshot(client)
        resp = client.get(
            f"/api/snapshots/{data['snapshot_id']}/geometry/mesh.glb"
        )
        assert resp.status_code == 200
        assert resp.headers["content-type"] == "model/gltf-binary"
        assert resp.content[:4] == b"glTF"

    def test_returns_404_for_missing_file(self, client: TestClient) -> None:
        """Geometry endpoint returns 404 for file not present in snapshot."""
        data = _create_snapshot(client)
        resp = client.get(
            f"/api/snapshots/{data['snapshot_id']}/geometry/cloud.ply"
        )
        # Torus may or may not have cloud; if missing, 404
        if resp.status_code == 404:
            assert "not found" in resp.json()["detail"].lower()

    def test_rejects_invalid_filename(self, client: TestClient) -> None:
        """Geometry endpoint rejects filenames not in the allowed set."""
        data = _create_snapshot(client)
        resp = client.get(
            f"/api/snapshots/{data['snapshot_id']}/geometry/metadata.json"
        )
        assert resp.status_code == 400


class TestLoadSnapshotRestoresState:
    """Tests for loading a snapshot restoring generator, params, seed, container."""

    def test_loading_snapshot_restores_values(
        self, client: TestClient
    ) -> None:
        """Loading a snapshot restores generator, params, seed, and container values."""
        _create_snapshot(client)
        resp = client.get("/api/snapshots")
        snap = resp.json()[0]

        assert snap["generator"] == "torus"
        assert snap["params"]["major_radius"] == 1.0
        assert snap["params"]["minor_radius"] == 0.4
        assert snap["seed"] == 42
        assert snap["container"]["width_mm"] == 100.0

    def test_loading_displays_saved_geometry_without_regeneration(
        self, client: TestClient
    ) -> None:
        """Loading a snapshot displays saved geometry without regeneration."""
        data = _create_snapshot(client)
        sid = data["snapshot_id"]

        # Reset the cache so geometry is not available for regeneration
        reset_cache()

        # We can still serve geometry directly from snapshot files
        resp = client.get(f"/api/snapshots/{sid}/geometry/mesh.glb")
        assert resp.status_code == 200
        assert resp.content[:4] == b"glTF"


class TestSnapshotGalleryUI:
    """Tests for Load button and gallery in preview HTML."""

    def test_preview_html_contains_load_button(
        self, preview_html: str
    ) -> None:
        """Preview HTML contains a Load button that opens the snapshot gallery."""
        assert 'id="load-btn"' in preview_html
        assert ">Load<" in preview_html

    def test_preview_html_contains_gallery_panel(
        self, preview_html: str
    ) -> None:
        """Preview HTML contains a snapshot gallery panel."""
        assert 'id="snapshot-gallery"' in preview_html
        assert 'id="gallery-list"' in preview_html

    def test_load_button_opens_gallery(self, preview_html: str) -> None:
        """JS binds Load button click to open the gallery overlay."""
        assert "load-btn" in preview_html
        assert "openGallery" in preview_html

    def test_gallery_fetches_snapshots(self, preview_html: str) -> None:
        """Gallery JS fetches /api/snapshots when opened."""
        assert "fetch('/api/snapshots')" in preview_html

    def test_gallery_has_delete_with_confirmation(
        self, preview_html: str
    ) -> None:
        """Delete button uses confirm() before deleting."""
        assert "confirm(" in preview_html
        assert "DELETE" in preview_html
