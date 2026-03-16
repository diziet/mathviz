"""Tests for snapshot save/load with ui_state, gallery display, and save-after-load."""

import json
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

from tests.test_preview.conftest import make_snapshot_request as _make_snapshot_request


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


SAMPLE_UI_STATE: dict[str, Any] = {
    "camera": {
        "position": {"x": 1.5, "y": 2.0, "z": 3.0},
        "target": {"x": 0.0, "y": 0.0, "z": 0.0},
        "zoom": 1.0,
    },
    "view_mode": "shaded",
    "stretch": {"x": 1.5, "y": 0.8, "z": 2.0},
    "camera_lock": "render",
    "show_bbox": True,
    "show_axes": True,
    "light_bg": False,
    "point_size": 0.2,
}


def _create_snapshot_with_ui_state(
    client: TestClient,
    ui_state: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Generate torus, save snapshot with ui_state, return response data."""
    gid = _generate_torus(client)
    req = _make_snapshot_request(gid)
    if ui_state is not None:
        req["ui_state"] = ui_state
    resp = client.post("/api/snapshots", json=req)
    assert resp.status_code == 200
    return resp.json()


class TestSaveAfterLoad:
    """Save button should be enabled after loading a snapshot."""

    def test_save_enabled_after_load_and_regenerate(
        self, client: TestClient
    ) -> None:
        """After loading a snapshot and re-generating, save succeeds."""
        data = _create_snapshot_with_ui_state(client, SAMPLE_UI_STATE)
        sid = data["snapshot_id"]

        # Verify the snapshot has geometry_id in listing
        resp = client.get("/api/snapshots")
        assert resp.status_code == 200
        snaps = resp.json()
        snap = next(s for s in snaps if s["snapshot_id"] == sid)
        assert snap.get("geometry_id") is not None

        # The geometry_id from the original generation is stored in metadata.
        # After loading, the UI uses this to enable the save button.
        # Simulate re-generating by calling generate again
        gid = _generate_torus(client)
        req = _make_snapshot_request(gid)
        req["ui_state"] = SAMPLE_UI_STATE
        resp = client.post("/api/snapshots", json=req)
        assert resp.status_code == 200

    def test_snapshot_listing_includes_geometry_id(
        self, client: TestClient
    ) -> None:
        """Snapshot listing includes geometry_id for save-after-load."""
        _create_snapshot_with_ui_state(client)
        resp = client.get("/api/snapshots")
        assert resp.status_code == 200
        snap = resp.json()[0]
        assert "geometry_id" in snap
        assert snap["geometry_id"] is not None


class TestSnapshotUiState:
    """Tests for ui_state in snapshot save payload."""

    def test_save_payload_includes_ui_state(
        self, client: TestClient, tmp_path: Path
    ) -> None:
        """Snapshot save includes full ui_state in metadata."""
        data = _create_snapshot_with_ui_state(client, SAMPLE_UI_STATE)
        meta_path = tmp_path / data["snapshot_id"] / "metadata.json"
        metadata = json.loads(meta_path.read_text())
        assert "ui_state" in metadata
        ui = metadata["ui_state"]
        assert ui["view_mode"] == "shaded"
        assert ui["stretch"] == {"x": 1.5, "y": 0.8, "z": 2.0}
        assert ui["camera_lock"] == "render"
        assert ui["show_bbox"] is True
        assert ui["show_axes"] is True
        assert ui["light_bg"] is False
        assert ui["point_size"] == 0.2
        assert ui["camera"]["position"]["x"] == 1.5
        assert ui["camera"]["target"]["y"] == 0.0
        assert ui["camera"]["zoom"] == 1.0

    def test_load_returns_ui_state(
        self, client: TestClient
    ) -> None:
        """Snapshot listing returns ui_state for restoring on load."""
        _create_snapshot_with_ui_state(client, SAMPLE_UI_STATE)
        resp = client.get("/api/snapshots")
        snap = resp.json()[0]
        assert "ui_state" in snap
        assert snap["ui_state"]["view_mode"] == "shaded"
        assert snap["ui_state"]["camera"]["position"]["x"] == 1.5

    def test_round_trip_ui_state_matches(
        self, client: TestClient, tmp_path: Path
    ) -> None:
        """Save then load: all ui_state fields match original."""
        _create_snapshot_with_ui_state(client, SAMPLE_UI_STATE)
        resp = client.get("/api/snapshots")
        snap = resp.json()[0]
        loaded_ui = snap["ui_state"]

        assert loaded_ui["camera"] == SAMPLE_UI_STATE["camera"]
        assert loaded_ui["view_mode"] == SAMPLE_UI_STATE["view_mode"]
        assert loaded_ui["stretch"] == SAMPLE_UI_STATE["stretch"]
        assert loaded_ui["camera_lock"] == SAMPLE_UI_STATE["camera_lock"]
        assert loaded_ui["show_bbox"] == SAMPLE_UI_STATE["show_bbox"]
        assert loaded_ui["show_axes"] == SAMPLE_UI_STATE["show_axes"]
        assert loaded_ui["light_bg"] == SAMPLE_UI_STATE["light_bg"]
        assert loaded_ui["point_size"] == SAMPLE_UI_STATE["point_size"]

    def test_save_without_ui_state_works(
        self, client: TestClient
    ) -> None:
        """Saving without ui_state still succeeds (backwards compat)."""
        data = _create_snapshot_with_ui_state(client, ui_state=None)
        assert "snapshot_id" in data

    def test_load_snapshot_without_ui_state_graceful(
        self, client: TestClient, tmp_path: Path
    ) -> None:
        """Loading a snapshot saved before ui_state existed works gracefully."""
        data = _create_snapshot_with_ui_state(client)
        # Remove ui_state from metadata to simulate old snapshot
        meta_path = tmp_path / data["snapshot_id"] / "metadata.json"
        metadata = json.loads(meta_path.read_text())
        metadata.pop("ui_state", None)
        meta_path.write_text(json.dumps(metadata))

        resp = client.get("/api/snapshots")
        snap = resp.json()[0]
        # ui_state should be absent or None, not an error
        assert snap.get("ui_state") is None


class TestGalleryParamsDisplay:
    """Tests for gallery card parameter display."""

    def test_gallery_has_params_grid_class(self, preview_html: str) -> None:
        """Gallery card uses snap-params-grid for structured display."""
        assert "snap-params-grid" in preview_html

    def test_gallery_shows_seed(self, preview_html: str) -> None:
        """Gallery card builder includes seed in the params grid."""
        assert "addRow('seed'" in preview_html

    def test_gallery_shows_container_dimensions(
        self, preview_html: str
    ) -> None:
        """Gallery card builder includes container dimensions."""
        assert "addRow('container'" in preview_html

    def test_snapshot_listing_has_seed_and_container(
        self, client: TestClient
    ) -> None:
        """Snapshot listing data includes seed and container for display."""
        _create_snapshot_with_ui_state(client)
        resp = client.get("/api/snapshots")
        snap = resp.json()[0]
        assert snap["seed"] == 42
        assert "container" in snap
        assert snap["container"]["width_mm"] == 100.0
        assert snap["container"]["height_mm"] == 100.0

    def test_gallery_card_shows_view_mode_from_ui_state(
        self, preview_html: str
    ) -> None:
        """Gallery card shows view mode when ui_state is present."""
        assert "snap.ui_state.view_mode" in preview_html


class TestRestoreUiStateHtml:
    """Tests for restoreUiState function in HTML."""

    def test_html_contains_restore_ui_state(self, preview_html: str) -> None:
        """HTML contains restoreUiState function."""
        assert "function restoreUiState" in preview_html

    def test_html_contains_capture_ui_state(self, preview_html: str) -> None:
        """HTML contains captureUiState function for save."""
        assert "function captureUiState" in preview_html

    def test_save_payload_includes_ui_state_call(
        self, preview_html: str
    ) -> None:
        """Save button handler includes ui_state in POST body."""
        assert "ui_state: captureUiState()" in preview_html

    def test_load_calls_restore_ui_state(self, preview_html: str) -> None:
        """loadSnapshot calls restoreUiState after loading geometry."""
        assert "restoreUiState(snap.ui_state)" in preview_html

    def test_load_preserves_geometry_id(self, preview_html: str) -> None:
        """loadSnapshot preserves geometry_id from snapshot for save."""
        assert "snap.geometry_id" in preview_html
        assert "state.geometryId = snap.geometry_id" in preview_html

    def test_html_has_stretch_controls(self, preview_html: str) -> None:
        """HTML contains per-axis stretch range sliders."""
        assert 'id="stretch-x"' in preview_html
        assert 'id="stretch-y"' in preview_html
        assert 'id="stretch-z"' in preview_html

    def test_html_has_show_axes_toggle(self, preview_html: str) -> None:
        """HTML contains show-axes checkbox."""
        assert 'id="show-axes"' in preview_html

    def test_capture_includes_stretch(self, preview_html: str) -> None:
        """captureUiState includes stretch values."""
        assert "stretch:" in preview_html
        assert "state.stretch.x" in preview_html

    def test_capture_includes_show_axes(self, preview_html: str) -> None:
        """captureUiState includes show_axes value."""
        assert "show_axes:" in preview_html

    def test_restore_applies_stretch(self, preview_html: str) -> None:
        """restoreUiState applies stretch to geometry."""
        assert "applyStretch()" in preview_html
