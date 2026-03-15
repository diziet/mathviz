"""Tests for the Lock Camera toggle in the preview UI."""

from typing import Generator

import pytest
from fastapi.testclient import TestClient

from mathviz.core.generator import register
from mathviz.generators.parametric.torus import TorusGenerator
from mathviz.preview.server import app, reset_cache, set_served_file


def _ensure_torus_registered() -> None:
    """Re-register the torus generator if missing from the registry."""
    import mathviz.core.generator as gen_mod

    if "torus" not in gen_mod._alias_map:
        gen_mod._discovered = True
        register(TorusGenerator)


@pytest.fixture(autouse=True)
def _ensure_generators() -> Generator[None, None, None]:
    """Ensure generators are registered and state is clean."""
    _ensure_torus_registered()
    reset_cache()
    set_served_file(None)
    yield
    reset_cache()
    set_served_file(None)


@pytest.fixture
def client() -> TestClient:
    """Return a FastAPI test client."""
    return TestClient(app)


@pytest.fixture
def html(client: TestClient) -> str:
    """Return the preview HTML content."""
    resp = client.get("/")
    assert resp.status_code == 200
    return resp.text


class TestLockCamera:
    """Tests for the Lock Camera checkbox in the preview HTML."""

    def test_html_contains_lock_camera_checkbox(self, html: str) -> None:
        """Preview HTML contains a Lock Camera checkbox with id lock-camera."""
        assert 'id="lock-camera"' in html
        assert 'type="checkbox"' in html
        assert "Lock Camera" in html

    def test_state_includes_camera_locked(self, html: str) -> None:
        """JS state object includes cameraLocked initialized to false."""
        assert "cameraLocked: false" in html

    def test_fit_camera_gated_by_camera_locked_in_display_mesh(
        self, html: str
    ) -> None:
        """fitCamera is not called during regeneration when cameraLocked is true."""
        assert "if (!state.cameraLocked)" in html
        assert "fitCamera" in html

    def test_fit_camera_gated_in_display_cloud(self, html: str) -> None:
        """displayCloud gates fitCamera behind cameraLocked check."""
        assert "!state.cameraLocked" in html

    def test_reset_view_works_regardless_of_lock(self, html: str) -> None:
        """Reset View button calls fitCamera without checking cameraLocked."""
        # resetView function should NOT check cameraLocked
        assert "function resetView()" in html
        # The resetView body calls fitCamera unconditionally
        assert "fitCamera(active)" in html

    def test_checkbox_toggles_state(self, html: str) -> None:
        """Toggling the checkbox updates state.cameraLocked."""
        assert "state.cameraLocked = e.target.checked" in html

    def test_controls_target_preserved_when_locked(self, html: str) -> None:
        """OrbitControls target is saved and restored when camera is locked."""
        assert "controls.target.clone()" in html
        assert "controls.target.copy(saved.target)" in html

    def test_camera_position_preserved_when_locked(self, html: str) -> None:
        """Camera position is saved and restored alongside controls target."""
        assert "camera.position.clone()" in html
        assert "camera.position.copy(saved.position)" in html

    def test_save_restore_helpers_exist(self, html: str) -> None:
        """DRY helpers saveCameraIfLocked and restoreCameraIfSaved exist."""
        assert "function saveCameraIfLocked()" in html
        assert "function restoreCameraIfSaved(saved)" in html

    def test_bounding_box_not_gated_by_camera_lock(self, html: str) -> None:
        """addBoundingBox is called regardless of cameraLocked state."""
        # In displayMesh, addBoundingBox should be outside the lock guard
        mesh_fn_start = html.index("async function displayMesh")
        mesh_fn_end = html.index("async function displayMesh") + 400
        mesh_fn = html[mesh_fn_start:mesh_fn_end]
        bbox_pos = mesh_fn.index("addBoundingBox")
        lock_pos = mesh_fn.index("if (!state.cameraLocked)")
        # addBoundingBox should come before the lock guard
        assert bbox_pos < lock_pos

    def test_lock_camera_near_other_options(self, html: str) -> None:
        """Lock Camera checkbox is in the Options section with other toggles."""
        bbox_pos = html.index('id="show-bbox"')
        lock_pos = html.index('id="lock-camera"')
        assert lock_pos > bbox_pos
