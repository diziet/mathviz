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


def _extract_fn_body(html: str, signature: str, length: int = 400) -> str:
    """Extract a JS function body from HTML by its signature."""
    start = html.index(signature)
    return html[start : start + length]


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

    def test_toggling_lock_sets_controls_enabled_false(self, html: str) -> None:
        """Toggling Lock Camera on sets controls.enabled to false."""
        assert "controls.enabled = !e.target.checked" in html

    def test_toggling_lock_off_sets_controls_enabled_true(self, html: str) -> None:
        """Toggling Lock Camera off re-enables controls (same expression)."""
        # When e.target.checked is false, !false = true
        assert "controls.enabled = !e.target.checked" in html

    def test_checkbox_toggles_state(self, html: str) -> None:
        """Toggling the checkbox updates state.cameraLocked."""
        assert "state.cameraLocked = e.target.checked" in html

    def test_setup_camera_helper_exists(self, html: str) -> None:
        """setupCameraForObject helper encapsulates lock-aware camera logic."""
        fn = _extract_fn_body(html, "function setupCameraForObject")
        assert "addBoundingBox(object3d)" in fn
        assert "if (!state.cameraLocked)" in fn
        assert "fitCamera(object3d)" in fn
        assert "updateClippingPlanes(object3d)" in fn

    def test_display_mesh_uses_setup_helper(self, html: str) -> None:
        """displayMesh delegates to setupCameraForObject."""
        fn = _extract_fn_body(html, "async function displayMesh")
        assert "setupCameraForObject(group)" in fn

    def test_display_cloud_uses_setup_helper(self, html: str) -> None:
        """displayCloud delegates to setupCameraForObject."""
        fn = _extract_fn_body(html, "function displayCloud")
        assert "setupCameraForObject(points)" in fn

    def test_bounding_box_always_added(self, html: str) -> None:
        """addBoundingBox is called regardless of lock state via helper."""
        fn = _extract_fn_body(html, "function setupCameraForObject")
        bbox_pos = fn.index("addBoundingBox")
        lock_pos = fn.index("if (!state.cameraLocked)")
        # addBoundingBox is called before the lock guard
        assert bbox_pos < lock_pos

    def test_clipping_planes_update_when_locked(self, html: str) -> None:
        """Near/far clipping planes update even when locked."""
        assert "function updateClippingPlanes(object3d)" in html
        fn = _extract_fn_body(html, "function setupCameraForObject")
        assert "updateClippingPlanes(object3d)" in fn

    def test_clipping_planes_updates_projection(self, html: str) -> None:
        """updateClippingPlanes sets near/far and calls updateProjectionMatrix."""
        fn = _extract_fn_body(html, "function updateClippingPlanes")
        assert "camera.near" in fn
        assert "camera.far" in fn
        assert "camera.updateProjectionMatrix()" in fn

    def test_concurrent_generate_calls_use_generation_id(self, html: str) -> None:
        """Concurrent displayGenerateResult calls are serialized via generationId."""
        assert "state.generationId += 1" in html
        assert "const myGenerationId = state.generationId" in html
        assert "if (myGenerationId !== state.generationId) return" in html

    def test_state_includes_generation_id(self, html: str) -> None:
        """State includes generationId for race prevention."""
        assert "generationId: 0" in html

    def test_no_dead_is_generating_state(self, html: str) -> None:
        """isGenerating was removed as dead state."""
        assert "isGenerating:" not in html

    def test_camera_position_identical_across_regeneration(self, html: str) -> None:
        """Camera position is saved before and restored after regeneration."""
        assert "camera.position.clone()" in html
        assert "camera.position.copy(saved.position)" in html

    def test_controls_target_preserved_when_locked(self, html: str) -> None:
        """OrbitControls target is saved and restored when camera is locked."""
        assert "controls.target.clone()" in html
        assert "controls.target.copy(saved.target)" in html

    def test_reset_view_works_when_locked(self, html: str) -> None:
        """Reset View temporarily re-enables controls when camera is locked."""
        fn = _extract_fn_body(html, "function resetView()")
        assert "const wasLocked = state.cameraLocked" in fn
        assert "if (wasLocked) controls.enabled = true" in fn
        assert "fitCamera(active)" in fn
        assert "if (wasLocked) controls.enabled = false" in fn

    def test_reset_view_uses_try_finally(self, html: str) -> None:
        """resetView uses try/finally to ensure controls re-disabled on error."""
        fn = _extract_fn_body(html, "function resetView()")
        assert "try {" in fn
        assert "} finally {" in fn

    def test_save_restore_helpers_exist(self, html: str) -> None:
        """DRY helpers saveCameraIfLocked and restoreCameraIfSaved exist."""
        assert "function saveCameraIfLocked()" in html
        assert "function restoreCameraIfSaved(saved)" in html

    def test_cursor_not_allowed_when_locked(self, html: str) -> None:
        """Canvas cursor set to not-allowed when camera is locked."""
        assert "not-allowed" in html
        assert "renderer.domElement.style.cursor" in html

    def test_fit_camera_gated_by_camera_locked(self, html: str) -> None:
        """fitCamera is not called during regeneration when cameraLocked is true."""
        assert "if (!state.cameraLocked)" in html
        assert "fitCamera" in html

    def test_lock_camera_near_other_options(self, html: str) -> None:
        """Lock Camera checkbox is in the Options section with other toggles."""
        bbox_pos = html.index('id="show-bbox"')
        lock_pos = html.index('id="lock-camera"')
        assert lock_pos > bbox_pos
