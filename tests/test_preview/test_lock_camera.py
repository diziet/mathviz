"""Tests for the Lock Camera three-state toggle in the preview UI."""

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


def _extract_fn_body(html: str, signature: str, length: int = 500) -> str:
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


class TestLockCameraButton:
    """Tests for the Lock Camera button existence and default state."""

    def test_button_exists_with_render_default(self, html: str) -> None:
        """Lock Camera button exists and defaults to render state."""
        assert 'id="lock-camera"' in html
        assert 'data-mode="render"' in html
        assert "Render Lock" in html

    def test_state_initialized_to_render(self, html: str) -> None:
        """JS state object initializes cameraLocked to 'render'."""
        assert "cameraLocked: 'render'" in html

    def test_button_near_other_options(self, html: str) -> None:
        """Lock Camera button is in the Options section with other toggles."""
        bbox_pos = html.index('id="show-bbox"')
        lock_pos = html.index('id="lock-camera"')
        assert lock_pos > bbox_pos


class TestLockCameraCycling:
    """Tests for the three-state cycling behavior."""

    def test_cycle_map_exists(self, html: str) -> None:
        """Click handler cycles render -> full -> off -> render."""
        assert "render: 'full'" in html
        assert "full: 'off'" in html
        assert "off: 'render'" in html

    def test_click_updates_dataset_mode(self, html: str) -> None:
        """Click handler updates btn.dataset.mode."""
        assert "btn.dataset.mode = next" in html

    def test_click_updates_state(self, html: str) -> None:
        """Click handler updates state.cameraLocked."""
        assert "state.cameraLocked = next" in html

    def test_labels_for_all_modes(self, html: str) -> None:
        """Labels are defined for all three modes."""
        assert "Render Lock" in html
        assert "Full Lock" in html
        assert "Free" in html


class TestOffMode:
    """Tests for off mode: fitCamera called, controls enabled."""

    def test_fit_camera_called_when_off(self, html: str) -> None:
        """fitCamera is called on regeneration when mode is off."""
        fn = _extract_fn_body(html, "function setupCameraForObject")
        assert "state.cameraLocked === 'off'" in fn
        assert "fitCamera(object3d)" in fn

    def test_controls_enabled_when_off(self, html: str) -> None:
        """Controls are enabled when mode is off."""
        # The handler sets controls.enabled = (next !== 'full')
        # When next is 'off', this is true
        assert "controls.enabled = next !== 'full'" in html


class TestRenderLockMode:
    """Tests for render lock: no fitCamera, controls remain enabled."""

    def test_no_fit_camera_in_render_mode(self, html: str) -> None:
        """fitCamera is NOT called on regeneration when mode is render."""
        fn = _extract_fn_body(html, "function setupCameraForObject")
        # fitCamera only called when cameraLocked === 'off'
        assert "state.cameraLocked === 'off'" in fn

    def test_controls_enabled_in_render_mode(self, html: str) -> None:
        """Controls remain enabled in render mode (not 'full')."""
        assert "controls.enabled = next !== 'full'" in html

    def test_camera_position_preserved(self, html: str) -> None:
        """Camera position is saved and restored across regeneration."""
        assert "camera.position.clone()" in html
        assert "camera.position.copy(saved.position)" in html

    def test_controls_target_preserved(self, html: str) -> None:
        """OrbitControls target is saved and restored."""
        assert "controls.target.clone()" in html
        assert "controls.target.copy(saved.target)" in html


class TestFullLockMode:
    """Tests for full lock: no fitCamera, controls disabled, cursor locked."""

    def test_no_fit_camera_in_full_mode(self, html: str) -> None:
        """fitCamera is NOT called on regeneration when mode is full."""
        fn = _extract_fn_body(html, "function setupCameraForObject")
        assert "state.cameraLocked === 'off'" in fn

    def test_controls_disabled_in_full_mode(self, html: str) -> None:
        """Controls disabled when mode is full."""
        assert "controls.enabled = next !== 'full'" in html

    def test_cursor_not_allowed_in_full_mode(self, html: str) -> None:
        """Cursor set to not-allowed when mode is full."""
        assert "next === 'full' ? 'not-allowed' : ''" in html
        assert "renderer.domElement.style.cursor" in html


class TestResetView:
    """Tests for Reset View behavior across all modes."""

    def test_reset_view_works_in_all_modes(self, html: str) -> None:
        """Reset View temporarily enables controls for full lock."""
        fn = _extract_fn_body(html, "function resetView()")
        assert "fitCamera(active)" in fn

    def test_reset_view_handles_full_lock(self, html: str) -> None:
        """Reset View re-enables controls temporarily for full lock."""
        fn = _extract_fn_body(html, "function resetView()")
        assert "const wasFullLock = state.cameraLocked === 'full'" in fn
        assert "if (wasFullLock) controls.enabled = true" in fn
        assert "if (wasFullLock) controls.enabled = false" in fn

    def test_reset_view_uses_try_finally(self, html: str) -> None:
        """resetView uses try/finally to ensure controls re-disabled on error."""
        fn = _extract_fn_body(html, "function resetView()")
        assert "try {" in fn
        assert "} finally {" in fn


class TestSaveRestoreHelpers:
    """Tests for saveCameraIfLocked / restoreCameraIfSaved."""

    def test_helpers_exist(self, html: str) -> None:
        """DRY helpers saveCameraIfLocked and restoreCameraIfSaved exist."""
        assert "function saveCameraIfLocked()" in html
        assert "function restoreCameraIfSaved(saved)" in html

    def test_save_activates_in_render_and_full(self, html: str) -> None:
        """saveCameraIfLocked returns null only when mode is off."""
        fn = _extract_fn_body(html, "function saveCameraIfLocked()")
        assert "state.cameraLocked === 'off'" in fn

    def test_restore_copies_position_and_target(self, html: str) -> None:
        """restoreCameraIfSaved restores camera position and target."""
        fn = _extract_fn_body(html, "function restoreCameraIfSaved")
        assert "controls.target.copy(saved.target)" in fn
        assert "camera.position.copy(saved.position)" in fn


class TestFullToOffTransition:
    """Tests for transitioning from full lock to off."""

    def test_controls_re_enabled_on_transition(self, html: str) -> None:
        """Transitioning from full -> off re-enables controls."""
        # When cycling full -> off, next !== 'full' so controls.enabled = true
        assert "controls.enabled = next !== 'full'" in html

    def test_cursor_reset_on_transition(self, html: str) -> None:
        """Transitioning from full -> off resets cursor."""
        # When next is 'off', cursor becomes ''
        assert "next === 'full' ? 'not-allowed' : ''" in html


class TestExistingBehaviorPreserved:
    """Tests for features that must still work."""

    def test_setup_camera_helper_exists(self, html: str) -> None:
        """setupCameraForObject helper encapsulates lock-aware camera logic."""
        fn = _extract_fn_body(html, "function setupCameraForObject")
        assert "addBoundingBox(object3d)" in fn
        assert "fitCamera(object3d)" in fn
        assert "updateClippingPlanes(object3d)" in fn

    def test_clipping_planes_update_when_locked(self, html: str) -> None:
        """Near/far clipping planes update even when locked."""
        assert "function updateClippingPlanes(object3d)" in html
        fn = _extract_fn_body(html, "function setupCameraForObject")
        assert "updateClippingPlanes(object3d)" in fn

    def test_concurrent_generate_calls_use_generation_id(
        self, html: str
    ) -> None:
        """Concurrent displayGenerateResult calls serialized via generationId."""
        assert "state.generationId += 1" in html
        assert "const myGenerationId = state.generationId" in html
        assert "if (myGenerationId !== state.generationId) return" in html

    def test_state_includes_generation_id(self, html: str) -> None:
        """State includes generationId for race prevention."""
        assert "generationId: 0" in html
