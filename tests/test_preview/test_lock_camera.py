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

    def test_click_updates_dataset_and_state(self, html: str) -> None:
        """Click handler updates btn.dataset.mode and state.cameraLocked."""
        assert "btn.dataset.mode = next" in html
        assert "state.cameraLocked = next" in html

    def test_labels_for_all_modes(self, html: str) -> None:
        """Labels are defined for all three modes."""
        assert "Render Lock" in html
        assert "Full Lock" in html
        assert "Free" in html

    def test_invalid_state_fallback(self, html: str) -> None:
        """Invalid cameraLocked value falls back to render via nullish coalescing."""
        assert "cycle[state.cameraLocked] ?? 'render'" in html


class TestClickHandlerBehavior:
    """Tests for controls/cursor behavior in the click handler."""

    def test_controls_enabled_gated_by_full(self, html: str) -> None:
        """Controls disabled only in full lock mode."""
        assert "controls.enabled = next !== 'full'" in html

    def test_cursor_not_allowed_in_full_only(self, html: str) -> None:
        """Cursor set to not-allowed only in full mode, cleared otherwise."""
        assert "next === 'full' ? 'not-allowed' : ''" in html
        assert "renderer.domElement.style.cursor" in html


class TestSetupCameraForObject:
    """Tests for the lock-aware camera setup helper."""

    def test_fit_camera_guard(self, html: str) -> None:
        """fitCamera runs when mode is off or on first render."""
        fn = _extract_fn_body(html, "function setupCameraForObject")
        assert "state.cameraLocked === 'off' || _firstRender" in fn
        assert "fitCamera(object3d)" in fn

    def test_clipping_planes_in_else_branch(self, html: str) -> None:
        """Clipping planes update when camera is locked (not off)."""
        fn = _extract_fn_body(html, "function setupCameraForObject")
        assert "updateClippingPlanes(object3d)" in fn

    def test_bounding_box_always_first(self, html: str) -> None:
        """addBoundingBox is called before the lock guard."""
        fn = _extract_fn_body(html, "function setupCameraForObject")
        bbox_pos = fn.index("addBoundingBox")
        lock_pos = fn.index("state.cameraLocked")
        assert bbox_pos < lock_pos


class TestFirstRenderFitsCamera:
    """Tests for the first-render override that ensures initial framing."""

    def test_first_render_flag_exists(self, html: str) -> None:
        """A _firstRender flag forces fitCamera on initial load."""
        assert "let _firstRender = true" in html

    def test_first_render_cleared_after_use(self, html: str) -> None:
        """_firstRender is set to false after the first fitCamera call."""
        fn = _extract_fn_body(html, "function setupCameraForObject")
        assert "_firstRender = false" in fn


class TestCameraPreservation:
    """Tests for camera save/restore across regenerations."""

    def test_camera_position_saved_and_restored(self, html: str) -> None:
        """Camera position is cloned before and copied after regeneration."""
        assert "camera.position.clone()" in html
        assert "camera.position.copy(saved.position)" in html

    def test_controls_target_saved_and_restored(self, html: str) -> None:
        """OrbitControls target is saved and restored."""
        assert "controls.target.clone()" in html
        assert "controls.target.copy(saved.target)" in html

    def test_save_returns_null_when_off(self, html: str) -> None:
        """saveCameraIfLocked returns null only when mode is off."""
        fn = _extract_fn_body(html, "function saveCameraIfLocked()")
        assert "state.cameraLocked === 'off'" in fn

    def test_helpers_exist(self, html: str) -> None:
        """DRY helpers saveCameraIfLocked and restoreCameraIfSaved exist."""
        assert "function saveCameraIfLocked()" in html
        assert "function restoreCameraIfSaved(saved)" in html


class TestResetView:
    """Tests for Reset View behavior across all modes."""

    def test_reset_view_calls_fit_camera(self, html: str) -> None:
        """Reset View calls fitCamera regardless of lock mode."""
        fn = _extract_fn_body(html, "function resetView()")
        assert "fitCamera(active)" in fn

    def test_reset_view_handles_full_lock(self, html: str) -> None:
        """Reset View re-enables controls temporarily for full lock."""
        fn = _extract_fn_body(html, "function resetView()")
        assert "const wasFullLock = state.cameraLocked === 'full'" in fn
        assert "if (wasFullLock) controls.enabled = true" in fn
        assert "if (wasFullLock) controls.enabled = false" in fn

    def test_reset_view_uses_try_finally(self, html: str) -> None:
        """resetView uses try/finally to ensure controls re-disabled."""
        fn = _extract_fn_body(html, "function resetView()")
        assert "try {" in fn
        assert "} finally {" in fn


class TestExistingBehaviorPreserved:
    """Tests for features that must still work."""

    def test_clipping_planes_function_exists(self, html: str) -> None:
        """updateClippingPlanes function exists and updates projection."""
        assert "function updateClippingPlanes(object3d)" in html

    def test_concurrent_generate_uses_generation_id(self, html: str) -> None:
        """Concurrent displayGenerateResult calls serialized via generationId."""
        assert "state.generationId += 1" in html
        assert "const myGenerationId = state.generationId" in html
        assert "if (myGenerationId !== state.generationId) return" in html

    def test_state_includes_generation_id(self, html: str) -> None:
        """State includes generationId for race prevention."""
        assert "generationId: 0" in html
