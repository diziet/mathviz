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

    def test_add_bounding_box_skipped_when_locked(self, html: str) -> None:
        """addBoundingBox is inside the !state.cameraLocked guard in displayMesh."""
        mesh_fn_start = html.index("async function displayMesh")
        mesh_fn_end = mesh_fn_start + 400
        mesh_fn = html[mesh_fn_start:mesh_fn_end]
        # addBoundingBox should be inside the lock guard
        lock_pos = mesh_fn.index("if (!state.cameraLocked)")
        bbox_pos = mesh_fn.index("addBoundingBox")
        assert bbox_pos > lock_pos

    def test_add_bounding_box_skipped_in_display_cloud(self, html: str) -> None:
        """addBoundingBox is inside the lock guard in displayCloud too."""
        cloud_fn_start = html.index("function displayCloud")
        cloud_fn_end = cloud_fn_start + 400
        cloud_fn = html[cloud_fn_start:cloud_fn_end]
        lock_pos = cloud_fn.index("if (!state.cameraLocked)")
        bbox_pos = cloud_fn.index("addBoundingBox")
        assert bbox_pos > lock_pos

    def test_clipping_planes_update_when_locked(self, html: str) -> None:
        """Near/far clipping planes update even when locked."""
        assert "function updateClippingPlanes(object3d)" in html
        # Called in displayMesh else branch
        mesh_fn_start = html.index("async function displayMesh")
        mesh_fn_end = mesh_fn_start + 400
        mesh_fn = html[mesh_fn_start:mesh_fn_end]
        assert "updateClippingPlanes(group)" in mesh_fn
        # Called in displayCloud else branch
        cloud_fn_start = html.index("function displayCloud")
        cloud_fn_end = cloud_fn_start + 400
        cloud_fn = html[cloud_fn_start:cloud_fn_end]
        assert "updateClippingPlanes(points)" in cloud_fn

    def test_clipping_planes_updates_projection(self, html: str) -> None:
        """updateClippingPlanes sets near/far and calls updateProjectionMatrix."""
        fn_start = html.index("function updateClippingPlanes")
        fn_end = fn_start + 400
        fn_body = html[fn_start:fn_end]
        assert "camera.near" in fn_body
        assert "camera.far" in fn_body
        assert "camera.updateProjectionMatrix()" in fn_body

    def test_concurrent_generate_calls_use_generation_id(self, html: str) -> None:
        """Concurrent displayGenerateResult calls are serialized via generationId."""
        assert "state.generationId += 1" in html
        assert "const myGenerationId = state.generationId" in html
        assert "if (myGenerationId !== state.generationId) return" in html

    def test_state_includes_generation_guard(self, html: str) -> None:
        """State includes isGenerating and generationId for race prevention."""
        assert "isGenerating: false" in html
        assert "generationId: 0" in html

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
        fn_start = html.index("function resetView()")
        fn_end = fn_start + 300
        fn_body = html[fn_start:fn_end]
        assert "const wasLocked = state.cameraLocked" in fn_body
        assert "if (wasLocked) controls.enabled = true" in fn_body
        assert "fitCamera(active)" in fn_body
        assert "if (wasLocked) controls.enabled = false" in fn_body

    def test_save_restore_helpers_exist(self, html: str) -> None:
        """DRY helpers saveCameraIfLocked and restoreCameraIfSaved exist."""
        assert "function saveCameraIfLocked()" in html
        assert "function restoreCameraIfSaved(saved)" in html

    def test_cursor_not_allowed_when_locked(self, html: str) -> None:
        """Canvas cursor set to not-allowed when camera is locked."""
        assert "not-allowed" in html
        assert "renderer.domElement.style.cursor" in html

    def test_fit_camera_gated_by_camera_locked_in_display_mesh(
        self, html: str
    ) -> None:
        """fitCamera is not called during regeneration when cameraLocked is true."""
        assert "if (!state.cameraLocked)" in html
        assert "fitCamera" in html

    def test_fit_camera_gated_in_display_cloud(self, html: str) -> None:
        """displayCloud gates fitCamera behind cameraLocked check."""
        assert "!state.cameraLocked" in html

    def test_lock_camera_near_other_options(self, html: str) -> None:
        """Lock Camera checkbox is in the Options section with other toggles."""
        bbox_pos = html.index('id="show-bbox"')
        lock_pos = html.index('id="lock-camera"')
        assert lock_pos > bbox_pos
