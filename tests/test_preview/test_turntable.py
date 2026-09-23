"""Tests for turntable animation and export feature.

Verifies that the turntable toggle, speed slider and export controls are
present in the preview HTML, and that the related JS functions contain the
expected statements.
"""

import re
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
    """Register torus if missing; reset the cache and the served file before and after the test."""
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


def _get_html(client: TestClient) -> str:
    """Fetch the preview HTML page."""
    resp = client.get("/")
    assert resp.status_code == 200
    return resp.text


def _extract_js_function(html: str, func_name: str) -> str:
    """Extract a JS function body from the HTML, or fail."""
    match = re.search(
        rf"function {re.escape(func_name)}\(.*?^}}",
        html,
        re.MULTILINE | re.DOTALL,
    )
    assert match is not None, f"JS function {func_name!r} not found in HTML"
    return match.group(0)


def _extract_element(html: str, element_id: str) -> str:
    """Extract an HTML element tag by id, or fail."""
    match = re.search(rf'<[^>]*id="{re.escape(element_id)}"[^>]*>', html)
    assert match is not None, f"Element {element_id!r} not found in HTML"
    return match.group(0)


class TestTurntableToggle:
    """Tests that turntable toggle exists and defaults to off."""

    def test_turntable_toggle_exists(self, client: TestClient) -> None:
        """Preview HTML contains a turntable toggle checkbox."""
        html = _get_html(client)
        assert 'id="turntable-toggle"' in html

    def test_turntable_toggle_is_checkbox(self, client: TestClient) -> None:
        """Turntable toggle is a checkbox input."""
        html = _get_html(client)
        tag = _extract_element(html, "turntable-toggle")
        assert 'type="checkbox"' in tag

    def test_turntable_defaults_to_off(self, client: TestClient) -> None:
        """Turntable toggle is unchecked by default."""
        html = _get_html(client)
        tag = _extract_element(html, "turntable-toggle")
        assert "checked" not in tag

    def test_state_turntable_defaults_false(self, client: TestClient) -> None:
        """JavaScript state initializes turntable to false."""
        html = _get_html(client)
        assert "turntable: false" in html


class TestTurntableEnablesAutoRotate:
    """Tests that enabling turntable sets controls.autoRotate to true."""

    def test_set_turntable_sets_auto_rotate(self, client: TestClient) -> None:
        """setTurntable function sets controls.autoRotate."""
        html = _get_html(client)
        body = _extract_js_function(html, "setTurntable")
        assert "controls.autoRotate = enabled" in body

    def test_set_turntable_updates_state(self, client: TestClient) -> None:
        """setTurntable updates state.turntable."""
        html = _get_html(client)
        body = _extract_js_function(html, "setTurntable")
        assert "state.turntable = enabled" in body


class TestSpeedSlider:
    """Tests that speed slider adjusts controls.autoRotateSpeed."""

    def test_speed_slider_exists(self, client: TestClient) -> None:
        """Preview HTML contains a turntable speed slider."""
        html = _get_html(client)
        assert 'id="turntable-speed"' in html

    def test_speed_slider_is_range(self, client: TestClient) -> None:
        """Speed slider is a range input with correct min/max."""
        html = _get_html(client)
        tag = _extract_element(html, "turntable-speed")
        assert 'type="range"' in tag
        assert 'min="0.5"' in tag
        assert 'max="5"' in tag

    def test_speed_slider_default_value(self, client: TestClient) -> None:
        """Speed slider defaults to 1."""
        html = _get_html(client)
        tag = _extract_element(html, "turntable-speed")
        assert 'value="1"' in tag

    def test_speed_slider_disabled_by_default(self, client: TestClient) -> None:
        """Speed slider is disabled when turntable is off."""
        html = _get_html(client)
        tag = _extract_element(html, "turntable-speed")
        assert "disabled" in tag

    def test_set_turntable_speed_updates_auto_rotate(
        self, client: TestClient,
    ) -> None:
        """setTurntableSpeed sets controls.autoRotateSpeed."""
        html = _get_html(client)
        body = _extract_js_function(html, "setTurntableSpeed")
        assert "controls.autoRotateSpeed" in body

    def test_speed_label_exists(self, client: TestClient) -> None:
        """Speed label element exists."""
        html = _get_html(client)
        assert 'id="turntable-speed-label"' in html


class TestExportButtonVisibility:
    """Tests for the export section's default state and the export controls."""

    def test_export_section_hidden_by_default(self, client: TestClient) -> None:
        """Export section is not visible when turntable is off."""
        html = _get_html(client)
        tag = _extract_element(html, "turntable-export-section")
        assert 'class="visible"' not in tag

    def test_export_button_exists(self, client: TestClient) -> None:
        """Export button exists in the HTML."""
        html = _get_html(client)
        assert 'id="export-btn"' in html

    def test_set_turntable_shows_export(self, client: TestClient) -> None:
        """setTurntable's body refers to turntable-export-section and 'visible'."""
        html = _get_html(client)
        body = _extract_js_function(html, "setTurntable")
        assert "turntable-export-section" in body
        assert "visible" in body

    def test_export_format_select_exists(self, client: TestClient) -> None:
        """Export format selector exists, and the page contains GIF and WebM."""
        html = _get_html(client)
        assert 'id="export-format"' in html
        assert "GIF" in html
        assert "WebM" in html

    def test_export_resolution_select_exists(self, client: TestClient) -> None:
        html = _get_html(client)
        assert 'id="export-resolution"' in html


class TestDisablingTurntable:
    """Tests that disabling turntable stops rotation."""

    def test_set_turntable_false_stops_rotation(
        self, client: TestClient,
    ) -> None:
        """setTurntable(false) sets autoRotate to false."""
        html = _get_html(client)
        body = _extract_js_function(html, "setTurntable")
        assert "controls.autoRotate = enabled" in body

    def test_set_turntable_false_hides_export(
        self, client: TestClient,
    ) -> None:
        """setTurntable's body calls classList.remove('visible')."""
        html = _get_html(client)
        body = _extract_js_function(html, "setTurntable")
        assert "classList.remove('visible')" in body

    def test_speed_slider_disabled_when_off(
        self, client: TestClient,
    ) -> None:
        """setTurntable's body contains "disabled"."""
        html = _get_html(client)
        body = _extract_js_function(html, "setTurntable")
        assert "disabled" in body


class TestTurntableWithViewModes:
    """Tests that rotation goes through OrbitControls and renderFrameAtAngle's mode checks."""

    def test_turntable_uses_controls_auto_rotate(
        self, client: TestClient,
    ) -> None:
        """Turntable uses OrbitControls.autoRotate (view-mode agnostic)."""
        html = _get_html(client)
        body = _extract_js_function(html, "setTurntable")
        # autoRotate belongs to OrbitControls, so it applies in every view mode.
        assert "controls.autoRotate" in body

    def test_animate_calls_controls_update(self, client: TestClient) -> None:
        """Animation loop calls controls.update() which handles autoRotate."""
        html = _get_html(client)
        body = _extract_js_function(html, "animate")
        assert "controls.update()" in body

    def test_render_frame_handles_compare_mode(
        self, client: TestClient,
    ) -> None:
        """renderFrameAtAngle's body refers to compareMode."""
        html = _get_html(client)
        body = _extract_js_function(html, "renderFrameAtAngle")
        assert "compareMode" in body

    def test_render_frame_handles_crystal_mode(
        self, client: TestClient,
    ) -> None:
        """renderFrameAtAngle's body refers to crystalActive."""
        html = _get_html(client)
        body = _extract_js_function(html, "renderFrameAtAngle")
        assert "crystalActive" in body


class TestExportProgress:
    """Tests for export progress indicator."""

    def test_progress_overlay_exists(self, client: TestClient) -> None:
        """Export progress overlay element exists."""
        html = _get_html(client)
        assert 'id="export-progress"' in html

    def test_progress_text_exists(self, client: TestClient) -> None:
        """Export progress text element exists."""
        html = _get_html(client)
        assert 'id="export-progress-text"' in html

    def test_progress_bar_exists(self, client: TestClient) -> None:
        """Export progress bar element exists."""
        html = _get_html(client)
        assert 'id="export-progress-bar"' in html
        assert 'id="export-progress-fill"' in html

    def test_export_updates_progress(self, client: TestClient) -> None:
        """exportWebM calls updateCaptureProgress."""
        html = _get_html(client)
        body = _extract_js_function(html, "exportWebM")
        assert "updateCaptureProgress" in body

    def test_download_blob_helper_exists(self, client: TestClient) -> None:
        """downloadBlob helper is defined for shared download logic."""
        html = _get_html(client)
        assert "function downloadBlob(" in html

    def test_update_capture_progress_helper_exists(
        self, client: TestClient,
    ) -> None:
        """updateCaptureProgress helper is defined for shared progress logic."""
        html = _get_html(client)
        assert "function updateCaptureProgress(" in html


class TestCaptureRestoreState:
    """Tests that turntable state is captured and restored."""

    def test_capture_includes_turntable(self, client: TestClient) -> None:
        """captureUiState includes turntable flag."""
        html = _get_html(client)
        body = _extract_js_function(html, "captureUiState")
        assert "turntable:" in body

    def test_capture_includes_turntable_speed(
        self, client: TestClient,
    ) -> None:
        """captureUiState includes turntable_speed."""
        html = _get_html(client)
        body = _extract_js_function(html, "captureUiState")
        assert "turntable_speed:" in body

    def test_restore_sets_turntable(self, client: TestClient) -> None:
        """restoreUiState restores turntable state."""
        html = _get_html(client)
        body = _extract_js_function(html, "restoreUiState")
        assert "ui.turntable" in body
        assert "setTurntable" in body

    def test_restore_sets_turntable_speed(self, client: TestClient) -> None:
        """restoreUiState restores turntable speed."""
        html = _get_html(client)
        body = _extract_js_function(html, "restoreUiState")
        assert "ui.turntable_speed" in body
        assert "setTurntableSpeed" in body
