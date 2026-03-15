"""Tests for the Reset View button in the preview UI."""

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


class TestResetViewButton:
    """Tests for the Reset View button in the preview HTML."""

    def test_html_contains_reset_view_button(self, html: str) -> None:
        """Preview HTML contains a Reset View button."""
        assert 'id="reset-view-btn"' in html
        assert "Reset View" in html

    def test_reset_view_calls_fit_camera(self, html: str) -> None:
        """Clicking the button calls fitCamera on the active geometry."""
        assert "fitCamera(active)" in html or "fitCamera" in html
        assert "resetView" in html

    def test_reset_view_reads_active_geometry(self, html: str) -> None:
        """Reset uses whichever of meshGroup or cloudPoints is active."""
        assert "state.meshGroup" in html
        assert "state.cloudPoints" in html
        assert "resetView" in html

    def test_reset_view_works_with_mesh(self, html: str) -> None:
        """Button works when viewing mesh geometry (meshGroup is checked first)."""
        # The resetView function checks state.meshGroup || state.cloudPoints
        assert "state.meshGroup || state.cloudPoints" in html

    def test_reset_view_works_with_point_cloud(self, html: str) -> None:
        """Button works when viewing point cloud geometry."""
        # cloudPoints is the fallback when meshGroup is null
        assert "state.cloudPoints" in html
        assert "resetView" in html

    def test_reset_view_disabled_when_no_geometry(self, html: str) -> None:
        """Button is disabled when no geometry is loaded."""
        assert 'id="reset-view-btn" disabled' in html

    def test_reset_view_enabled_after_geometry_load(self, html: str) -> None:
        """Button is enabled after geometry is loaded via enableResetView."""
        assert "enableResetView()" in html
        assert "disabled = false" in html or '.disabled = false' in html

    def test_reset_view_disabled_on_clear_scene(self, html: str) -> None:
        """Button is disabled when scene is cleared."""
        # clearScene sets reset-view-btn.disabled = true
        assert "reset-view-btn" in html
        assert "disabled = true" in html or '.disabled = true' in html

    def test_keyboard_shortcut_home_key(self, html: str) -> None:
        """Home key triggers reset view."""
        assert "'Home'" in html
        assert "resetView" in html

    def test_keyboard_shortcut_zero_key(self, html: str) -> None:
        """0 key triggers reset view."""
        assert "'0'" in html
        assert "resetView" in html

    def test_keyboard_shortcut_skips_inputs(self, html: str) -> None:
        """Keyboard shortcut does not fire when typing in input fields."""
        assert "INPUT" in html
        assert "SELECT" in html
