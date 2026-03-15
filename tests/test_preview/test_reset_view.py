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

    def test_html_contains_fit_camera_in_reset(self, html: str) -> None:
        """HTML contains resetView function that calls fitCamera."""
        assert "fitCamera(active)" in html
        assert "resetView" in html

    def test_html_contains_active_geometry_check(self, html: str) -> None:
        """HTML checks both meshGroup and cloudPoints for active geometry."""
        assert "state.meshGroup || state.cloudPoints" in html

    def test_html_contains_reset_btn_click_handler(self, html: str) -> None:
        """HTML wires resetView to the button click event."""
        assert "resetViewBtn.addEventListener('click', resetView)" in html

    def test_html_reset_btn_starts_disabled(self, html: str) -> None:
        """Button is disabled in the initial HTML (no geometry loaded)."""
        assert 'id="reset-view-btn" disabled' in html

    def test_html_contains_enable_reset_view_call(self, html: str) -> None:
        """HTML calls enableResetView after geometry display."""
        assert "enableResetView()" in html
        assert "resetViewBtn.disabled = false" in html

    def test_html_disables_reset_on_clear_scene(self, html: str) -> None:
        """HTML disables reset button when clearScene is called."""
        assert "resetViewBtn.disabled = true" in html

    def test_html_contains_home_key_shortcut(self, html: str) -> None:
        """HTML binds the Home key to reset view."""
        assert "'Home'" in html
        assert "resetView" in html

    def test_html_shortcut_checks_modifier_keys(self, html: str) -> None:
        """Keyboard shortcut skips when modifier keys are held."""
        assert "e.ctrlKey" in html
        assert "e.metaKey" in html
        assert "e.altKey" in html

    def test_html_shortcut_checks_disabled_state(self, html: str) -> None:
        """Keyboard shortcut respects the button's disabled state."""
        assert "resetViewBtn.disabled" in html

    def test_html_shortcut_guards_editable_elements(self, html: str) -> None:
        """Keyboard shortcut skips input, select, textarea, and contentEditable."""
        assert "INPUT" in html
        assert "SELECT" in html
        assert "TEXTAREA" in html
        assert "isContentEditable" in html
