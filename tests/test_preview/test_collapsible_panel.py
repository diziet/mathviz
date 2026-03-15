"""Tests for the collapsible Dimensions/Margins panel in the preview UI."""

from typing import Generator

import pytest
from fastapi.testclient import TestClient

from mathviz.core.generator import register
from mathviz.generators.parametric.torus import TorusGenerator
from mathviz.preview.server import app, reset_cache


def _ensure_torus_registered() -> None:
    """Re-register the torus generator if missing from the registry."""
    import mathviz.core.generator as gen_mod

    if "torus" not in gen_mod._alias_map:
        gen_mod._discovered = True
        register(TorusGenerator)


@pytest.fixture(autouse=True)
def _ensure_generators() -> Generator[None, None, None]:
    """Ensure generators are registered and cache is clean."""
    _ensure_torus_registered()
    reset_cache()
    yield
    reset_cache()


@pytest.fixture
def client() -> TestClient:
    """Return a FastAPI test client."""
    return TestClient(app)


@pytest.fixture
def preview_html(client: TestClient) -> str:
    """Fetch the preview HTML once for assertion-only tests."""
    resp = client.get("/")
    assert resp.status_code == 200
    return resp.text


class TestCollapsiblePanelHTML:
    """Tests that the preview HTML contains collapsible panel markup and JS."""

    def test_html_contains_toggle_header(self, preview_html: str) -> None:
        """HTML contains a clickable toggle header with chevron."""
        assert 'id="container-toggle"' in preview_html
        assert "Dimensions / Margins" in preview_html
        assert "chevron" in preview_html

    def test_html_panel_has_collapsed_class_by_default(self, preview_html: str) -> None:
        """Panel element starts with the collapsed class in markup."""
        assert 'id="container-panel" class="collapsed"' in preview_html
        assert "setContainerCollapsed(savedCollapsed" in preview_html

    def test_html_contains_toggle_listener(self, preview_html: str) -> None:
        """HTML contains JS that registers a click listener on the toggle."""
        assert "containerToggle.addEventListener" in preview_html
        assert "setContainerCollapsed(!containerPanel.classList.contains" in preview_html

    def test_html_contains_collapse_class_management(self, preview_html: str) -> None:
        """HTML contains JS that adds and removes the collapsed class."""
        assert "classList.contains('collapsed')" in preview_html
        assert "classList.add('collapsed')" in preview_html
        assert "classList.remove('collapsed')" in preview_html

    def test_html_uses_display_none_for_collapsed_body(self, preview_html: str) -> None:
        """Collapsed body uses display:none, preserving input values in DOM."""
        assert '#container-panel.collapsed #container-body{display:none}' in preview_html
        assert 'id="container-body"' in preview_html
        assert 'id="dim-w"' in preview_html
        assert 'id="margin-x"' in preview_html

    def test_html_contains_chevron_indicators(self, preview_html: str) -> None:
        """HTML contains both collapsed and expanded chevron characters."""
        # &#9656; = ▸ (collapsed), &#9662; = ▾ (expanded)
        assert "&#9656;" in preview_html
        assert "&#9662;" in preview_html

    def test_html_contains_localstorage_persistence(self, preview_html: str) -> None:
        """HTML contains JS for saving/loading collapsed state via localStorage."""
        assert "localStorage.setItem('containerPanelCollapsed'" in preview_html
        assert "localStorage.getItem('containerPanelCollapsed')" in preview_html
