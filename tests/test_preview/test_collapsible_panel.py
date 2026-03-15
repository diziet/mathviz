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
    """Tests for the collapsible container panel markup and behaviour."""

    def test_has_clickable_header_with_toggle(self, preview_html: str) -> None:
        """Container panel has a clickable header with collapse toggle."""
        assert 'id="container-toggle"' in preview_html
        assert "Dimensions / Margins" in preview_html
        assert "chevron" in preview_html

    def test_collapsed_by_default(self, preview_html: str) -> None:
        """Panel is collapsed by default (dimension inputs not visible)."""
        # The panel element starts with class="collapsed"
        assert 'id="container-panel" class="collapsed"' in preview_html
        # JS also defaults to collapsed when no localStorage value exists
        assert "setContainerCollapsed(savedCollapsed" in preview_html

    def test_clicking_header_expands_panel(self, preview_html: str) -> None:
        """Clicking the header expands the panel (inputs become visible)."""
        # The toggle click handler calls setContainerCollapsed which removes
        # the .collapsed class, making #container-body visible
        assert "containerToggle.addEventListener" in preview_html
        assert "setContainerCollapsed(!containerPanel.classList.contains" in preview_html

    def test_clicking_again_collapses(self, preview_html: str) -> None:
        """Clicking the header again collapses the panel."""
        # The toggle is a simple boolean flip — same handler toggles both ways
        assert "classList.contains('collapsed')" in preview_html
        assert "classList.add('collapsed')" in preview_html
        assert "classList.remove('collapsed')" in preview_html

    def test_dimension_values_preserved(self, preview_html: str) -> None:
        """Dimension/margin values are preserved when collapsing and expanding."""
        # Inputs live inside #container-body which uses display:none when
        # collapsed — values are never cleared. The CSS rule hides, not removes.
        assert '#container-panel.collapsed #container-body{display:none}' in preview_html
        # The body wraps all inputs — they remain in the DOM
        assert 'id="container-body"' in preview_html
        assert 'id="dim-w"' in preview_html
        assert 'id="margin-x"' in preview_html

    def test_chevron_indicators(self, preview_html: str) -> None:
        """Chevron switches between collapsed and expanded indicators."""
        # &#9656; = ▸ (collapsed), &#9662; = ▾ (expanded)
        assert "&#9656;" in preview_html
        assert "&#9662;" in preview_html

    def test_localstorage_persistence(self, preview_html: str) -> None:
        """Collapsed state is saved to localStorage."""
        assert "localStorage.setItem('containerPanelCollapsed'" in preview_html
        assert "localStorage.getItem('containerPanelCollapsed')" in preview_html
