"""Tests for the visual generator browser modal (Task 124).

All tests but test_generators_api_returns_data check for substrings in the
preview HTML; no script runs.
"""

from collections.abc import Generator

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
    """Register torus if missing; reset the cache before and after the test."""
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
    """Fetch preview HTML for assertion tests."""
    resp = client.get("/")
    assert resp.status_code == 200
    return resp.text


class TestBrowserModalStructure:
    """Tests for the generator browser modal HTML structure."""

    def test_cmd_k_opens_browser(self, preview_html: str) -> None:
        """Preview HTML contains metaKey or ctrlKey, a key === 'k' check and
        openBrowser."""
        assert "metaKey" in preview_html or "ctrlKey" in preview_html
        assert "key === 'k'" in preview_html or "key==='k'" in preview_html
        assert "openBrowser" in preview_html

    def test_browser_shows_all_categories(self, preview_html: str) -> None:
        """Preview HTML contains the generator-browser and browser-content ids,
        getGroupedGenerators and renderCategoryGrid."""
        assert 'id="generator-browser"' in preview_html
        assert 'id="browser-content"' in preview_html
        assert "getGroupedGenerators" in preview_html
        assert "renderCategoryGrid" in preview_html

    def test_category_card_shows_name_count_thumbs(
        self, preview_html: str
    ) -> None:
        """Preview HTML contains category-name, category-count and
        category-thumbs."""
        assert "category-name" in preview_html
        assert "category-count" in preview_html
        assert "category-thumbs" in preview_html

    def test_clicking_category_shows_generators(
        self, preview_html: str
    ) -> None:
        """Preview HTML contains showCategoryGenerators and
        browsing_generators_in_category."""
        assert "showCategoryGenerators" in preview_html
        assert "browsing_generators_in_category" in preview_html

    def test_clicking_generator_loads_and_closes(
        self, preview_html: str
    ) -> None:
        """Preview HTML contains closeBrowser and selectGenerator."""
        assert "closeBrowser" in preview_html
        assert "selectGenerator" in preview_html

    def test_search_filters_generators(self, preview_html: str) -> None:
        """Preview HTML contains browser-search, renderSearchResults and
        filterGenerators."""
        assert 'id="browser-search"' in preview_html
        assert "renderSearchResults" in preview_html
        assert "filterGenerators" in preview_html

    def test_escape_closes_from_category_grid(
        self, preview_html: str
    ) -> None:
        """Preview HTML contains Escape and closeBrowser."""
        assert "Escape" in preview_html
        assert "closeBrowser" in preview_html

    def test_back_navigation_returns_to_categories(
        self, preview_html: str
    ) -> None:
        """Preview HTML contains browser-back, browserGoBack and Back to
        categories."""
        assert 'id="browser-back"' in preview_html
        assert "browserGoBack" in preview_html
        assert "Back to categories" in preview_html

    def test_selected_generator_highlighted(
        self, preview_html: str
    ) -> None:
        """Preview HTML contains selected and searchInput.value."""
        assert "selected" in preview_html
        assert "searchInput.value" in preview_html

    def test_modal_populates_from_generators_api(
        self, preview_html: str
    ) -> None:
        """Preview HTML contains fetchGenerators, allGenerators and
        /api/generators."""
        assert "fetchGenerators" in preview_html
        assert "allGenerators" in preview_html
        assert "/api/generators" in preview_html


class TestBrowserModalFeatures:
    """Tests for generator browser modal JS features."""

    def test_browser_overlay_structure(self, preview_html: str) -> None:
        """Browser has overlay, panel, header, search, close button."""
        assert 'id="generator-browser"' in preview_html
        assert 'id="browser-panel"' in preview_html
        assert 'id="browser-header"' in preview_html
        assert 'id="browser-search"' in preview_html
        assert 'id="browser-close"' in preview_html

    def test_browser_state_machine(self, preview_html: str) -> None:
        """Preview HTML contains browserState and the three state names."""
        assert "browserState" in preview_html
        assert "browsing_categories" in preview_html
        assert "browsing_generators_in_category" in preview_html
        assert "'closed'" in preview_html

    def test_thumbnail_lazy_loading(self, preview_html: str) -> None:
        """Preview HTML contains /api/generators/, /thumbnail and loading."""
        assert "/api/generators/" in preview_html
        assert "/thumbnail" in preview_html
        assert "loading" in preview_html

    def test_category_shortcut_numbers(self, preview_html: str) -> None:
        """Preview HTML contains the category-shortcut class."""
        assert "category-shortcut" in preview_html

    def test_number_key_shortcuts(self, preview_html: str) -> None:
        """Preview HTML contains parseInt(e.key."""
        assert "parseInt(e.key" in preview_html

    def test_search_auto_focus(self, preview_html: str) -> None:
        """Preview HTML calls browserSearch.focus()."""
        assert "browserSearch.focus()" in preview_html

    def test_backdrop_click_closes(self, preview_html: str) -> None:
        """Preview HTML compares e.target === browserOverlay."""
        assert "e.target === browserOverlay" in preview_html

    def test_generators_api_returns_data(self, client: TestClient) -> None:
        """GET /api/generators returns a list including torus."""
        resp = client.get("/api/generators")
        assert resp.status_code == 200
        data = resp.json()
        names = [g["name"] for g in data]
        assert "torus" in names
        for gen in data:
            assert "category" in gen
            assert "name" in gen

    def test_css_placeholder_spinner(self, preview_html: str) -> None:
        """Preview HTML contains thumb-placeholder and thumb-spin."""
        assert "thumb-placeholder" in preview_html
        assert "thumb-spin" in preview_html

    def test_backspace_goes_back(self, preview_html: str) -> None:
        """Preview HTML contains Backspace and browserGoBack."""
        assert "Backspace" in preview_html
        assert "browserGoBack" in preview_html
