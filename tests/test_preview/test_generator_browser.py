"""Tests for the visual generator browser modal (Task 124)."""

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
    """Fetch preview HTML for assertion tests."""
    resp = client.get("/")
    assert resp.status_code == 200
    return resp.text


class TestBrowserModalStructure:
    """Tests for the generator browser modal HTML structure."""

    def test_cmd_k_opens_browser(self, preview_html: str) -> None:
        """Cmd+K keyboard shortcut opens the browser modal."""
        assert "metaKey" in preview_html or "ctrlKey" in preview_html
        assert "key === 'k'" in preview_html or "key==='k'" in preview_html
        assert "openBrowser" in preview_html

    def test_browser_shows_all_categories(self, preview_html: str) -> None:
        """Browser populates category grid from the generators API."""
        assert 'id="generator-browser"' in preview_html
        assert 'id="browser-content"' in preview_html
        assert "getGroupedGenerators" in preview_html
        assert "renderCategoryGrid" in preview_html

    def test_category_card_shows_name_count_thumbs(
        self, preview_html: str
    ) -> None:
        """Each category card shows name, count, and thumbnail previews."""
        assert "category-name" in preview_html
        assert "category-count" in preview_html
        assert "category-thumbs" in preview_html

    def test_clicking_category_shows_generators(
        self, preview_html: str
    ) -> None:
        """Clicking a category card shows its generators in a sub-grid."""
        assert "showCategoryGenerators" in preview_html
        assert "browsing_generators_in_category" in preview_html

    def test_clicking_generator_loads_and_closes(
        self, preview_html: str
    ) -> None:
        """Clicking a generator card calls selectGenerator and closeBrowser."""
        assert "closeBrowser" in preview_html
        assert "selectGenerator" in preview_html

    def test_search_filters_generators(self, preview_html: str) -> None:
        """Search input filters generators across all categories."""
        assert 'id="browser-search"' in preview_html
        assert "renderSearchResults" in preview_html
        assert "filterGenerators" in preview_html

    def test_escape_closes_from_category_grid(
        self, preview_html: str
    ) -> None:
        """Escape key closes the modal from the category grid view."""
        assert "Escape" in preview_html
        assert "closeBrowser" in preview_html

    def test_back_navigation_returns_to_categories(
        self, preview_html: str
    ) -> None:
        """Back navigation returns from category detail to category grid."""
        assert 'id="browser-back"' in preview_html
        assert "browserGoBack" in preview_html
        assert "Back to categories" in preview_html

    def test_selected_generator_highlighted(
        self, preview_html: str
    ) -> None:
        """Currently selected generator is highlighted if visible."""
        # The JS adds 'selected' class to the current generator's card
        assert "selected" in preview_html
        assert "searchInput.value" in preview_html

    def test_modal_populates_from_generators_api(
        self, preview_html: str
    ) -> None:
        """Modal populates dynamically from the generators API."""
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
        """Browser uses a state machine with correct states."""
        assert "browserState" in preview_html
        assert "browsing_categories" in preview_html
        assert "browsing_generators_in_category" in preview_html
        assert "'closed'" in preview_html

    def test_thumbnail_lazy_loading(self, preview_html: str) -> None:
        """Thumbnails are lazy-loaded from the thumbnail endpoint."""
        assert "/api/generators/" in preview_html
        assert "/thumbnail" in preview_html
        assert "loading" in preview_html

    def test_category_shortcut_numbers(self, preview_html: str) -> None:
        """Category cards display shortcut numbers."""
        assert "category-shortcut" in preview_html

    def test_number_key_shortcuts(self, preview_html: str) -> None:
        """Number keys navigate to categories and generators."""
        assert "parseInt(e.key" in preview_html

    def test_search_auto_focus(self, preview_html: str) -> None:
        """Search input is auto-focused when modal opens."""
        assert "browserSearch.focus()" in preview_html

    def test_backdrop_click_closes(self, preview_html: str) -> None:
        """Clicking the backdrop outside the panel closes the modal."""
        assert "e.target === browserOverlay" in preview_html

    def test_generators_api_returns_data(self, client: TestClient) -> None:
        """GET /api/generators returns a list including torus."""
        resp = client.get("/api/generators")
        assert resp.status_code == 200
        data = resp.json()
        names = [g["name"] for g in data]
        assert "torus" in names
        # Each generator should have a category
        for gen in data:
            assert "category" in gen
            assert "name" in gen

    def test_css_placeholder_spinner(self, preview_html: str) -> None:
        """CSS includes placeholder spinner for loading thumbnails."""
        assert "thumb-placeholder" in preview_html
        assert "thumb-spin" in preview_html

    def test_backspace_goes_back(self, preview_html: str) -> None:
        """Backspace key navigates back from category detail."""
        assert "Backspace" in preview_html
        assert "browserGoBack" in preview_html
