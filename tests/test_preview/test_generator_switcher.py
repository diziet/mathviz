"""Tests for the generator switcher with type-ahead search in the preview UI."""

from typing import Generator

import pytest
from fastapi.testclient import TestClient

from mathviz.core.generator import list_generators, register
from mathviz.generators.parametric.torus import TorusGenerator
from mathviz.preview.server import app, reset_cache


def _ensure_torus_registered() -> None:
    """Re-register the torus generator if missing from the registry."""
    import mathviz.core.generator as gen_mod

    if "torus" not in gen_mod._alias_map:
        gen_mod._discovered = True
        register(TorusGenerator)


# --- Fixtures ---


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
    """Fetch the preview HTML once for assertion-only tests."""
    resp = client.get("/")
    assert resp.status_code == 200
    return resp.text


# --- GET /api/generators returns expected data ---


class TestGeneratorListEndpoint:
    """Tests that GET /api/generators returns data needed by the switcher."""

    def test_returns_non_empty_list_with_name_and_category(
        self, client: TestClient,
    ) -> None:
        """GET /api/generators returns a non-empty list with name and category fields."""
        resp = client.get("/api/generators")
        assert resp.status_code == 200
        data = resp.json()
        assert len(data) > 0
        for entry in data:
            assert "name" in entry
            assert "category" in entry
            assert isinstance(entry["name"], str)
            assert isinstance(entry["category"], str)

    def test_includes_all_registered_generators(self, client: TestClient) -> None:
        """Generator list includes all registered generators from the registry."""
        resp = client.get("/api/generators")
        api_names = {g["name"] for g in resp.json()}
        registry_names = {m.name for m in list_generators()}
        assert api_names == registry_names


# --- HTML contains generator selector elements ---


class TestGeneratorSelectorHTML:
    """Tests that the preview HTML contains the generator selector UI."""

    def test_html_contains_generator_selector_input(self, preview_html: str) -> None:
        """Preview GET / HTML contains the generator selector input element."""
        assert 'id="generator-search"' in preview_html

    def test_html_contains_generator_dropdown(self, preview_html: str) -> None:
        """Preview HTML contains the dropdown container for generator results."""
        assert 'id="generator-dropdown"' in preview_html

    def test_html_fetches_generators_api(self, preview_html: str) -> None:
        """Preview HTML contains /api/generators and fetchGenerators."""
        assert "/api/generators" in preview_html
        assert "fetchGenerators" in preview_html

    def test_html_calls_generate_on_selection(self, preview_html: str) -> None:
        """Preview HTML contains loadFromAPI and /api/generate."""
        assert "loadFromAPI" in preview_html
        assert "/api/generate" in preview_html

    def test_html_updates_url_on_selection(self, preview_html: str) -> None:
        """Preview HTML contains history.replaceState."""
        assert "history.replaceState" in preview_html

    def test_html_has_type_ahead_filtering(self, preview_html: str) -> None:
        """Preview HTML contains filterGenerators and generator-search."""
        assert "filterGenerators" in preview_html
        assert 'id="generator-search"' in preview_html


# --- Seed control ---


class TestSeedControl:
    """Tests for the seed input field and randomize button."""

    def test_seed_input_field_present(self, preview_html: str) -> None:
        """Seed input field is present in the preview HTML."""
        assert 'id="seed-input"' in preview_html

    def test_seed_randomize_button_present(self, preview_html: str) -> None:
        """Randomize button is present next to the seed input."""
        assert 'id="seed-random-btn"' in preview_html

    def test_seed_change_triggers_regeneration(self, preview_html: str) -> None:
        """Preview HTML contains seed-input and applyGenerator."""
        assert "seed-input" in preview_html
        assert "applyGenerator" in preview_html

    def test_seed_input_preloads_from_url(self, preview_html: str) -> None:
        """Preview HTML contains seedInput.value and parseQueryParams."""
        assert "seedInput.value" in preview_html
        assert "parseQueryParams" in preview_html


# --- Generator selector styling ---


class TestGeneratorSelectorStyling:
    """Tests for proper styling of the generator selector."""

    def test_dropdown_has_overlay_styling(self, preview_html: str) -> None:
        """Preview HTML contains generator-dropdown and position:absolute."""
        assert "generator-dropdown" in preview_html
        assert "position:absolute" in preview_html

    def test_category_display_in_dropdown(self, preview_html: str) -> None:
        """Preview HTML contains gen-category."""
        assert "gen-category" in preview_html
        assert "category" in preview_html
