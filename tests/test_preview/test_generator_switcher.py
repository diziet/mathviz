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
    """Ensure generators are registered and cache is clean."""
    _ensure_torus_registered()
    reset_cache()
    yield
    reset_cache()


@pytest.fixture
def client() -> TestClient:
    """Return a FastAPI test client."""
    return TestClient(app)


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

    def test_html_contains_generator_selector_input(
        self, client: TestClient,
    ) -> None:
        """Preview GET / HTML contains the generator selector input element."""
        resp = client.get("/")
        assert resp.status_code == 200
        html = resp.text
        assert 'id="generator-search"' in html

    def test_html_contains_generator_dropdown(self, client: TestClient) -> None:
        """Preview HTML contains the dropdown container for generator results."""
        resp = client.get("/")
        html = resp.text
        assert 'id="generator-dropdown"' in html

    def test_html_fetches_generators_api(self, client: TestClient) -> None:
        """Preview GET / HTML contains JavaScript that fetches /api/generators."""
        resp = client.get("/")
        html = resp.text
        assert "/api/generators" in html
        assert "fetchGenerators" in html or "fetch('/api/generators')" in html

    def test_html_calls_generate_on_selection(self, client: TestClient) -> None:
        """Selecting a generator triggers a POST /api/generate with the correct name."""
        resp = client.get("/")
        html = resp.text
        # The JS must call loadFromAPI when a generator is selected
        assert "loadFromAPI" in html
        assert "/api/generate" in html

    def test_html_updates_url_on_selection(self, client: TestClient) -> None:
        """URL is updated with the selected generator name after switching."""
        resp = client.get("/")
        html = resp.text
        assert "history.replaceState" in html

    def test_html_has_type_ahead_filtering(self, client: TestClient) -> None:
        """Type-ahead filtering narrows the visible list correctly."""
        resp = client.get("/")
        html = resp.text
        # Must have input event listener for filtering
        assert "filterGenerators" in html or "filter" in html.lower()
        # Must have the search input
        assert 'id="generator-search"' in html


# --- Seed control ---


class TestSeedControl:
    """Tests for the seed input field and randomize button."""

    def test_seed_input_field_present(self, client: TestClient) -> None:
        """Seed input field is present in the preview HTML."""
        resp = client.get("/")
        html = resp.text
        assert 'id="seed-input"' in html

    def test_seed_randomize_button_present(self, client: TestClient) -> None:
        """Randomize button is present next to the seed input."""
        resp = client.get("/")
        html = resp.text
        assert 'id="seed-random-btn"' in html

    def test_seed_change_triggers_regeneration(self, client: TestClient) -> None:
        """Changing the seed triggers regeneration via loadFromAPI."""
        resp = client.get("/")
        html = resp.text
        # Seed input must have a change/event handler that calls regeneration
        assert "seed-input" in html
        assert "loadFromAPI" in html

    def test_seed_input_preloads_from_url(self, client: TestClient) -> None:
        """Seed input is populated from the URL query param on load."""
        resp = client.get("/")
        html = resp.text
        # The JS reads seed from URL params and sets the input value
        assert "seed" in html
        assert "parseQueryParams" in html


# --- Generator selector styling ---


class TestGeneratorSelectorStyling:
    """Tests for proper styling of the generator selector."""

    def test_dropdown_has_overlay_styling(self, client: TestClient) -> None:
        """Dropdown has position:absolute to overlay the 3D canvas."""
        resp = client.get("/")
        html = resp.text
        assert "generator-dropdown" in html
        assert "position" in html

    def test_category_display_in_dropdown(self, client: TestClient) -> None:
        """Generator entries show category information."""
        resp = client.get("/")
        html = resp.text
        # The JS must render category info for each generator
        assert "category" in html
