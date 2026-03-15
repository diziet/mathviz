"""Tests for default view mode being Point Cloud.

# TODO: Runtime JS view-mode selection logic in displayGenerateResult and
# loadFromFile (choosing 'points' vs 'shaded' based on hasMesh/hasCloud) is
# not tested here — these tests only verify static HTML content. A browser-based
# test (e.g. Playwright) would be needed to cover the runtime conditional paths.
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


class TestViewModeDefault:
    """Tests that the default view mode is Point Cloud."""

    def test_points_option_is_selected_by_default(self, client: TestClient) -> None:
        """The points option in the view-mode select has the selected attribute."""
        resp = client.get("/")
        html = resp.text
        # Find the points option and verify it has 'selected'
        match = re.search(r'<option\s+value="points"[^>]*>', html)
        assert match is not None, "points option not found"
        assert "selected" in match.group(0)

    def test_shaded_option_not_selected(self, client: TestClient) -> None:
        """The shaded option should not have the selected attribute."""
        resp = client.get("/")
        html = resp.text
        match = re.search(r'<option\s+value="shaded"[^>]*>', html)
        assert match is not None, "shaded option not found"
        assert "selected" not in match.group(0)

    def test_js_state_initializes_viewmode_to_points(self, client: TestClient) -> None:
        """JavaScript state object initializes viewMode to 'points'."""
        resp = client.get("/")
        html = resp.text
        assert "viewMode: 'points'" in html

    def test_initial_mesh_visibility_uses_state_viewmode(
        self, client: TestClient,
    ) -> None:
        """Mesh children visibility is set based on state.viewMode at creation."""
        resp = client.get("/")
        html = resp.text
        # Shaded mesh visibility driven by state
        assert "shadedMesh.visible = (state.viewMode === 'shaded')" in html
        # Points visibility driven by state
        assert "pts.visible = (state.viewMode === 'points')" in html

    def test_load_from_file_sets_shaded_for_mesh_formats(
        self, client: TestClient,
    ) -> None:
        """loadFromFile sets viewMode to shaded for STL/GLB/GLTF files."""
        resp = client.get("/")
        html = resp.text
        # After loading a mesh file, view mode should switch to shaded
        assert "await displayMesh(url, 'file');" in html
        assert "state.viewMode = 'shaded'" in html

    def test_load_from_file_sets_points_for_ply(
        self, client: TestClient,
    ) -> None:
        """loadFromFile sets viewMode to points for PLY files."""
        resp = client.get("/")
        html = resp.text
        assert "displayCloud(points, vertexCount, 'file');" in html

    def test_display_generate_guards_empty_response(
        self, client: TestClient,
    ) -> None:
        """displayGenerateResult returns early when no mesh or cloud data."""
        resp = client.get("/")
        html = resp.text
        assert "if (!hasMesh && !hasCloud)" in html
