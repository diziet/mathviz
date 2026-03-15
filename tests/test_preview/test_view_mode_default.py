"""Tests for default view mode being Point Cloud.

Verifies that view mode is only overridden when incompatible with available
data, not unconditionally after every generation or file load.

# NOTE: These tests verify static HTML source patterns (string matching on
# the served page). They do NOT exercise runtime JS behavior — a browser-based
# test (e.g. Playwright) would be needed to cover the runtime conditional
# paths such as loading a PLY while in wireframe mode.
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


def _get_html(client: TestClient) -> str:
    """Fetch the preview HTML page."""
    resp = client.get("/")
    assert resp.status_code == 200
    return resp.text


class TestViewModeDefault:
    """Tests that the default view mode is Point Cloud."""

    def test_points_option_is_selected_by_default(self, client: TestClient) -> None:
        """The points option in the view-mode select has the selected attribute."""
        html = _get_html(client)
        match = re.search(r'<option\s+value="points"[^>]*>', html)
        assert match is not None, "points option not found"
        assert "selected" in match.group(0)

    def test_shaded_option_not_selected(self, client: TestClient) -> None:
        """The shaded option should not have the selected attribute."""
        html = _get_html(client)
        match = re.search(r'<option\s+value="shaded"[^>]*>', html)
        assert match is not None, "shaded option not found"
        assert "selected" not in match.group(0)

    def test_js_state_initializes_viewmode_to_points(self, client: TestClient) -> None:
        """JavaScript state object initializes viewMode to 'points'."""
        html = _get_html(client)
        assert "viewMode: 'points'" in html

    def test_initial_mesh_visibility_uses_state_viewmode(
        self, client: TestClient,
    ) -> None:
        """Mesh children visibility is set based on state.viewMode at creation."""
        html = _get_html(client)
        assert "shadedMesh.visible = (state.viewMode === 'shaded')" in html
        assert "pts.visible = (state.viewMode === 'points')" in html

    def test_display_generate_guards_empty_response(
        self, client: TestClient,
    ) -> None:
        """displayGenerateResult returns early when no mesh or cloud data."""
        html = _get_html(client)
        assert "if (!hasMesh && !hasCloud)" in html


class TestViewModeNotOverridden:
    """Tests that view mode is preserved when compatible with available data."""

    def test_view_mode_needs_mesh_helper_exists(self, client: TestClient) -> None:
        """The viewModeNeedsMesh helper is defined and checks shaded/wireframe."""
        html = _get_html(client)
        assert "function viewModeNeedsMesh()" in html
        assert "'shaded'" in html
        assert "'wireframe'" in html

    def test_no_unconditional_shaded_override(self, client: TestClient) -> None:
        """displayGenerateResult must not unconditionally set shaded mode."""
        html = _get_html(client)
        # The old else branch that forced shaded should be gone
        assert "state.viewMode = 'shaded'" not in html

    def test_mesh_only_preserves_points_mode(self, client: TestClient) -> None:
        """For mesh-only generators, displayGenerateResult does not force shaded."""
        html = _get_html(client)
        gen_fn = re.search(
            r"async function displayGenerateResult.*?^}",
            html,
            re.MULTILINE | re.DOTALL,
        )
        assert gen_fn is not None
        fn_body = gen_fn.group(0)
        # Only fallback assignment should be to 'points' (for incompatible mode)
        assert "state.viewMode = 'shaded'" not in fn_body
        # The guard uses the helper and only fires when no mesh is available
        assert "viewModeNeedsMesh()" in fn_body

    def test_cloud_only_preserves_points_mode(self, client: TestClient) -> None:
        """For cloud-only data, displayGenerateResult keeps points mode."""
        html = _get_html(client)
        gen_fn = re.search(
            r"async function displayGenerateResult.*?^}",
            html,
            re.MULTILINE | re.DOTALL,
        )
        assert gen_fn is not None
        fn_body = gen_fn.group(0)
        # Cloud-only means !hasMesh, so the guard fires only if mode needs mesh.
        # Points mode (the default) does not need mesh, so no override occurs.
        assert "viewModeNeedsMesh() && !hasMesh" in fn_body
        # Verify dropdown is synced to state (not hardcoded)
        assert "document.getElementById('view-mode').value = state.viewMode" in fn_body

    def test_dropdown_synced_after_generation(self, client: TestClient) -> None:
        """Dropdown value is synced from state.viewMode after generation."""
        html = _get_html(client)
        # After the compatibility check, the dropdown is always synced
        assert (
            "document.getElementById('view-mode').value = state.viewMode"
            in html
        )

    def test_incompatible_mode_falls_back_to_points(
        self, client: TestClient,
    ) -> None:
        """When viewMode needs mesh but no mesh is available, fall back to points."""
        html = _get_html(client)
        # The guard uses the extracted helper and falls back to points
        pattern = r"viewModeNeedsMesh\(\)\s*&&\s*!hasMesh"
        assert re.search(pattern, html), (
            "Missing incompatibility guard for mesh-requiring view modes"
        )

    def test_load_from_file_mesh_preserves_mode(
        self, client: TestClient,
    ) -> None:
        """loadFromFile for mesh formats preserves current view mode."""
        html = _get_html(client)
        # Mesh formats (STL/GLB/GLTF) support all view modes — no override
        # The old `state.viewMode = 'shaded'` after displayMesh should be gone
        mesh_block = re.search(
            r"ext === 'stl'.*?else if.*?ext === 'ply'",
            html,
            re.DOTALL,
        )
        assert mesh_block is not None
        assert "state.viewMode = 'shaded'" not in mesh_block.group(0)

    def test_load_from_file_ply_falls_back_when_incompatible(
        self, client: TestClient,
    ) -> None:
        """loadFromFile for PLY falls back to points when mode needs mesh."""
        html = _get_html(client)
        ply_block = re.search(
            r"ext === 'ply'.*?applyViewMode",
            html,
            re.DOTALL,
        )
        assert ply_block is not None
        block = ply_block.group(0)
        assert "viewModeNeedsMesh()" in block
        assert "state.viewMode = 'points'" in block

    def test_user_wireframe_preserved_across_mesh_regeneration(
        self, client: TestClient,
    ) -> None:
        """User-selected wireframe is preserved when regenerating mesh-only."""
        html = _get_html(client)
        # Wireframe requires mesh — mesh-only data has mesh, so it's compatible.
        # Verify no unconditional override to shaded or points for mesh data.
        gen_fn = re.search(
            r"async function displayGenerateResult.*?^}",
            html,
            re.MULTILINE | re.DOTALL,
        )
        assert gen_fn is not None
        fn_body = gen_fn.group(0)
        # Should NOT contain unconditional assignments
        assert "state.viewMode = 'shaded'" not in fn_body
        # The only assignment should be the fallback for incompatible modes
        assignments = re.findall(r"state\.viewMode\s*=\s*'(\w+)'", fn_body)
        assert assignments == ["points"], (
            f"Expected only fallback to 'points', got: {assignments}"
        )

    def test_switching_generators_preserves_compatible_mode(
        self, client: TestClient,
    ) -> None:
        """Switching generators preserves view mode when compatible."""
        html = _get_html(client)
        # The function must not have generator-specific mode logic
        gen_fn = re.search(
            r"async function displayGenerateResult.*?^}",
            html,
            re.MULTILINE | re.DOTALL,
        )
        assert gen_fn is not None
        fn_body = gen_fn.group(0)
        # No references to specific generator names
        assert "schwarz" not in fn_body.lower()
        assert "lorenz" not in fn_body.lower()
