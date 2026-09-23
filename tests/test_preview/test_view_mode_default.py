"""Tests that the default view mode is Vertex Cloud.

The view mode changes only when it is incompatible with the available data,
not after every generation or file load.

These tests match strings in the served HTML source. They do not run the
JavaScript. Covering the runtime branches, such as loading a PLY in wireframe
mode, needs a browser-based test (for example, Playwright).
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
    """Register torus if missing; reset the cache and the served file before and after the test."""
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
    """Tests that the default view mode is Vertex Cloud."""

    def test_vertex_option_is_selected_by_default(self, client: TestClient) -> None:
        """The vertex option in the view-mode select has the selected attribute."""
        html = _get_html(client)
        match = re.search(r'<option\s+value="vertex"[^>]*>', html)
        assert match is not None, "vertex option not found"
        assert "selected" in match.group(0)

    def test_shaded_option_not_selected(self, client: TestClient) -> None:
        """The shaded option should not have the selected attribute."""
        html = _get_html(client)
        match = re.search(r'<option\s+value="shaded"[^>]*>', html)
        assert match is not None, "shaded option not found"
        assert "selected" not in match.group(0)

    def test_js_state_initializes_viewmode_to_vertex(self, client: TestClient) -> None:
        """JavaScript state object initializes viewMode to 'vertex'."""
        html = _get_html(client)
        assert "viewMode: 'vertex'" in html

    def test_initial_mesh_visibility_uses_state_viewmode(
        self, client: TestClient,
    ) -> None:
        """Mesh children visibility is set based on state.viewMode at creation."""
        html = _get_html(client)
        assert "shadedMesh.visible = (state.viewMode === 'shaded')" in html
        assert "pts.visible = (state.viewMode === 'vertex')" in html

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
        # An earlier version had an else branch that always set shaded mode.
        assert "state.viewMode = 'shaded'" not in html

    def test_mesh_only_preserves_vertex_mode(self, client: TestClient) -> None:
        """For mesh-only generators, displayGenerateResult does not force shaded."""
        html = _get_html(client)
        gen_fn = re.search(
            r"async function displayGenerateResult.*?^}",
            html,
            re.MULTILINE | re.DOTALL,
        )
        assert gen_fn is not None
        fn_body = gen_fn.group(0)
        assert "state.viewMode = 'shaded'" not in fn_body
        # The guard uses the helper and only fires when no mesh is available
        assert "viewModeNeedsMesh()" in fn_body

    def test_cloud_only_preserves_vertex_mode(self, client: TestClient) -> None:
        """For cloud-only data, displayGenerateResult keeps vertex mode."""
        html = _get_html(client)
        gen_fn = re.search(
            r"async function displayGenerateResult.*?^}",
            html,
            re.MULTILINE | re.DOTALL,
        )
        assert gen_fn is not None
        fn_body = gen_fn.group(0)
        # Cloud-only means !hasMesh, so the guard fires only if mode needs mesh.
        # Vertex mode (the default) does not need mesh, so no override occurs.
        assert "viewModeNeedsMesh() && !hasMesh" in fn_body
        # The dropdown takes its value from state.viewMode, not a hard-coded mode.
        assert "document.getElementById('view-mode').value = state.viewMode" in fn_body

    def test_dropdown_synced_after_generation(self, client: TestClient) -> None:
        """Dropdown value is synced from state.viewMode after generation."""
        html = _get_html(client)
        # After the compatibility check, the dropdown is always synced
        assert (
            "document.getElementById('view-mode').value = state.viewMode"
            in html
        )

    def test_incompatible_mode_falls_back_to_vertex(
        self, client: TestClient,
    ) -> None:
        """When viewMode needs mesh but no mesh is available, fall back to vertex."""
        html = _get_html(client)
        # The guard uses the extracted helper and falls back to vertex
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
        # An earlier version set `state.viewMode = 'shaded'` after displayMesh.
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
        """loadFromFile for PLY falls back to vertex when mode needs mesh."""
        html = _get_html(client)
        ply_block = re.search(
            r"ext === 'ply'.*?applyViewMode",
            html,
            re.DOTALL,
        )
        assert ply_block is not None
        block = ply_block.group(0)
        assert "viewModeNeedsMesh()" in block
        assert "state.viewMode = 'vertex'" in block

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
        assert "state.viewMode = 'shaded'" not in fn_body
        # The only assignment should be the fallback for incompatible modes
        assignments = re.findall(r"state\.viewMode\s*=\s*'(\w+)'", fn_body)
        assert assignments == ["vertex"], (
            f"Expected only fallback to 'vertex', got: {assignments}"
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
        assert "schwarz" not in fn_body.lower()
        assert "lorenz" not in fn_body.lower()
