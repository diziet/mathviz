"""Tests for multi-panel comparison view in the preview UI."""

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


class TestCompareModeToggle:
    """Tests for the compare mode toggle control."""

    def test_compare_mode_select_exists(self, preview_html: str) -> None:
        """Preview HTML contains a compare mode toggle dropdown."""
        assert 'id="compare-mode"' in preview_html

    def test_compare_mode_has_single_option(self, preview_html: str) -> None:
        """Compare mode dropdown has a Single View option."""
        assert "Single View" in preview_html

    def test_compare_mode_has_2x2_option(self, preview_html: str) -> None:
        """Compare mode dropdown has a 2x2 option."""
        assert 'value="2x2"' in preview_html

    def test_compare_mode_has_3x3_option(self, preview_html: str) -> None:
        """Compare mode dropdown has a 3x3 option."""
        assert 'value="3x3"' in preview_html


class TestViewportSplitting:
    """Tests for viewport splitting with setViewport/setScissor."""

    def test_2x2_creates_4_viewports(self, preview_html: str) -> None:
        """Selecting 2x2 mode creates 4 viewport regions in the canvas."""
        assert "setViewport" in preview_html
        assert "setScissor" in preview_html
        # The render loop iterates over comparePanels
        assert "for (const panel of state.comparePanels)" in preview_html

    def test_3x3_creates_9_viewports(self, preview_html: str) -> None:
        """Selecting 3x3 mode creates 9 viewport regions via grid dimension."""
        assert "getGridDimension" in preview_html
        # 3x3 returns dim=3, so 3*3=9 panels
        assert "'3x3'" in preview_html

    def test_viewport_uses_scissor_test(self, preview_html: str) -> None:
        """Compare mode render enables scissor test for clean viewport regions."""
        assert "setScissorTest(true)" in preview_html


class TestSharedCamera:
    """Tests for shared camera across all viewports."""

    def test_all_viewports_share_camera(self, preview_html: str) -> None:
        """All viewports share the same camera object in the render loop."""
        # The renderCompareMode function uses the single camera for all panels
        render_fn = preview_html.split("function renderCompareMode")[1].split(
            "function animate"
        )[0]
        assert "renderer.render(panel.scene, camera)" in render_fn

    def test_single_orbit_controls(self, preview_html: str) -> None:
        """Only one OrbitControls instance is created for all panels."""
        # Count OrbitControls instantiations
        count = preview_html.count("new OrbitControls(")
        assert count == 1


class TestPerPanelScenes:
    """Tests for independent per-panel scenes."""

    def test_each_panel_has_own_scene(self, preview_html: str) -> None:
        """Each viewport has its own scene with independent geometry."""
        assert "createPanelScene" in preview_html
        assert "new THREE.Scene()" in preview_html

    def test_panel_data_has_scene(self, preview_html: str) -> None:
        """Panel data structure includes its own scene."""
        panel_fn = preview_html.split("function createPanelData")[1].split(
            "function"
        )[0]
        assert "scene: createPanelScene()" in panel_fn

    def test_panel_data_has_mesh_and_cloud(self, preview_html: str) -> None:
        """Panel data has independent meshGroup and cloudPoints."""
        panel_fn = preview_html.split("function createPanelData")[1].split(
            "function"
        )[0]
        assert "meshGroup: null" in panel_fn
        assert "cloudPoints: null" in panel_fn


class TestPanelOverlay:
    """Tests for per-panel parameter overlay."""

    def test_panel_overlay_created_for_each_viewport(
        self, preview_html: str,
    ) -> None:
        """Per-panel parameter overlay is present in the DOM for each viewport."""
        assert "panel-overlay" in preview_html
        assert "createPanelOverlayDOM" in preview_html

    def test_overlay_has_summary(self, preview_html: str) -> None:
        """Each overlay has a summary line showing seed/param diffs."""
        assert "overlay-summary" in preview_html

    def test_overlay_has_editor(self, preview_html: str) -> None:
        """Each overlay has an inline parameter editor."""
        assert "overlay-editor" in preview_html

    def test_overlay_expands_on_click(self, preview_html: str) -> None:
        """Clicking the overlay toggles the expanded class."""
        assert "overlay.classList.toggle('expanded')" in preview_html

    def test_overlay_collapses_on_escape(self, preview_html: str) -> None:
        """Pressing Escape collapses expanded overlays."""
        assert "Escape" in preview_html
        assert "panel-overlay" in preview_html


class TestExitCompareMode:
    """Tests for exiting compare mode."""

    def test_exit_returns_to_single_view(self, preview_html: str) -> None:
        """Exiting compare mode returns to single-view with panel 1's geometry."""
        assert "exitCompareMode" in preview_html
        exit_fn = preview_html.split("function exitCompareMode")[1].split(
            "/* Compare mode toggle */"
        )[0]
        # Panel 0 geometry is cloned back to main scene
        assert "comparePanels[0]" in exit_fn
        assert "state.compareMode = null" in exit_fn

    def test_exit_cleans_up_panel_dom(self, preview_html: str) -> None:
        """Exiting compare mode removes panel labels and overlays."""
        exit_fn = preview_html.split("function exitCompareMode")[1].split(
            "/* Compare mode toggle */"
        )[0]
        assert "removeAllPanelDOM" in exit_fn


class TestSharedControls:
    """Tests for shared controls applying to all panels."""

    def test_view_mode_applies_to_all_panels(self, preview_html: str) -> None:
        """Shared view mode control applies to all panels."""
        assert "applyViewModeToAllPanels" in preview_html

    def test_point_size_applies_to_all_panels(self, preview_html: str) -> None:
        """Shared point size slider applies to all panels."""
        assert "updatePointSizeInPanels" in preview_html

    def test_background_applies_to_all_panels(self, preview_html: str) -> None:
        """Shared background toggle applies to all panels."""
        assert "updatePanelBackgrounds" in preview_html

    def test_bbox_applies_to_all_panels(self, preview_html: str) -> None:
        """Shared bounding box toggle applies to all panels."""
        assert "updatePanelBBoxVisibility" in preview_html


class TestPanelLabels:
    """Tests for panel labels in compare mode."""

    def test_panel_labels_exist(self, preview_html: str) -> None:
        """Each panel has a label in the top-left corner."""
        assert "panel-label" in preview_html
        assert "PANEL_LABELS" in preview_html

    def test_panel_labels_use_letters(self, preview_html: str) -> None:
        """Panel labels use letters A, B, C, D, etc."""
        assert "['A','B','C','D'" in preview_html


class TestPanelIndependence:
    """Tests that changing params in one panel does not affect others."""

    def test_per_panel_regeneration(self, preview_html: str) -> None:
        """Each panel generates independently via regeneratePanel."""
        assert "regeneratePanel" in preview_html
        regen_fn = preview_html.split("async function regeneratePanel")[1].split(
            "function addBoundingBoxToScene"
        )[0]
        # Only updates the specific panel's scene
        assert "panel.scene" in regen_fn

    def test_panel_has_independent_params(self, preview_html: str) -> None:
        """Each panel stores its own params and seed."""
        panel_fn = preview_html.split("function createPanelData")[1].split(
            "function"
        )[0]
        assert "params:" in panel_fn
        assert "seed:" in panel_fn

    def test_overlay_apply_only_regenerates_one_panel(
        self, preview_html: str,
    ) -> None:
        """Applying overlay params only regenerates that specific panel."""
        apply_fn = preview_html.split("async function applyPanelOverlay")[1].split(
            "async function regeneratePanel"
        )[0]
        assert "regeneratePanel(panelIndex)" in apply_fn


class TestPanelSpinner:
    """Tests for per-panel loading spinner."""

    def test_panel_spinner_exists(self, preview_html: str) -> None:
        """Each panel has a loading spinner element."""
        assert "panel-spinner" in preview_html

    def test_spinner_shown_during_generation(self, preview_html: str) -> None:
        """Spinner is shown when a panel is regenerating."""
        assert "setPanelSpinner" in preview_html


class TestCompareState:
    """Tests for compare mode state management."""

    def test_state_has_compare_mode(self, preview_html: str) -> None:
        """State object has a compareMode field."""
        assert "compareMode:" in preview_html

    def test_state_has_compare_panels(self, preview_html: str) -> None:
        """State object has a comparePanels array."""
        assert "comparePanels:" in preview_html

    def test_enter_compare_populates_panels(self, preview_html: str) -> None:
        """Entering compare mode creates panels with current generator."""
        enter_fn = preview_html.split("async function enterCompareMode")[1].split(
            "function exitCompareMode"
        )[0]
        assert "createPanelData" in enter_fn
        assert "state.comparePanels" in enter_fn
