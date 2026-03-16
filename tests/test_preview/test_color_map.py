"""Tests for color map view mode in the Three.js viewer."""

import re

import pytest
from fastapi.testclient import TestClient

from mathviz.preview.server import app, reset_cache, set_served_file


@pytest.fixture(autouse=True)
def _clean_state() -> None:
    """Reset server state between tests."""
    reset_cache()
    set_served_file(None)
    yield
    reset_cache()
    set_served_file(None)


@pytest.fixture
def html() -> str:
    """Fetch the viewer HTML once for all assertion tests."""
    client = TestClient(app)
    resp = client.get("/")
    assert resp.status_code == 200
    return resp.text


class TestColorMapViewModeOption:
    """Color Map appears as a view mode option."""

    def test_colormap_option_in_dropdown(self, html: str) -> None:
        """View mode dropdown contains a Color Map option."""
        assert 'value="colormap"' in html

    def test_colormap_option_label(self, html: str) -> None:
        """Color Map option has descriptive label text."""
        assert "Color Map" in html


class TestColorMapControlsVisibility:
    """Selecting Color Map mode shows the metric and gradient selectors."""

    def test_colormap_controls_hidden_by_default(self, html: str) -> None:
        """Color map controls are hidden when not in color map mode."""
        assert re.search(
            r'id="colormap-controls"[^>]*style="display:none"', html
        ), "Colormap controls should be hidden by default"

    def test_metric_selector_exists(self, html: str) -> None:
        """Metric selector dropdown is present."""
        assert 'id="colormap-metric"' in html

    def test_gradient_selector_exists(self, html: str) -> None:
        """Gradient selector dropdown is present."""
        assert 'id="colormap-gradient"' in html

    def test_enter_colormap_shows_controls(self, html: str) -> None:
        """enterColorMapMode sets controls to visible."""
        assert re.search(
            r"colormap-controls.*display.*block", html, re.DOTALL
        ), "enterColorMapMode should show colormap controls"


class TestHeightMetric:
    """Height metric colors vertices by Z coordinate."""

    def test_height_metric_option(self, html: str) -> None:
        """Height metric is available in the metric selector."""
        assert 'value="height"' in html
        assert "Height (Z)" in html

    def test_height_uses_z_coordinate(self, html: str) -> None:
        """Height metric reads the Z coordinate (index 2) from position array."""
        assert re.search(
            r"posArray\[i \* 3 \+ 2\]", html
        ), "Height metric should read Z coordinate"


class TestDistanceMetric:
    """Distance metric colors vertices by distance from origin."""

    def test_distance_metric_option(self, html: str) -> None:
        """Distance metric is available in the metric selector."""
        assert 'value="distance"' in html
        assert "Distance from Center" in html

    def test_distance_uses_sqrt(self, html: str) -> None:
        """Distance metric computes sqrt(x^2 + y^2 + z^2)."""
        assert re.search(
            r"Math\.sqrt\(x \* x \+ y \* y \+ z \* z\)", html
        ), "Distance metric should use Euclidean distance"


class TestMetricUpdateWithoutRegeneration:
    """Changing metric updates colors without regenerating geometry."""

    def test_update_color_map_function(self, html: str) -> None:
        """updateColorMap function exists for recoloring in place."""
        assert "function updateColorMap" in html

    def test_metric_change_calls_update(self, html: str) -> None:
        """Metric selector change event calls updateColorMap."""
        assert re.search(
            r"colormap-metric.*updateColorMap", html, re.DOTALL
        ), "Metric change should trigger updateColorMap"

    def test_gradient_change_calls_update(self, html: str) -> None:
        """Gradient selector change event calls updateColorMap."""
        assert re.search(
            r"colormap-gradient.*updateColorMap", html, re.DOTALL
        ), "Gradient change should trigger updateColorMap"


class TestGradientPresets:
    """Gradient presets produce visually distinct color mappings."""

    def test_viridis_gradient(self, html: str) -> None:
        """Viridis gradient preset is defined."""
        assert "viridis" in html
        assert 'value="viridis"' in html

    def test_inferno_gradient(self, html: str) -> None:
        """Inferno gradient preset is defined."""
        assert "inferno" in html
        assert 'value="inferno"' in html

    def test_coolwarm_gradient(self, html: str) -> None:
        """Coolwarm gradient preset is defined."""
        assert "coolwarm" in html
        assert 'value="coolwarm"' in html

    def test_rainbow_gradient(self, html: str) -> None:
        """Rainbow gradient preset is defined."""
        assert "rainbow" in html
        assert 'value="rainbow"' in html

    def test_custom_gradient(self, html: str) -> None:
        """Custom gradient option with color pickers is available."""
        assert 'value="custom"' in html
        assert 'id="colormap-start-color"' in html
        assert 'id="colormap-end-color"' in html

    def test_gradient_data_structure(self, html: str) -> None:
        """Gradient color stops are defined as RGB arrays."""
        assert "COLORMAP_GRADIENTS" in html
        assert re.search(
            r"COLORMAP_GRADIENTS\s*=\s*\{", html
        ), "COLORMAP_GRADIENTS object should be defined"


class TestColorMapWithGeometryTypes:
    """Color mapping works with both point clouds and meshes."""

    def test_applies_to_points(self, html: str) -> None:
        """Color map applies vertex colors to Points objects."""
        assert re.search(
            r"PointsMaterial\(\{[^}]*vertexColors:\s*true", html
        ), "PointsMaterial with vertexColors should be created"

    def test_applies_to_meshes(self, html: str) -> None:
        """Color map applies vertex colors to Mesh objects."""
        assert re.search(
            r"MeshStandardMaterial\(\{[^}]*vertexColors:\s*true", html
        ), "MeshStandardMaterial with vertexColors should be created"

    def test_buffer_attribute_for_colors(self, html: str) -> None:
        """Vertex colors are set as a BufferAttribute on geometry."""
        assert re.search(
            r"setAttribute\('color'.*BufferAttribute", html, re.DOTALL
        ), "Color attribute should be set on geometry"


class TestExitColorMapMode:
    """Switching away from Color Map mode restores original material."""

    def test_exit_function_exists(self, html: str) -> None:
        """exitColorMapMode function is defined."""
        assert "function exitColorMapMode" in html

    def test_exit_restores_materials(self, html: str) -> None:
        """exitColorMapMode restores pre-colormap materials."""
        assert "preColormapMaterial" in html

    def test_exit_disposes_colormap_materials(self, html: str) -> None:
        """exitColorMapMode disposes color map materials."""
        assert re.search(
            r"child\.material\.dispose\(\)", html
        ), "Color map materials should be disposed on exit"

    def test_exit_removes_color_attribute(self, html: str) -> None:
        """exitColorMapMode removes the color attribute from geometry."""
        assert re.search(
            r"deleteAttribute\('color'\)", html
        ), "Color attribute should be removed on exit"

    def test_exit_hides_controls(self, html: str) -> None:
        """exitColorMapMode hides the color map controls."""
        assert re.search(
            r"exitColorMapMode.*colormap-controls.*none", html, re.DOTALL
        ), "Controls should be hidden when exiting colormap mode"

    def test_mode_transition_triggers_exit(self, html: str) -> None:
        """applyViewMode calls exitColorMapMode when leaving colormap."""
        assert re.search(
            r"viewMode !== 'colormap'.*colormapActive.*exitColorMapMode",
            html,
            re.DOTALL,
        ), "Switching away from colormap should call exitColorMapMode"


class TestColorMapNormalization:
    """Metric values are normalized to [0, 1] for gradient mapping."""

    def test_normalize_function(self, html: str) -> None:
        """normalizeValues function exists."""
        assert "function normalizeValues" in html

    def test_sample_gradient_function(self, html: str) -> None:
        """sampleGradient function exists for gradient interpolation."""
        assert "function sampleGradient" in html

    def test_build_vertex_colors_function(self, html: str) -> None:
        """buildVertexColors function computes final RGB array."""
        assert "function buildVertexColors" in html
