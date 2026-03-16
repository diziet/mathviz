"""Tests for point cloud density slider.

Verifies that the density slider is present in the preview HTML,
defaults to 1.0, and correctly controls point cloud display density.
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


def _extract_js_function(html: str, func_name: str) -> str:
    """Extract a JS function body from the HTML, or fail."""
    match = re.search(
        rf"function {re.escape(func_name)}\(.*?^}}",
        html,
        re.MULTILINE | re.DOTALL,
    )
    assert match is not None, f"JS function {func_name!r} not found in HTML"
    return match.group(0)


def _extract_element(html: str, element_id: str) -> str:
    """Extract an HTML element tag by id, or fail."""
    match = re.search(rf'<[^>]*id="{re.escape(element_id)}"[^>]*>', html)
    assert match is not None, f"Element {element_id!r} not found in HTML"
    return match.group(0)


class TestDensitySliderPresent:
    """Tests that the density slider input exists in the HTML."""

    def test_density_slider_input_exists(self, client: TestClient) -> None:
        """Preview HTML contains a density slider input element."""
        html = _get_html(client)
        assert 'id="density-slider"' in html

    def test_density_slider_is_range_input(self, client: TestClient) -> None:
        """Density slider is a range input with correct attributes."""
        html = _get_html(client)
        tag = _extract_element(html, "density-slider")
        assert 'type="range"' in tag
        assert 'min="0.01"' in tag
        assert 'max="1"' in tag
        assert 'step="0.01"' in tag

    def test_density_label_exists(self, client: TestClient) -> None:
        """A section title labeled 'Density' exists."""
        html = _get_html(client)
        assert 'id="density-control"' in html
        assert ">Density<" in html


class TestDensitySliderDefault:
    """Tests that the density slider defaults to 1.0 (full density)."""

    def test_slider_default_value_is_1(self, client: TestClient) -> None:
        """Slider HTML attribute defaults to value 1."""
        html = _get_html(client)
        tag = _extract_element(html, "density-slider")
        assert 'value="1"' in tag

    def test_state_density_defaults_to_1(self, client: TestClient) -> None:
        """JavaScript state initializes density to 1.0."""
        html = _get_html(client)
        assert "density: 1.0" in html


class TestDensityHalfPoints:
    """Tests that setting density to 0.5 shows approximately half the points."""

    def test_density_filter_uses_every_nth_sampling(
        self, client: TestClient,
    ) -> None:
        """applyDensityFilter computes displayCount from density fraction."""
        html = _get_html(client)
        assert "Math.round(totalPoints * density)" in html

    def test_density_filter_copies_into_buffer(
        self, client: TestClient,
    ) -> None:
        """applyDensityFilter copies sampled points into the pre-allocated buffer."""
        html = _get_html(client)
        body = _extract_js_function(html, "applyDensityFilter")
        assert "posAttr.array[" in body
        assert "posAttr.needsUpdate = true" in body


class TestDensityOnePercent:
    """Tests that setting density to 0.01 shows approximately 1% of points."""

    def test_density_min_clamps_to_one_point(self, client: TestClient) -> None:
        """At minimum density, at least 1 point is always shown."""
        html = _get_html(client)
        assert "Math.max(1," in html


class TestDensityPersistsAcrossRegeneration:
    """Tests that density value persists across regeneration."""

    def test_density_not_reset_in_clear_scene(self, client: TestClient) -> None:
        """clearScene does not reset state.density."""
        html = _get_html(client)
        body = _extract_js_function(html, "clearScene")
        assert "state.density" not in body

    def test_density_applied_after_display_cloud(
        self, client: TestClient,
    ) -> None:
        """displayCloud calls applyDensityFilter to apply current density."""
        html = _get_html(client)
        body = _extract_js_function(html, "displayCloud")
        assert "applyDensityFilter()" in body

    def test_density_in_capture_ui_state(self, client: TestClient) -> None:
        """captureUiState includes density in the saved state."""
        html = _get_html(client)
        body = _extract_js_function(html, "captureUiState")
        assert "density:" in body

    def test_density_in_restore_ui_state(self, client: TestClient) -> None:
        """restoreUiState restores density from saved state."""
        html = _get_html(client)
        body = _extract_js_function(html, "restoreUiState")
        assert "ui.density" in body


class TestDensitySliderHiddenWhenNotPoints:
    """Tests that the density slider is hidden when view mode is not points."""

    def test_density_slider_initially_hidden(self, client: TestClient) -> None:
        """Density slider starts with display:none (shown when view is points)."""
        html = _get_html(client)
        tag = _extract_element(html, "density-slider")
        assert "display:none" in tag

    def test_density_visibility_updates_on_view_mode_change(
        self, client: TestClient,
    ) -> None:
        """View mode change listener calls updateDensitySliderVisibility."""
        html = _get_html(client)
        assert "updateDensitySliderVisibility()" in html

    def test_visibility_function_checks_points_mode(
        self, client: TestClient,
    ) -> None:
        """updateDensitySliderVisibility checks if viewMode is point-like."""
        html = _get_html(client)
        body = _extract_js_function(html, "updateDensitySliderVisibility")
        assert "isPointLikeMode(state.viewMode)" in body
        assert "density-slider" in body
        assert "density-control" in body


class TestInfoPanelDensity:
    """Tests that info panel shows filtered vs total point count."""

    def test_update_density_info_function_exists(
        self, client: TestClient,
    ) -> None:
        """updateDensityInfo function is defined."""
        html = _get_html(client)
        assert "function updateDensityInfo(" in html

    def test_density_info_shows_fraction_format(
        self, client: TestClient,
    ) -> None:
        """updateDensityInfo formats as 'displayed / total (pct%)'."""
        html = _get_html(client)
        body = _extract_js_function(html, "updateDensityInfo")
        assert "toLocaleString()" in body
        assert "info-points" in body
        assert "Points:" in body

    def test_density_info_updates_info_state(
        self, client: TestClient,
    ) -> None:
        """updateDensityInfo syncs infoState.points to avoid stale cache."""
        html = _get_html(client)
        body = _extract_js_function(html, "updateDensityInfo")
        assert "infoState.points" in body


class TestDensityPerformance:
    """Tests for density slider performance optimizations."""

    def test_dirty_flag_coalesces_raf(self, client: TestClient) -> None:
        """Slider uses a dirty flag to avoid redundant rAF calls."""
        html = _get_html(client)
        assert "densityDirty" in html

    def test_dynamic_draw_usage_buffer(self, client: TestClient) -> None:
        """displayCloud sets DynamicDrawUsage for GPU buffer reuse."""
        html = _get_html(client)
        body = _extract_js_function(html, "displayCloud")
        assert "DynamicDrawUsage" in body

    def test_needs_update_instead_of_set_attribute(
        self, client: TestClient,
    ) -> None:
        """applyDensityFilter uses needsUpdate instead of creating new buffers."""
        html = _get_html(client)
        body = _extract_js_function(html, "applyDensityFilter")
        assert "posAttr.needsUpdate = true" in body
        assert "new THREE.BufferAttribute" not in body


class TestDensityCompareMode:
    """Tests that density propagates to compare-mode panels."""

    def test_compare_mode_density_propagation(
        self, client: TestClient,
    ) -> None:
        """Slider listener calls updateDensityInPanels in compare mode."""
        html = _get_html(client)
        assert "updateDensityInPanels()" in html

    def test_update_density_in_panels_function_exists(
        self, client: TestClient,
    ) -> None:
        """updateDensityInPanels function is defined."""
        html = _get_html(client)
        assert "function updateDensityInPanels()" in html
