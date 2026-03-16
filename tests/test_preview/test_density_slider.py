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


class TestDensitySliderPresent:
    """Tests that the density slider input exists in the HTML."""

    def test_density_slider_input_exists(self, client: TestClient) -> None:
        """Preview HTML contains a density slider input element."""
        html = _get_html(client)
        assert 'id="density-slider"' in html

    def test_density_slider_is_range_input(self, client: TestClient) -> None:
        """Density slider is a range input with correct attributes."""
        html = _get_html(client)
        match = re.search(r'<input[^>]*id="density-slider"[^>]*>', html)
        assert match is not None
        tag = match.group(0)
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
        match = re.search(r'<input[^>]*id="density-slider"[^>]*>', html)
        assert match is not None
        assert 'value="1"' in match.group(0)

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

    def test_density_filter_creates_subset(self, client: TestClient) -> None:
        """applyDensityFilter creates a filtered Float32Array subset."""
        html = _get_html(client)
        assert "new Float32Array(displayCount * 3)" in html


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
        clear_fn = re.search(
            r"function clearScene\(\).*?^}",
            html,
            re.MULTILINE | re.DOTALL,
        )
        assert clear_fn is not None
        assert "state.density" not in clear_fn.group(0)

    def test_density_applied_after_display_cloud(
        self, client: TestClient,
    ) -> None:
        """displayCloud calls applyDensityFilter to apply current density."""
        html = _get_html(client)
        display_fn = re.search(
            r"function displayCloud\(.*?^}",
            html,
            re.MULTILINE | re.DOTALL,
        )
        assert display_fn is not None
        assert "applyDensityFilter()" in display_fn.group(0)

    def test_density_in_capture_ui_state(self, client: TestClient) -> None:
        """captureUiState includes density in the saved state."""
        html = _get_html(client)
        capture_fn = re.search(
            r"function captureUiState\(\).*?^}",
            html,
            re.MULTILINE | re.DOTALL,
        )
        assert capture_fn is not None
        assert "density:" in capture_fn.group(0)

    def test_density_in_restore_ui_state(self, client: TestClient) -> None:
        """restoreUiState restores density from saved state."""
        html = _get_html(client)
        restore_fn = re.search(
            r"function restoreUiState\(.*?^}",
            html,
            re.MULTILINE | re.DOTALL,
        )
        assert restore_fn is not None
        assert "ui.density" in restore_fn.group(0)


class TestDensitySliderHiddenWhenNotPoints:
    """Tests that the density slider is hidden when view mode is not points."""

    def test_density_slider_initially_hidden(self, client: TestClient) -> None:
        """Density slider starts with display:none (shown when view is points)."""
        html = _get_html(client)
        match = re.search(r'<input[^>]*id="density-slider"[^>]*>', html)
        assert match is not None
        assert "display:none" in match.group(0)

    def test_density_visibility_updates_on_view_mode_change(
        self, client: TestClient,
    ) -> None:
        """View mode change listener calls updateDensitySliderVisibility."""
        html = _get_html(client)
        assert "updateDensitySliderVisibility()" in html

    def test_visibility_function_checks_points_mode(
        self, client: TestClient,
    ) -> None:
        """updateDensitySliderVisibility checks if viewMode is 'points'."""
        html = _get_html(client)
        vis_fn = re.search(
            r"function updateDensitySliderVisibility\(\).*?^}",
            html,
            re.MULTILINE | re.DOTALL,
        )
        assert vis_fn is not None
        body = vis_fn.group(0)
        assert "state.viewMode === 'points'" in body
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
        info_fn = re.search(
            r"function updateDensityInfo\(.*?^}",
            html,
            re.MULTILINE | re.DOTALL,
        )
        assert info_fn is not None
        body = info_fn.group(0)
        assert "toLocaleString()" in body
        assert "info-points" in body
        assert "Points:" in body
