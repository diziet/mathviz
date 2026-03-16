"""Tests for colored axis labels and per-axis stretch controls."""

import re
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


# --- Axes toggle tests ---


class TestAxesToggle:
    """Tests for the Show Axes toggle checkbox."""

    def test_show_axes_toggle_exists(self, preview_html: str) -> None:
        """Preview HTML contains a Show Axes toggle checkbox."""
        assert 'id="show-axes"' in preview_html

    def test_axes_toggle_defaults_to_off(self, preview_html: str) -> None:
        """Axes toggle defaults to off (no checked attribute)."""
        match = re.search(r'<input[^>]*id="show-axes"[^>]*>', preview_html)
        assert match is not None
        tag = match.group(0)
        assert "checked" not in tag

    def test_enabling_axes_adds_helper(self, preview_html: str) -> None:
        """Enabling axes adds AxesHelper or equivalent to the scene."""
        assert "AxesHelper" in preview_html
        assert "createAxesWithLabels" in preview_html


# --- Axis label color tests ---


class TestAxisLabelColors:
    """Tests for axis label colors (X=red, Y=green, Z=blue)."""

    def test_axis_color_x_red(self, preview_html: str) -> None:
        """X axis label uses red color (#ff4444)."""
        assert "#ff4444" in preview_html

    def test_axis_color_y_green(self, preview_html: str) -> None:
        """Y axis label uses green color (#44ff44)."""
        assert "#44ff44" in preview_html

    def test_axis_color_z_blue(self, preview_html: str) -> None:
        """Z axis label uses blue color (#4488ff)."""
        assert "#4488ff" in preview_html

    def test_axis_colors_defined_in_constant(self, preview_html: str) -> None:
        """Axis colors are defined with correct XYZ mapping."""
        assert "x: '#ff4444'" in preview_html
        assert "y: '#44ff44'" in preview_html
        assert "z: '#4488ff'" in preview_html

    def test_axis_labels_are_sprites(self, preview_html: str) -> None:
        """Axis labels use Sprite + CanvasTexture for text rendering."""
        assert "Sprite" in preview_html
        assert "CanvasTexture" in preview_html

    def test_axis_label_text_xyz(self, preview_html: str) -> None:
        """Axis labels display X, Y, Z text."""
        assert "text: 'X'" in preview_html
        assert "text: 'Y'" in preview_html
        assert "text: 'Z'" in preview_html


# --- Stretch control tests ---


class TestStretchControls:
    """Tests for per-axis stretch (scale) controls."""

    def test_stretch_x_input_exists(self, preview_html: str) -> None:
        """Preview HTML contains stretch X scale input."""
        assert 'id="stretch-x"' in preview_html

    def test_stretch_y_input_exists(self, preview_html: str) -> None:
        """Preview HTML contains stretch Y scale input."""
        assert 'id="stretch-y"' in preview_html

    def test_stretch_z_input_exists(self, preview_html: str) -> None:
        """Preview HTML contains stretch Z scale input."""
        assert 'id="stretch-z"' in preview_html

    def test_stretch_number_inputs_exist(self, preview_html: str) -> None:
        """Preview HTML contains number inputs for stretch values."""
        assert 'id="stretch-x-num"' in preview_html
        assert 'id="stretch-y-num"' in preview_html
        assert 'id="stretch-z-num"' in preview_html

    def test_scale_inputs_default_to_one(self, preview_html: str) -> None:
        """Scale inputs default to 1.0."""
        for axis in ["x", "y", "z"]:
            match = re.search(
                rf'<input[^>]*id="stretch-{axis}"[^>]*>', preview_html
            )
            assert match is not None
            tag = match.group(0)
            assert 'value="1"' in tag or 'value="1.0"' in tag

    def test_reset_scale_button_exists(self, preview_html: str) -> None:
        """Preview HTML contains a Reset Scale button."""
        assert 'id="reset-scale-btn"' in preview_html
        assert "Reset Scale" in preview_html


# --- Stretch applies to geometry, not bounding box ---


class TestStretchBehavior:
    """Tests for stretch behavior on geometry vs bounding box."""

    def test_stretch_applies_to_mesh_group(self, preview_html: str) -> None:
        """Setting scale applies to meshGroup scale, not bounding box."""
        assert "meshGroup.scale.set" in preview_html

    def test_stretch_applies_to_cloud_points(self, preview_html: str) -> None:
        """Setting scale applies to cloudPoints scale."""
        assert "cloudPoints.scale.set" in preview_html

    def test_bounding_box_not_affected_by_stretch(
        self, preview_html: str
    ) -> None:
        """Bounding box is not affected by stretch values."""
        # applyStretch only touches meshGroup and cloudPoints, not bboxHelper
        stretch_fn = preview_html.split("function applyStretch")[1].split(
            "}"
        )[0]
        assert "bboxHelper" not in stretch_fn

    def test_stretch_persists_across_regeneration(
        self, preview_html: str
    ) -> None:
        """Stretch values persist across regeneration via applyStretch call."""
        # displayGenerateResult should call applyStretch after loading
        display_fn = preview_html.split("async function displayGenerateResult")[
            1
        ].split("/* ──")[0]
        assert "applyStretch()" in display_fn

    def test_stretch_state_initialized(self, preview_html: str) -> None:
        """Stretch state is initialized with default values."""
        assert "stretch: {x: 1, y: 1, z: 1}" in preview_html
