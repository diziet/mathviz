"""Tests for usable volume display correctness on page load."""

import re
from typing import Generator

import pytest
from fastapi.testclient import TestClient

from mathviz.core.container import Container
from mathviz.core.generator import register
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


@pytest.fixture
def preview_html(client: TestClient) -> str:
    """Fetch the preview HTML once for assertion-only tests."""
    resp = client.get("/")
    assert resp.status_code == 200
    return resp.text


# --- Initial HTML usable volume text ---


class TestInitialUsableVolumeText:
    """Initial HTML usable volume text matches current container defaults."""

    def test_initial_text_matches_defaults(self, preview_html: str) -> None:
        """Hardcoded usable volume text matches 100x100x100 with 5mm margins."""
        c = Container()
        ux, uy, uz = c.usable_volume
        expected = f"Usable: {ux:g} x {uy:g} x {uz:g} mm"
        assert expected in preview_html

    def test_no_stale_30mm_depth(self, preview_html: str) -> None:
        """Old 40mm-depth usable volume text (30mm) is not present."""
        assert "90 x 90 x 30" not in preview_html


# --- updateUsableVolume called on init ---


class TestUsableVolumeInitCall:
    """updateUsableVolume() is called during page initialization."""

    def test_update_called_in_init_section(self, preview_html: str) -> None:
        """updateUsableVolume() appears in the init section of the script."""
        init_match = re.search(
            r"/\* ── Init.*?\*/\s*updateUsableVolume\(\)", preview_html
        )
        assert init_match is not None, (
            "updateUsableVolume() should be called in the Init section"
        )

    def test_update_function_exists(self, preview_html: str) -> None:
        """The updateUsableVolume function is defined in the HTML."""
        assert "function updateUsableVolume()" in preview_html


# --- Usable volume calculation correctness ---


class TestUsableVolumeCalculation:
    """Usable volume calculation is correct: dimension - 2 * margin per axis."""

    def test_usable_volume_formula(self) -> None:
        """Usable volume is dimension - 2*margin per axis."""
        c = Container(
            width_mm=120, height_mm=80, depth_mm=60,
            margin_x_mm=10, margin_y_mm=5, margin_z_mm=3,
        )
        assert c.usable_volume == (100.0, 70.0, 54.0)

    def test_default_container_usable_volume(self) -> None:
        """Default container (100x100x100, 5mm margins) yields 90x90x90."""
        c = Container()
        assert c.usable_volume == (90.0, 90.0, 90.0)

    def test_js_formula_matches_python(self, preview_html: str) -> None:
        """JS updateUsableVolume uses same formula as Python Container."""
        assert "cp.width_mm - 2 * cp.margin_x_mm" in preview_html
        assert "cp.height_mm - 2 * cp.margin_y_mm" in preview_html
        assert "cp.depth_mm - 2 * cp.margin_z_mm" in preview_html


# --- Depth input updates usable volume ---


class TestDepthInputUpdatesVolume:
    """Changing depth input updates the usable volume display."""

    def test_depth_input_has_event_listener(self, preview_html: str) -> None:
        """Depth input (dim-d) triggers updateUsableVolume on input event."""
        assert 'id="dim-d"' in preview_html
        assert "dimD" in preview_html
        assert "[dimW, dimH, dimD]" in preview_html
        assert "addEventListener('input', updateUsableVolume)" in preview_html

    def test_margin_input_triggers_update(self, preview_html: str) -> None:
        """Margin inputs trigger updateUsableVolume via onMarginInput."""
        assert "onMarginInput" in preview_html
        assert "updateUsableVolume()" in preview_html
