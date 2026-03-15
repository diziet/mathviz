"""Tests for the container/bounding box editor panel in the preview UI."""

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


# --- POST /api/generate with container params ---


class TestGenerateWithContainer:
    """Tests for POST /api/generate with custom container dimensions."""

    def test_custom_container_returns_200(self, client: TestClient) -> None:
        """POST /api/generate with custom container dimensions returns 200."""
        resp = client.post(
            "/api/generate",
            json={
                "generator": "torus",
                "seed": 42,
                "container": {
                    "width_mm": 80,
                    "height_mm": 80,
                    "depth_mm": 30,
                    "margin_x_mm": 3,
                    "margin_y_mm": 3,
                    "margin_z_mm": 3,
                },
            },
        )
        assert resp.status_code == 200
        data = resp.json()
        assert "geometry_id" in data

    def test_custom_container_changes_cache_key(self, client: TestClient) -> None:
        """Different container dimensions produce different geometry_ids."""
        resp_default = client.post(
            "/api/generate",
            json={"generator": "torus", "seed": 42},
        )
        resp_custom = client.post(
            "/api/generate",
            json={
                "generator": "torus",
                "seed": 42,
                "container": {
                    "width_mm": 50,
                    "height_mm": 50,
                    "depth_mm": 20,
                    "margin_x_mm": 2,
                    "margin_y_mm": 2,
                    "margin_z_mm": 2,
                },
            },
        )
        assert resp_default.json()["geometry_id"] != resp_custom.json()["geometry_id"]

    def test_default_container_matches_no_container(self, client: TestClient) -> None:
        """Omitting container uses default (100x100x40, 5mm margins)."""
        resp = client.post(
            "/api/generate",
            json={"generator": "torus", "seed": 42},
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["mesh_url"] is not None

    def test_invalid_margins_return_422(self, client: TestClient) -> None:
        """Margin >= half dimension returns 422 with error message."""
        resp = client.post(
            "/api/generate",
            json={
                "generator": "torus",
                "seed": 42,
                "container": {
                    "width_mm": 10,
                    "height_mm": 10,
                    "depth_mm": 10,
                    "margin_x_mm": 6,
                    "margin_y_mm": 1,
                    "margin_z_mm": 1,
                },
            },
        )
        assert resp.status_code == 422
        assert "margin" in resp.json()["detail"].lower()


# --- Usable volume calculation ---


class TestUsableVolume:
    """Tests for usable volume calculation correctness."""

    def test_usable_volume_formula(self) -> None:
        """Usable volume is dimension - 2*margin per axis."""
        from mathviz.core.container import Container

        c = Container(
            width_mm=100, height_mm=80, depth_mm=40,
            margin_x_mm=5, margin_y_mm=10, margin_z_mm=3,
        )
        assert c.usable_volume == (90.0, 60.0, 34.0)

    def test_usable_volume_default_container(self) -> None:
        """Default container (100x100x40, 5mm margins) has usable 90x90x30."""
        from mathviz.core.container import Container

        c = Container.with_uniform_margin()
        assert c.usable_volume == (90.0, 90.0, 30.0)


# --- HTML panel presence ---


class TestContainerEditorHTML:
    """Tests that the preview HTML contains the container editor panel."""

    def test_html_contains_container_panel(self, preview_html: str) -> None:
        """Preview HTML contains the container editor panel."""
        assert 'id="container-panel"' in preview_html

    def test_html_contains_dimension_inputs(self, preview_html: str) -> None:
        """Preview HTML contains W, H, D dimension inputs."""
        assert 'id="dim-w"' in preview_html
        assert 'id="dim-h"' in preview_html
        assert 'id="dim-d"' in preview_html

    def test_html_contains_margin_inputs(self, preview_html: str) -> None:
        """Preview HTML contains margin X, Y, Z inputs."""
        assert 'id="margin-x"' in preview_html
        assert 'id="margin-y"' in preview_html
        assert 'id="margin-z"' in preview_html

    def test_html_contains_usable_volume_display(self, preview_html: str) -> None:
        """Preview HTML contains usable volume display."""
        assert 'id="usable-volume"' in preview_html
        assert "Usable:" in preview_html

    def test_html_contains_apply_button(self, preview_html: str) -> None:
        """Preview HTML contains the Apply button."""
        assert 'id="container-apply-btn"' in preview_html
        assert "Apply" in preview_html

    def test_html_contains_reset_button(self, preview_html: str) -> None:
        """Preview HTML contains the Reset button."""
        assert 'id="container-reset-btn"' in preview_html
        assert "Reset" in preview_html

    def test_html_apply_triggers_generate(self, preview_html: str) -> None:
        """Apply button triggers a POST /api/generate with container params."""
        assert "container-apply-btn" in preview_html
        assert "loadFromAPI" in preview_html
        assert "getContainerParams" in preview_html

    def test_html_reset_restores_defaults(self, preview_html: str) -> None:
        """Reset button restores default values in the input fields."""
        assert "container-reset-btn" in preview_html
        assert "resetContainerDefaults" in preview_html

    def test_html_has_uniform_margin_checkbox(self, preview_html: str) -> None:
        """Preview HTML contains a uniform margin checkbox."""
        assert 'id="uniform-margin"' in preview_html
