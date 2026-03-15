"""Tests for the parameter editor panel in the preview UI."""

from typing import Any, Generator

import pytest
from fastapi.testclient import TestClient

from mathviz.core.generator import register
from mathviz.generators.attractors.lorenz import LorenzGenerator
from mathviz.generators.fractals.mandelbulb import MandelbulbGenerator
from mathviz.generators.parametric.torus import TorusGenerator
from mathviz.preview.server import app, reset_cache


def _ensure_generators_registered() -> None:
    """Re-register test generators if missing from the registry."""
    import mathviz.core.generator as gen_mod

    gen_mod._discovered = True
    for cls in (TorusGenerator, LorenzGenerator, MandelbulbGenerator):
        if cls.name not in gen_mod._alias_map:
            register(cls)


# --- Fixtures ---


@pytest.fixture(autouse=True)
def _setup() -> Generator[None, None, None]:
    """Ensure generators are registered and cache is clean."""
    _ensure_generators_registered()
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


# --- GET /api/generators/{name}/params ---


class TestGetGeneratorParams:
    """Tests for GET /api/generators/{name}/params."""

    def test_lorenz_returns_default_params(self, client: TestClient) -> None:
        """Lorenz params include sigma, rho, beta, transient_steps."""
        resp = client.get("/api/generators/lorenz/params")
        assert resp.status_code == 200
        data = resp.json()
        params = data["params"]
        assert params["sigma"] == 10.0
        assert params["rho"] == 28.0
        assert abs(params["beta"] - 8.0 / 3.0) < 0.01
        assert params["transient_steps"] == 1000

    def test_mandelbulb_returns_default_params(self, client: TestClient) -> None:
        """Mandelbulb params include power, max_iterations, extent."""
        resp = client.get("/api/generators/mandelbulb/params")
        assert resp.status_code == 200
        data = resp.json()
        params = data["params"]
        assert params["power"] == 8.0
        assert params["max_iterations"] == 10
        assert params["extent"] == 1.5

    def test_unknown_generator_returns_404(self, client: TestClient) -> None:
        """Unknown generator returns 404."""
        resp = client.get("/api/generators/nonexistent_xyz/params")
        assert resp.status_code == 404
        assert "not found" in resp.json()["detail"].lower()

    def test_lorenz_resolution_params_included(self, client: TestClient) -> None:
        """Lorenz resolution includes integration_steps with default."""
        resp = client.get("/api/generators/lorenz/params")
        data = resp.json()
        assert "resolution" in data
        assert data["resolution"]["integration_steps"] == 100_000

    def test_mandelbulb_resolution_params_included(self, client: TestClient) -> None:
        """Mandelbulb resolution includes voxel_resolution."""
        resp = client.get("/api/generators/mandelbulb/params")
        data = resp.json()
        assert "resolution" in data
        assert data["resolution"]["voxel_resolution"] == 128

    def test_descriptions_included(self, client: TestClient) -> None:
        """Resolution descriptions are included in the response."""
        resp = client.get("/api/generators/lorenz/params")
        data = resp.json()
        assert "descriptions" in data
        assert "integration_steps" in data["descriptions"]
        assert len(data["descriptions"]["integration_steps"]) > 0


# --- POST /api/generate with custom params ---


class TestGenerateWithCustomParams:
    """Tests for POST /api/generate with custom params."""

    def test_custom_params_different_geometry(self, client: TestClient) -> None:
        """Custom params produce different geometry_id than defaults."""
        resp_default = client.post(
            "/api/generate",
            json={
                "generator": "torus",
                "seed": 42,
                "resolution": {"grid_resolution": 16},
            },
        )
        resp_custom = client.post(
            "/api/generate",
            json={
                "generator": "torus",
                "seed": 42,
                "params": {"major_radius": 2.0, "minor_radius": 0.8},
                "resolution": {"grid_resolution": 16},
            },
        )
        assert resp_default.status_code == 200
        assert resp_custom.status_code == 200
        assert resp_default.json()["geometry_id"] != resp_custom.json()["geometry_id"]

    def test_invalid_params_returns_error(self, client: TestClient) -> None:
        """Invalid params return 400 with descriptive error message."""
        resp = client.post(
            "/api/generate",
            json={
                "generator": "lorenz",
                "seed": 42,
                "params": {"sigma": -1.0},
            },
        )
        assert resp.status_code == 400
        detail = resp.json()["detail"]
        assert "sigma" in detail.lower()
        assert "positive" in detail.lower()


# --- HTML panel presence ---


class TestParamEditorHTML:
    """Tests that the preview HTML contains the parameter editor panel."""

    def test_html_contains_param_panel(self, preview_html: str) -> None:
        """Preview HTML contains the parameter editor panel."""
        assert 'id="param-panel"' in preview_html

    def test_html_contains_param_fields_container(self, preview_html: str) -> None:
        """Preview HTML contains the param fields container."""
        assert 'id="param-fields"' in preview_html

    def test_html_contains_resolution_section(self, preview_html: str) -> None:
        """Preview HTML contains the resolution section."""
        assert 'id="param-resolution-fields"' in preview_html

    def test_html_contains_apply_button(self, preview_html: str) -> None:
        """Preview HTML contains the Apply button for params."""
        assert 'id="param-apply-btn"' in preview_html

    def test_html_contains_reset_button(self, preview_html: str) -> None:
        """Preview HTML contains the Reset button for params."""
        assert 'id="param-reset-btn"' in preview_html

    def test_html_fetches_params_on_generator_change(self, preview_html: str) -> None:
        """Generator selection triggers param fetch."""
        assert "fetchAndPopulateParams" in preview_html

    def test_html_apply_sends_params(self, preview_html: str) -> None:
        """Apply button collects and sends params in POST body."""
        assert "getEditorParams" in preview_html
        assert "getEditorResolution" in preview_html

    def test_html_reset_restores_defaults(self, preview_html: str) -> None:
        """Reset button restores default param values."""
        assert "resetParamDefaults" in preview_html

    def test_html_shows_error_display(self, preview_html: str) -> None:
        """HTML contains error display element."""
        assert 'id="param-error"' in preview_html
