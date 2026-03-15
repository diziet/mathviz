"""Tests for parameter and resolution editor UI in preview HTML."""

from typing import Generator

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


# --- Parameter editor section ---


class TestParamEditorSection:
    """Preview HTML contains a parameter editor section with input elements."""

    def test_param_panel_exists(self, preview_html: str) -> None:
        """Parameter editor panel is present in the HTML."""
        assert 'id="param-panel"' in preview_html

    def test_param_fields_container_exists(self, preview_html: str) -> None:
        """Container for dynamically created parameter inputs exists."""
        assert 'id="param-fields"' in preview_html

    def test_create_param_input_function(self, preview_html: str) -> None:
        """JavaScript contains createParamInput for dynamic input creation."""
        assert "createParamInput" in preview_html

    def test_param_inputs_have_data_attributes(self, preview_html: str) -> None:
        """Parameter inputs use data-param-name for value collection."""
        assert "dataset.paramName" in preview_html or "data-param-name" in preview_html
        assert "dataset.paramType" in preview_html or "data-param-type" in preview_html

    def test_boolean_params_use_checkbox(self, preview_html: str) -> None:
        """Boolean parameters render as checkboxes."""
        assert "type='checkbox'" in preview_html or 'type = \'checkbox\'' in preview_html or "checkbox" in preview_html

    def test_number_params_use_number_input(self, preview_html: str) -> None:
        """Numeric parameters render as number inputs."""
        assert "type='number'" in preview_html or "type = 'number'" in preview_html or "'number'" in preview_html


# --- Resolution editor section ---


class TestResolutionEditorSection:
    """Preview HTML contains a resolution editor section."""

    def test_resolution_section_exists(self, preview_html: str) -> None:
        """Resolution section container is present."""
        assert 'id="param-resolution-section"' in preview_html

    def test_resolution_fields_container_exists(self, preview_html: str) -> None:
        """Resolution fields container is present."""
        assert 'id="param-resolution-fields"' in preview_html

    def test_resolution_title(self, preview_html: str) -> None:
        """Resolution section has a title."""
        assert "Resolution" in preview_html

    def test_resolution_descriptions_shown(self, preview_html: str) -> None:
        """Resolution param descriptions are displayed as hints."""
        assert "param-hint" in preview_html

    def test_resolution_high_value_warning(self, preview_html: str) -> None:
        """High resolution values trigger a visual warning."""
        assert "resolution-warning" in preview_html


# --- API returns params and resolution ---


class TestParamsAPIResponse:
    """Fetching /api/generators/lorenz/params returns params and resolution."""

    def test_lorenz_params_returned(self, client: TestClient) -> None:
        """Lorenz params endpoint returns params dict."""
        resp = client.get("/api/generators/lorenz/params")
        assert resp.status_code == 200
        data = resp.json()
        assert "params" in data
        assert "sigma" in data["params"]
        assert "rho" in data["params"]

    def test_lorenz_resolution_returned(self, client: TestClient) -> None:
        """Lorenz params endpoint returns resolution dict."""
        resp = client.get("/api/generators/lorenz/params")
        data = resp.json()
        assert "resolution" in data
        assert "integration_steps" in data["resolution"]

    def test_lorenz_descriptions_returned(self, client: TestClient) -> None:
        """Lorenz params endpoint returns descriptions dict."""
        resp = client.get("/api/generators/lorenz/params")
        data = resp.json()
        assert "descriptions" in data

    def test_torus_params_and_resolution(self, client: TestClient) -> None:
        """Torus returns both params and resolution defaults."""
        resp = client.get("/api/generators/torus/params")
        assert resp.status_code == 200
        data = resp.json()
        assert "params" in data
        assert "resolution" in data
        assert "grid_resolution" in data["resolution"]


# --- Dynamic parameter input creation ---


class TestDynamicParamInputs:
    """Parameter inputs are dynamically created based on generator defaults."""

    def test_populate_param_fields_function(self, preview_html: str) -> None:
        """populateParamFields function creates inputs from API data."""
        assert "populateParamFields" in preview_html

    def test_fetch_and_populate_function(self, preview_html: str) -> None:
        """fetchAndPopulateParams fetches and populates the panel."""
        assert "fetchAndPopulateParams" in preview_html

    def test_collect_param_values_function(self, preview_html: str) -> None:
        """collectParamValues extracts typed values from inputs."""
        assert "collectParamValues" in preview_html

    def test_int_params_have_step_1(self, preview_html: str) -> None:
        """Integer params use step='1' for the number input."""
        assert "'1'" in preview_html

    def test_float_params_have_step_01(self, preview_html: str) -> None:
        """Float params use step='0.1' for the number input."""
        assert "'0.1'" in preview_html


# --- Apply button sends params and resolution ---


class TestApplyButton:
    """Apply button sends both params and resolution in POST body."""

    def test_apply_button_exists(self, preview_html: str) -> None:
        """Apply button element is present."""
        assert 'id="param-apply-btn"' in preview_html

    def test_apply_collects_params(self, preview_html: str) -> None:
        """Apply handler calls getEditorParams."""
        assert "getEditorParams" in preview_html

    def test_apply_collects_resolution(self, preview_html: str) -> None:
        """Apply handler calls getEditorResolution."""
        assert "getEditorResolution" in preview_html

    def test_apply_sends_both_in_body(self, preview_html: str) -> None:
        """POST body includes both params and resolution keys."""
        assert "params: genParams" in preview_html
        assert "resolution" in preview_html

    def test_apply_shows_loading_indicator(self, preview_html: str) -> None:
        """Apply shows a loading indicator during generation."""
        assert 'id="param-loading"' in preview_html

    def test_apply_with_resolution_via_api(self, client: TestClient) -> None:
        """POST /api/generate accepts both params and resolution."""
        resp = client.post(
            "/api/generate",
            json={
                "generator": "torus",
                "seed": 42,
                "params": {"major_radius": 1.5},
                "resolution": {"grid_resolution": 16},
            },
        )
        assert resp.status_code == 200
        data = resp.json()
        assert "geometry_id" in data
        assert data["mesh_url"] is not None


# --- Reset button restores defaults ---


class TestResetButton:
    """Reset button restores default values."""

    def test_reset_button_exists(self, preview_html: str) -> None:
        """Reset button element is present."""
        assert 'id="param-reset-btn"' in preview_html

    def test_reset_defaults_function(self, preview_html: str) -> None:
        """resetParamDefaults function is defined."""
        assert "resetParamDefaults" in preview_html

    def test_reset_uses_cached_defaults(self, preview_html: str) -> None:
        """Reset restores from cachedParamDefaults."""
        assert "cachedParamDefaults" in preview_html

    def test_reset_clears_error(self, preview_html: str) -> None:
        """Reset hides the error display."""
        assert "paramError" in preview_html or "param-error" in preview_html


# --- Generator switch clears and repopulates ---


class TestGeneratorSwitch:
    """Switching generators clears and repopulates the parameter panel."""

    def test_select_generator_calls_fetch(self, preview_html: str) -> None:
        """selectGenerator triggers fetchAndPopulateParams."""
        assert "selectGenerator" in preview_html
        assert "fetchAndPopulateParams" in preview_html

    def test_populate_clears_fields(self, preview_html: str) -> None:
        """populateParamFields clears existing fields first."""
        assert "innerHTML = ''" in preview_html or "innerHTML=''" in preview_html

    def test_different_generators_have_different_params(
        self, client: TestClient
    ) -> None:
        """Different generators return different parameter sets."""
        lorenz = client.get("/api/generators/lorenz/params").json()
        torus = client.get("/api/generators/torus/params").json()
        assert set(lorenz["params"].keys()) != set(torus["params"].keys())

    def test_different_generators_have_different_resolution(
        self, client: TestClient
    ) -> None:
        """Different generators return different resolution params."""
        lorenz = client.get("/api/generators/lorenz/params").json()
        torus = client.get("/api/generators/torus/params").json()
        assert set(lorenz["resolution"].keys()) != set(torus["resolution"].keys())


# --- Info panel updates after generation ---


class TestInfoPanelUpdates:
    """Info panel updates with vertex/face/point counts after generation."""

    def test_info_vertices_element(self, preview_html: str) -> None:
        """Info panel has vertex count element."""
        assert 'id="info-vertices"' in preview_html

    def test_info_faces_element(self, preview_html: str) -> None:
        """Info panel has face count element."""
        assert 'id="info-faces"' in preview_html

    def test_info_points_element(self, preview_html: str) -> None:
        """Info panel has point count element."""
        assert 'id="info-points"' in preview_html

    def test_update_info_function(self, preview_html: str) -> None:
        """updateInfo function updates the info panel."""
        assert "updateInfo" in preview_html

    def test_generate_returns_geometry_urls(self, client: TestClient) -> None:
        """Generation returns mesh/cloud URLs for the client to load."""
        resp = client.post(
            "/api/generate",
            json={
                "generator": "torus",
                "seed": 42,
                "resolution": {"grid_resolution": 16},
            },
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["mesh_url"] is not None or data["cloud_url"] is not None
