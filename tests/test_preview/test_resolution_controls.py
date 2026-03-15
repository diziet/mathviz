"""Tests for resolution controls in the preview UI."""

from typing import Any, Generator

import pytest
from fastapi.testclient import TestClient

from mathviz.core.generator import list_generators, register
from mathviz.generators.attractors.lorenz import LorenzGenerator
from mathviz.generators.fractals.mandelbulb import MandelbulbGenerator
from mathviz.generators.parametric.torus import TorusGenerator
from mathviz.preview.server import app, get_cache, reset_cache


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


# --- get_default_resolution() unit tests ---


class TestGetDefaultResolution:
    """Tests for get_default_resolution() on generator instances."""

    def test_lorenz_returns_integration_steps(self) -> None:
        """Lorenz returns integration_steps=100000."""
        gen = LorenzGenerator()
        res = gen.get_default_resolution()
        assert res == {"integration_steps": 100_000}

    def test_mandelbulb_returns_voxel_resolution(self) -> None:
        """Mandelbulb returns voxel_resolution=128."""
        gen = MandelbulbGenerator()
        res = gen.get_default_resolution()
        assert res == {"voxel_resolution": 128}

    def test_torus_returns_grid_resolution(self) -> None:
        """Torus returns grid_resolution=128."""
        gen = TorusGenerator()
        res = gen.get_default_resolution()
        assert res == {"grid_resolution": 128}

    def test_all_generators_with_resolution_params_return_defaults(self) -> None:
        """Every generator with resolution_params returns values from get_default_resolution()."""
        all_metas = list_generators()
        for meta in all_metas:
            if not meta.resolution_params:
                continue
            instance = meta.generator_class.create()
            defaults = instance.get_default_resolution()
            for param_name in meta.resolution_params:
                assert param_name in defaults, (
                    f"{meta.name}: resolution param {param_name!r} "
                    f"not in get_default_resolution() result {defaults}"
                )
                assert isinstance(defaults[param_name], (int, float)), (
                    f"{meta.name}: resolution param {param_name!r} "
                    f"has non-numeric default {defaults[param_name]!r}"
                )


# --- API endpoint tests ---


class TestResolutionAPI:
    """Tests for resolution defaults via the API."""

    def test_params_endpoint_returns_resolution_defaults(
        self, client: TestClient,
    ) -> None:
        """GET /api/generators/{name}/params includes resolution defaults."""
        resp = client.get("/api/generators/lorenz/params")
        assert resp.status_code == 200
        data = resp.json()
        assert "resolution" in data
        assert data["resolution"]["integration_steps"] == 100_000
        assert "descriptions" in data
        assert "integration_steps" in data["descriptions"]

    def test_torus_params_endpoint_returns_resolution(
        self, client: TestClient,
    ) -> None:
        """Torus resolution defaults include grid_resolution."""
        resp = client.get("/api/generators/torus/params")
        assert resp.status_code == 200
        data = resp.json()
        assert data["resolution"]["grid_resolution"] == 128


# --- POST /api/generate resolution behavior ---


class TestResolutionGenerate:
    """Tests for resolution-aware generation."""

    def test_higher_resolution_produces_more_vertices(
        self, client: TestClient,
    ) -> None:
        """Higher grid_resolution produces more vertices."""
        resp_low = client.post(
            "/api/generate",
            json={
                "generator": "torus",
                "seed": 42,
                "resolution": {"grid_resolution": 8},
            },
        )
        resp_high = client.post(
            "/api/generate",
            json={
                "generator": "torus",
                "seed": 42,
                "resolution": {"grid_resolution": 32},
            },
        )
        assert resp_low.status_code == 200
        assert resp_high.status_code == 200

        cache = get_cache()
        low_entry = cache.get(resp_low.json()["geometry_id"])
        high_entry = cache.get(resp_high.json()["geometry_id"])
        assert low_entry is not None
        assert high_entry is not None

        low_verts = low_entry.math_object.mesh.vertices.shape[0]
        high_verts = high_entry.math_object.mesh.vertices.shape[0]
        assert high_verts > low_verts

    def test_lower_resolution_produces_fewer_vertices(
        self, client: TestClient,
    ) -> None:
        """Lower grid_resolution produces fewer vertices than default."""
        resp_default = client.post(
            "/api/generate",
            json={
                "generator": "torus",
                "seed": 42,
                "resolution": {"grid_resolution": 32},
            },
        )
        resp_low = client.post(
            "/api/generate",
            json={
                "generator": "torus",
                "seed": 42,
                "resolution": {"grid_resolution": 8},
            },
        )
        assert resp_default.status_code == 200
        assert resp_low.status_code == 200

        cache = get_cache()
        default_entry = cache.get(resp_default.json()["geometry_id"])
        low_entry = cache.get(resp_low.json()["geometry_id"])
        assert default_entry is not None
        assert low_entry is not None

        default_verts = default_entry.math_object.mesh.vertices.shape[0]
        low_verts = low_entry.math_object.mesh.vertices.shape[0]
        assert low_verts < default_verts

    def test_no_resolution_uses_defaults(self, client: TestClient) -> None:
        """POST without resolution field uses generator defaults."""
        resp = client.post(
            "/api/generate",
            json={"generator": "torus", "seed": 42},
        )
        assert resp.status_code == 200
        data = resp.json()
        assert "geometry_id" in data
        assert data["mesh_url"] is not None


# --- HTML resolution UI tests ---


class TestResolutionHTML:
    """Tests for resolution controls in the preview HTML."""

    def test_html_contains_resolution_fields(self, preview_html: str) -> None:
        """Preview HTML contains resolution input fields container."""
        assert 'id="param-resolution-fields"' in preview_html

    def test_html_contains_resolution_section(self, preview_html: str) -> None:
        """Preview HTML contains resolution section with title."""
        assert 'id="param-resolution-section"' in preview_html
        assert ">Resolution<" in preview_html

    def test_info_panel_has_vertex_count(self, preview_html: str) -> None:
        """Info panel has vertex count element."""
        assert 'id="info-vertices"' in preview_html

    def test_info_panel_has_face_count(self, preview_html: str) -> None:
        """Info panel has face count element."""
        assert 'id="info-faces"' in preview_html

    def test_info_panel_has_point_count(self, preview_html: str) -> None:
        """Info panel has point count element."""
        assert 'id="info-points"' in preview_html

    def test_html_populates_resolution_from_api(self, preview_html: str) -> None:
        """JavaScript populates resolution fields from API data."""
        assert "param-resolution-fields" in preview_html
        assert "populateParamFields" in preview_html

    def test_html_sends_resolution_in_generate(self, preview_html: str) -> None:
        """Apply sends resolution in POST body."""
        assert "getEditorResolution" in preview_html
