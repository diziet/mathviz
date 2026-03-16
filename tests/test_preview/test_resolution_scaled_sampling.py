"""Tests for the HD Cloud (resolution-scaled) view mode.

Verifies that resolution-scaled sampling scales density with resolution,
matches normal density at default resolution, respects the sample cap,
and doesn't break existing view modes.
"""

from typing import Any

import numpy as np
import pytest
from fastapi.testclient import TestClient

from mathviz.core.generator import register
from mathviz.core.math_object import MathObject, Mesh
from mathviz.generators.parametric.torus import TorusGenerator
from mathviz.pipeline.dense_sampling import (
    MAX_RESOLUTION_SCALED_SAMPLES,
    _DENSE_SURFACE_DENSITY,
    _compute_resolution_scale,
    apply_resolution_scaled_sampling,
)
from mathviz.preview.cache import compute_cache_key
from mathviz.preview.server import app, get_cache, reset_cache


def _ensure_torus_registered() -> None:
    """Re-register the torus generator if missing from the registry."""
    import mathviz.core.generator as gen_mod

    if "torus" not in gen_mod._alias_map:
        gen_mod._discovered = True
        register(TorusGenerator)


@pytest.fixture(autouse=True)
def _ensure_generators() -> None:
    """Ensure real generators are registered and cache is clean."""
    _ensure_torus_registered()
    reset_cache()
    yield
    reset_cache()


@pytest.fixture
def client() -> TestClient:
    """Return a FastAPI test client."""
    return TestClient(app)


def _torus_payload(**overrides: Any) -> dict[str, Any]:
    """Build a standard torus generation request."""
    payload: dict[str, Any] = {
        "generator": "torus",
        "params": {"major_radius": 1.0, "minor_radius": 0.4},
        "seed": 42,
    }
    payload.update(overrides)
    return payload


def _generate(client: TestClient, **overrides: Any) -> dict[str, Any]:
    """Run a generation request and return the JSON response."""
    resp = client.post("/api/generate", json=_torus_payload(**overrides))
    assert resp.status_code == 200
    return resp.json()


def _small_mesh() -> Mesh:
    """Create a small planar mesh (2 mm² area) for controlled tests."""
    vertices = np.array([
        [0, 0, 0], [1, 0, 0], [0, 1, 0], [1, 1, 0],
    ], dtype=np.float64)
    faces = np.array([[0, 1, 2], [1, 3, 2]], dtype=np.int64)
    return Mesh(vertices=vertices, faces=faces)


def _sample_small_mesh(voxel_resolution: int = 128, **kwargs: Any) -> MathObject:
    """Apply resolution-scaled sampling to a small mesh at given resolution."""
    obj = MathObject(generator_name="test", mesh=_small_mesh())
    return apply_resolution_scaled_sampling(
        obj,
        resolution_kwargs={"voxel_resolution": voxel_resolution},
        default_resolution={"voxel_resolution": 128},
        max_samples=500_000,
        **kwargs,
    )


class TestResolutionScaling:
    """Doubling voxel resolution roughly quadruples point count."""

    def test_double_resolution_quadruples_points(self) -> None:
        """At 2x resolution, point count should be roughly 4x."""
        count_default = len(_sample_small_mesh(128).point_cloud.points)
        count_double = len(_sample_small_mesh(256).point_cloud.points)

        ratio = count_double / count_default
        assert ratio > 3.0, f"Expected ~4x, got {ratio:.1f}x"
        assert ratio < 5.0, f"Expected ~4x, got {ratio:.1f}x"

    def test_quadruple_resolution_16x_points(self) -> None:
        """At 4x resolution, point count should be roughly 16x."""
        count_default = len(_sample_small_mesh(128).point_cloud.points)
        count_quad = len(_sample_small_mesh(512).point_cloud.points)

        ratio = count_quad / count_default
        assert ratio > 14.0, f"Expected ~16x, got {ratio:.1f}x"
        assert ratio < 18.0, f"Expected ~16x, got {ratio:.1f}x"


class TestDefaultResolutionMatchesNormal:
    """Default resolution produces the same count as normal SPARSE_SHELL."""

    def test_default_resolution_same_density(self) -> None:
        """At default resolution, scale is 1.0 so density equals base."""
        obj = MathObject(generator_name="test", mesh=_small_mesh())

        result_scaled = _sample_small_mesh(128)

        from mathviz.pipeline.dense_sampling import apply_post_transform_sampling

        result_dense = apply_post_transform_sampling(obj)

        assert len(result_scaled.point_cloud.points) == len(
            result_dense.point_cloud.points
        )

    def test_no_resolution_kwarg_uses_scale_1(self) -> None:
        """When resolution_kwargs is empty, scale defaults to 1.0."""
        obj = MathObject(generator_name="test", mesh=_small_mesh())
        result = apply_resolution_scaled_sampling(
            obj,
            resolution_kwargs={},
            default_resolution={"voxel_resolution": 128},
            max_samples=500_000,
        )
        assert result.point_cloud is not None
        assert len(result.point_cloud.points) > 0


class TestSampleCountCap:
    """Sample count is capped at the configured maximum."""

    def test_cap_value(self) -> None:
        """The cap is configured at 500,000."""
        assert MAX_RESOLUTION_SCALED_SAMPLES == 500_000

    def test_cap_applied_in_function(self) -> None:
        """The sampling function respects a custom max_samples cap."""
        vertices = np.array([
            [0, 0, 0], [1000, 0, 0], [0, 1000, 0], [1000, 1000, 0],
        ], dtype=np.float64)
        faces = np.array([[0, 1, 2], [1, 3, 2]], dtype=np.int64)
        mesh = Mesh(vertices=vertices, faces=faces)
        obj = MathObject(generator_name="test", mesh=mesh)

        result = apply_resolution_scaled_sampling(
            obj,
            resolution_kwargs={"voxel_resolution": 512},
            default_resolution={"voxel_resolution": 128},
            max_samples=100,
        )
        assert result.point_cloud is not None
        assert len(result.point_cloud.points) <= 100

    def test_server_respects_cap(self, client: TestClient) -> None:
        """Server-side HD Cloud sampling doesn't exceed the cap."""
        data = _generate(
            client,
            sampling="resolution_scaled",
            resolution={"grid_resolution": 1024},
        )
        entry = get_cache().get(data["geometry_id"])
        assert entry is not None
        cloud = entry.math_object.point_cloud
        assert cloud is not None
        assert len(cloud.points) <= MAX_RESOLUTION_SCALED_SAMPLES


class TestExistingModesUnaffected:
    """Existing view modes are unaffected by the HD Cloud addition."""

    def test_default_sampling_unchanged(self, client: TestClient) -> None:
        """Default request (no sampling field) works as before."""
        data = _generate(client)
        assert "geometry_id" in data

    def test_dense_sampling_unchanged(self, client: TestClient) -> None:
        """Dense mode still works as before."""
        data = _generate(client, sampling="post_transform")
        assert "geometry_id" in data

    def test_hd_cloud_option_present_in_html(
        self, client: TestClient
    ) -> None:
        """The HD Cloud option is present in the dropdown."""
        resp = client.get("/")
        assert resp.status_code == 200
        assert '<option value="hd_cloud">HD Cloud</option>' in resp.text

    def test_all_original_options_present(
        self, client: TestClient
    ) -> None:
        """All original view mode options are still present."""
        resp = client.get("/")
        html = resp.text
        for value in ("shaded", "wireframe", "points", "dense", "crystal", "colormap"):
            assert f'value="{value}"' in html


class TestResolutionScaledCacheKey:
    """Cache keys differ between resolution_scaled and other modes."""

    def test_cache_key_differs_from_default(self) -> None:
        """resolution_scaled and default produce different keys."""
        params: dict[str, Any] = {"major_radius": 1.0}
        key_default = compute_cache_key("torus", params, 42, {})
        key_hd = compute_cache_key(
            "torus", params, 42, {}, sampling="resolution_scaled",
        )
        assert key_default != key_hd

    def test_cache_key_differs_from_dense(self) -> None:
        """resolution_scaled and post_transform produce different keys."""
        params: dict[str, Any] = {"major_radius": 1.0}
        key_dense = compute_cache_key(
            "torus", params, 42, {}, sampling="post_transform",
        )
        key_hd = compute_cache_key(
            "torus", params, 42, {}, sampling="resolution_scaled",
        )
        assert key_dense != key_hd

    def test_server_caches_separately(self, client: TestClient) -> None:
        """Server stores default, dense, and hd_cloud under different keys."""
        default_data = _generate(client)
        dense_data = _generate(client, sampling="post_transform")
        hd_data = _generate(client, sampling="resolution_scaled")

        ids = {
            default_data["geometry_id"],
            dense_data["geometry_id"],
            hd_data["geometry_id"],
        }
        assert len(ids) == 3


class TestComputeResolutionScale:
    """Unit tests for _compute_resolution_scale helper."""

    def test_same_resolution_returns_1(self) -> None:
        """Same resolution as default produces scale 1.0."""
        scale = _compute_resolution_scale(
            {"voxel_resolution": 128}, {"voxel_resolution": 128},
        )
        assert scale == pytest.approx(1.0)

    def test_double_returns_4(self) -> None:
        """Double resolution produces scale 4.0."""
        scale = _compute_resolution_scale(
            {"voxel_resolution": 256}, {"voxel_resolution": 128},
        )
        assert scale == pytest.approx(4.0)

    def test_half_returns_quarter(self) -> None:
        """Half resolution produces scale 0.25."""
        scale = _compute_resolution_scale(
            {"voxel_resolution": 64}, {"voxel_resolution": 128},
        )
        assert scale == pytest.approx(0.25)

    def test_missing_key_returns_1(self) -> None:
        """Missing resolution key falls back to 1.0."""
        scale = _compute_resolution_scale({}, {"voxel_resolution": 128})
        assert scale == pytest.approx(1.0)

    def test_grid_resolution_works(self) -> None:
        """Works with grid_resolution parameter name."""
        scale = _compute_resolution_scale(
            {"grid_resolution": 256}, {"grid_resolution": 128},
        )
        assert scale == pytest.approx(4.0)
