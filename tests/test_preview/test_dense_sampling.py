"""Tests for the Dense Cloud view mode with post-transform sampling.

Verifies that post-transform sampling produces denser clouds, respects the
sample cap, doesn't break existing modes, and uses separate cache keys.
"""

import re
from typing import Any

import pytest
from fastapi.testclient import TestClient

from mathviz.core.container import Container
from mathviz.core.generator import register
from mathviz.generators.parametric.torus import TorusGenerator
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


def _get_html(client: TestClient) -> str:
    """Fetch the preview HTML page."""
    resp = client.get("/")
    assert resp.status_code == 200
    return resp.text


def _generate(client: TestClient, **overrides: Any) -> dict[str, Any]:
    """Run a generation request and return the JSON response."""
    resp = client.post("/api/generate", json=_torus_payload(**overrides))
    assert resp.status_code == 200
    return resp.json()


class TestPostTransformSamplingDensity:
    """Post-transform sampling produces more points than pre-transform."""

    def test_dense_produces_more_points(self, client: TestClient) -> None:
        """Dense mode cloud has significantly more points than default."""
        # Default sampling (pre-transform)
        default_data = _generate(client)
        default_id = default_data["geometry_id"]
        default_entry = get_cache().get(default_id)
        assert default_entry is not None
        default_cloud = default_entry.math_object.point_cloud

        reset_cache()

        # Post-transform sampling (dense)
        dense_data = _generate(client, sampling="post_transform")
        dense_id = dense_data["geometry_id"]
        dense_entry = get_cache().get(dense_id)
        assert dense_entry is not None
        dense_cloud = dense_entry.math_object.point_cloud

        assert dense_cloud is not None
        # Torus in physical space (~63k mm²) should produce far more
        # points than in abstract space (~24 unit²)
        if default_cloud is not None:
            assert len(dense_cloud.points) > len(default_cloud.points)
        else:
            # Even if default has no cloud, dense must have one
            assert len(dense_cloud.points) > 0


class TestDenseSampleCap:
    """Sample count is capped at the configured maximum."""

    def test_sample_count_capped(self, client: TestClient) -> None:
        """Dense sampling never exceeds MAX_DENSE_SAMPLES."""
        from mathviz.pipeline.dense_sampling import MAX_DENSE_SAMPLES

        data = _generate(client, sampling="post_transform")
        entry = get_cache().get(data["geometry_id"])
        assert entry is not None
        cloud = entry.math_object.point_cloud
        assert cloud is not None
        assert len(cloud.points) <= MAX_DENSE_SAMPLES


class TestExistingViewModesUnaffected:
    """Existing view modes are unaffected by the dense cloud addition."""

    def test_default_sampling_unchanged(self, client: TestClient) -> None:
        """Default request (no sampling field) works as before."""
        data = _generate(client)
        assert "geometry_id" in data
        assert data.get("mesh_url") is not None or data.get("cloud_url") is not None

    def test_shaded_option_still_present(self, client: TestClient) -> None:
        """The shaded option is still in the HTML dropdown."""
        html = _get_html(client)
        assert '<option value="shaded">' in html

    def test_points_still_default(self, client: TestClient) -> None:
        """Points is still the default selected view mode."""
        html = _get_html(client)
        match = re.search(r'<option\s+value="points"[^>]*>', html)
        assert match is not None
        assert "selected" in match.group(0)

    def test_dense_option_present(self, client: TestClient) -> None:
        """The dense cloud option is present in the dropdown."""
        html = _get_html(client)
        assert '<option value="dense">Dense Cloud</option>' in html

    def test_all_original_options_present(self, client: TestClient) -> None:
        """All original view mode options are still present."""
        html = _get_html(client)
        for value in ("shaded", "wireframe", "points", "crystal", "colormap"):
            assert f'value="{value}"' in html


class TestDenseCacheKeySeparation:
    """Cache keys differ between normal and dense sampling."""

    def test_cache_keys_differ(self) -> None:
        """Same params with different sampling produce different cache keys."""
        params: dict[str, Any] = {"major_radius": 1.0, "minor_radius": 0.4}
        container = Container()
        container_kwargs = {
            "width_mm": container.width_mm,
            "height_mm": container.height_mm,
            "depth_mm": container.depth_mm,
            "margin_x_mm": container.margin_x_mm,
            "margin_y_mm": container.margin_y_mm,
            "margin_z_mm": container.margin_z_mm,
        }

        default_key = compute_cache_key(
            "torus", params, 42, {},
            container_kwargs=container_kwargs,
            sampling="default",
        )
        dense_key = compute_cache_key(
            "torus", params, 42, {},
            container_kwargs=container_kwargs,
            sampling="post_transform",
        )
        assert default_key != dense_key

    def test_default_key_backward_compatible(self) -> None:
        """Default sampling key is identical to key without sampling arg."""
        params: dict[str, Any] = {"major_radius": 1.0}
        key_with = compute_cache_key("torus", params, 42, {}, sampling="default")
        key_without = compute_cache_key("torus", params, 42, {})
        assert key_with == key_without

    def test_server_caches_separately(self, client: TestClient) -> None:
        """Server stores dense and default results under different keys."""
        default_data = _generate(client)
        dense_data = _generate(client, sampling="post_transform")

        assert default_data["geometry_id"] != dense_data["geometry_id"]

        # Both should be in cache
        cache = get_cache()
        assert cache.get(default_data["geometry_id"]) is not None
        assert cache.get(dense_data["geometry_id"]) is not None
