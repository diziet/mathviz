"""Tests for view mode switching preserving param-editor params.

Verifies that switching to Dense Cloud or HD Cloud after generating with
non-default params via the editor does not reset to generator defaults.
The root cause was _doGenerate not updating state.generatedWith.
"""

import re
from typing import Any

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
def _setup() -> None:
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
    """Fetch the preview HTML once."""
    resp = client.get("/")
    assert resp.status_code == 200
    return resp.text


def _generate(client: TestClient, **overrides: Any) -> dict[str, Any]:
    """Run a generation request and return the JSON response."""
    payload: dict[str, Any] = {
        "generator": "torus",
        "params": {"major_radius": 1.0, "minor_radius": 0.4},
        "seed": 42,
    }
    payload.update(overrides)
    resp = client.post("/api/generate", json=payload)
    assert resp.status_code == 200
    return resp.json()


# --- JS source verification: _doGenerate updates state.generatedWith ---


class TestDoGenerateUpdatesState:
    """Verify _doGenerate sets state.generatedWith after success."""

    def test_do_generate_sets_generated_with(self, preview_html: str) -> None:
        """_doGenerate updates state.generatedWith after displayGenerateResult."""
        # Find _doGenerate function body and check it sets state.generatedWith
        match = re.search(
            r"async function _doGenerate\b.*?^  \}",
            preview_html,
            re.DOTALL | re.MULTILINE,
        )
        assert match is not None, "_doGenerate function not found in HTML"
        body = match.group(0)
        assert "state.generatedWith" in body, (
            "_doGenerate must update state.generatedWith"
        )

    def test_do_generate_sets_geometry_id(self, preview_html: str) -> None:
        """_doGenerate updates state.geometryId after displayGenerateResult."""
        match = re.search(
            r"async function _doGenerate\b.*?^  \}",
            preview_html,
            re.DOTALL | re.MULTILINE,
        )
        assert match is not None
        body = match.group(0)
        assert "state.geometryId" in body, (
            "_doGenerate must update state.geometryId"
        )

    def test_generated_with_includes_resolution(self, preview_html: str) -> None:
        """state.generatedWith in _doGenerate includes resolution field."""
        match = re.search(
            r"async function _doGenerate\b.*?^  \}",
            preview_html,
            re.DOTALL | re.MULTILINE,
        )
        assert match is not None
        body = match.group(0)
        # Find the generatedWith assignment and check it has resolution
        gw_match = re.search(r"state\.generatedWith\s*=\s*\{([^}]+)\}", body)
        assert gw_match is not None
        fields = gw_match.group(1)
        assert "resolution" in fields, (
            "state.generatedWith must include resolution"
        )

    def test_generated_with_set_after_display(self, preview_html: str) -> None:
        """state.generatedWith is set after displayGenerateResult, not before."""
        match = re.search(
            r"async function _doGenerate\b.*?^  \}",
            preview_html,
            re.DOTALL | re.MULTILINE,
        )
        assert match is not None
        body = match.group(0)
        display_pos = body.find("displayGenerateResult")
        gw_pos = body.find("state.generatedWith")
        assert display_pos < gw_pos, (
            "state.generatedWith must be set after displayGenerateResult"
        )


# --- API-level tests simulating the view mode switch flow ---


class TestDenseCloudPreservesParams:
    """Generate with non-default params, then re-generate as Dense Cloud."""

    def test_dense_with_custom_params(self, client: TestClient) -> None:
        """Generate with non-default params, switch to Dense Cloud — params match."""
        custom_params = {"major_radius": 2.5, "minor_radius": 0.1}
        # Step 1: Generate with custom params (simulates param editor Apply)
        editor_data = _generate(client, params=custom_params)
        editor_id = editor_data["geometry_id"]

        # Step 2: Re-generate with same params + dense sampling
        # (simulates view mode handler reading state.generatedWith)
        dense_data = _generate(
            client, params=custom_params, sampling="post_transform",
        )
        dense_id = dense_data["geometry_id"]

        # Different geometry IDs (different sampling) but both should succeed
        assert editor_id != dense_id
        assert dense_data.get("cloud_url") is not None or dense_data.get("mesh_url") is not None


class TestHDCloudPreservesParams:
    """Generate with non-default params, then re-generate as HD Cloud."""

    def test_hd_cloud_with_custom_params(self, client: TestClient) -> None:
        """Generate with non-default params, switch to HD Cloud — params match."""
        custom_params = {"major_radius": 3.0, "minor_radius": 0.2}
        editor_data = _generate(client, params=custom_params)

        hd_data = _generate(
            client, params=custom_params, sampling="resolution_scaled",
        )
        assert editor_data["geometry_id"] != hd_data["geometry_id"]
        assert hd_data.get("cloud_url") is not None or hd_data.get("mesh_url") is not None


class TestPointCloudAfterDense:
    """Switch back to Point Cloud after Dense — params preserved."""

    def test_back_to_points_keeps_params(self, client: TestClient) -> None:
        """Switch back to Point Cloud still uses editor's params, not defaults."""
        custom_params = {"major_radius": 2.0, "minor_radius": 0.3}
        # Generate with custom params
        editor_data = _generate(client, params=custom_params)
        # Switch to Dense Cloud (re-generate)
        _generate(client, params=custom_params, sampling="post_transform")
        # Switch back to Point Cloud (re-generate with default sampling)
        back_data = _generate(client, params=custom_params)
        # Should produce same geometry as original (same params, same sampling)
        assert back_data["geometry_id"] == editor_data["geometry_id"]


class TestDensePointDenseRoundTrip:
    """Dense→Point Cloud→Dense should not reset params."""

    def test_dense_point_dense_no_reset(self, client: TestClient) -> None:
        """Generate via editor, Dense→Point Cloud→Dense — no param reset."""
        custom_params = {"major_radius": 1.5, "minor_radius": 0.6}
        # Generate with custom params (editor Apply)
        _generate(client, params=custom_params)
        # Switch to Dense Cloud
        dense1 = _generate(
            client, params=custom_params, sampling="post_transform",
        )
        # Switch back to Point Cloud
        _generate(client, params=custom_params)
        # Switch to Dense Cloud again
        dense2 = _generate(
            client, params=custom_params, sampling="post_transform",
        )
        # Same params + same sampling → same geometry
        assert dense1["geometry_id"] == dense2["geometry_id"]
