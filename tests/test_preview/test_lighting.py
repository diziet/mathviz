"""Tests for improved lighting, shadows, and materials in the preview UI."""

import re

import pytest
from fastapi.testclient import TestClient

from mathviz.preview.server import app, reset_cache, set_served_file


@pytest.fixture(autouse=True)
def _clean_state() -> None:
    """Reset server state between tests."""
    reset_cache()
    set_served_file(None)
    yield
    reset_cache()
    set_served_file(None)


@pytest.fixture
def html() -> str:
    """Fetch the viewer HTML once for all assertion tests."""
    client = TestClient(app)
    resp = client.get("/")
    assert resp.status_code == 200
    return resp.text


class TestLightSources:
    """Verify the scene has at least 3 light sources configured."""

    def test_has_hemisphere_light(self, html: str) -> None:
        """HemisphereLight is present in the scene."""
        assert "HemisphereLight" in html

    def test_has_at_least_three_lights(self, html: str) -> None:
        """At least 3 light sources are added to the scene."""
        light_pattern = re.compile(
            r"new THREE\.(DirectionalLight|AmbientLight|HemisphereLight|PointLight|SpotLight)"
        )
        matches = light_pattern.findall(html)
        assert len(matches) >= 3, f"Expected >=3 lights, found {len(matches)}: {matches}"

    def test_key_light_intensity(self, html: str) -> None:
        """Key directional light has intensity around 1.2."""
        assert "DirectionalLight(0xffffff, 1.2)" in html


class TestShadowMapping:
    """Verify shadow mapping is enabled."""

    def test_shadow_map_enabled(self, html: str) -> None:
        """Shadow mapping is enabled on the renderer."""
        assert "shadowMap.enabled = true" in html

    def test_shadow_map_type(self, html: str) -> None:
        """PCFSoftShadowMap is configured."""
        assert "PCFSoftShadowMap" in html

    def test_light_casts_shadow(self, html: str) -> None:
        """At least one light has castShadow = true."""
        assert "castShadow = true" in html


class TestMeshMaterials:
    """Verify mesh materials support shadows and use improved material."""

    def test_mesh_cast_shadow(self, html: str) -> None:
        """Meshes are configured with castShadow."""
        assert "castShadow = true" in html

    def test_mesh_receive_shadow(self, html: str) -> None:
        """Meshes are configured with receiveShadow."""
        assert "receiveShadow = true" in html

    def test_physical_material_used(self, html: str) -> None:
        """MeshPhysicalMaterial is used for shaded meshes."""
        assert "MeshPhysicalMaterial" in html


class TestNormalComputation:
    """Verify normals are computed on loaded geometries."""

    def test_compute_vertex_normals_called(self, html: str) -> None:
        """computeVertexNormals() is called on loaded geometries."""
        assert "computeVertexNormals()" in html


class TestToneMapping:
    """Verify tone mapping is configured on the renderer."""

    def test_tone_mapping_type(self, html: str) -> None:
        """ACESFilmicToneMapping is configured."""
        assert "ACESFilmicToneMapping" in html

    def test_tone_mapping_exposure(self, html: str) -> None:
        """Tone mapping exposure is set."""
        assert "toneMappingExposure" in html
