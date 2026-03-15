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
        assert re.search(r"key:\s*1\.2", html), "Key light intensity 1.2 not found"


class TestShadowMapping:
    """Verify shadow mapping is enabled."""

    def test_shadow_map_enabled(self, html: str) -> None:
        """Shadow mapping is enabled on the renderer."""
        assert "shadowMap.enabled = true" in html

    def test_shadow_map_type(self, html: str) -> None:
        """PCFSoftShadowMap is configured."""
        assert "PCFSoftShadowMap" in html

    def test_light_casts_shadow(self, html: str) -> None:
        """At least one light has castShadow = true in createLightingRig."""
        assert re.search(r"key\.castShadow\s*=\s*true", html), (
            "Key light castShadow not found in lighting rig"
        )


class TestMeshMaterials:
    """Verify mesh materials support shadows and use improved material."""

    def test_mesh_cast_and_receive_shadow(self, html: str) -> None:
        """Meshes are configured with both castShadow and receiveShadow."""
        assert re.search(r"shadedMesh\.castShadow\s*=\s*true", html), (
            "Mesh castShadow not found"
        )
        assert re.search(r"shadedMesh\.receiveShadow\s*=\s*true", html), (
            "Mesh receiveShadow not found"
        )

    def test_physical_material_used(self, html: str) -> None:
        """MeshPhysicalMaterial is used for shaded meshes."""
        assert "MeshPhysicalMaterial" in html

    def test_shadow_on_both_light_and_mesh(self, html: str) -> None:
        """castShadow appears on both lights and meshes."""
        count = html.count("castShadow = true")
        assert count >= 2, f"Expected castShadow on light + mesh, found {count} occurrences"


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


class TestLightingRigShared:
    """Verify createLightingRig is used to avoid duplication."""

    def test_create_lighting_rig_exists(self, html: str) -> None:
        """A shared createLightingRig function exists."""
        assert "function createLightingRig" in html

    def test_apply_light_intensities_exists(self, html: str) -> None:
        """A shared applyLightIntensities function exists."""
        assert "function applyLightIntensities" in html

    def test_light_intensities_constant(self, html: str) -> None:
        """LIGHT_INTENSITIES constant defines dark and light mode values."""
        assert "LIGHT_INTENSITIES" in html
        assert re.search(r"dark:\s*\{", html), "Dark mode intensities not found"
        assert re.search(r"light:\s*\{", html), "Light mode intensities not found"
