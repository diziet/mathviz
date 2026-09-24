"""Tests for crystal preview mode in the Three.js viewer."""

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


class TestCrystalViewModeOption:
    """Crystal option appears in the view mode dropdown."""

    def test_crystal_option_in_dropdown(self, html: str) -> None:
        """View mode dropdown contains a Crystal option."""
        assert 'value="crystal"' in html

    def test_crystal_option_label(self, html: str) -> None:
        """Crystal option has descriptive label text."""
        assert "Crystal Preview" in html


class TestCrystalGlassBlock:
    """Crystal mode creates a MeshPhysicalMaterial glass block."""

    def test_glass_block_uses_physical_material(self, html: str) -> None:
        """Glass block uses MeshPhysicalMaterial with transparent opacity."""
        assert "MeshPhysicalMaterial" in html
        assert re.search(r"transparent:\s*true", html), (
            "Glass block not set to transparent"
        )

    def test_glass_block_opacity(self, html: str) -> None:
        """Glass block has low opacity so points are visible inside."""
        assert re.search(r"opacity:\s*0\.15", html), "Glass opacity not 0.15"

    def test_glass_block_clearcoat(self, html: str) -> None:
        """Glass block has clearcoat for polished surface."""
        assert re.search(r"clearcoat:\s*1\.0", html), "Clearcoat not set"

    def test_glass_block_roughness(self, html: str) -> None:
        """Glass block has low roughness for polished surface."""
        assert re.search(r"roughness:\s*0\.02", html), "Roughness not 0.02"

    def test_glass_block_backside_rendering(self, html: str) -> None:
        """Glass block uses BackSide so front face doesn't occlude points."""
        assert re.search(r"side:\s*THREE\.BackSide", html), "BackSide not set"

    def test_glass_block_env_map(self, html: str) -> None:
        """Preview HTML contains PMREMGenerator and envMap."""
        assert "PMREMGenerator" in html
        assert "envMap" in html


class TestCrystalPointsMaterial:
    """Crystal mode uses PointsMaterial or ShaderMaterial for inner points."""

    def test_crystal_points_material(self, html: str) -> None:
        """Crystal mode creates a PointsMaterial for soft glow points."""
        assert "createCrystalPointsMaterial" in html
        assert re.search(
            r"new THREE\.PointsMaterial\(", html
        ), "PointsMaterial not found in crystal code"

    def test_crystal_point_color(self, html: str) -> None:
        """Crystal points have white/blue tint color."""
        assert re.search(r"color:\s*0xe8f0ff", html), (
            "Crystal point color 0xe8f0ff not found"
        )

    def test_crystal_additive_blending(self, html: str) -> None:
        """Preview HTML contains AdditiveBlending."""
        assert "AdditiveBlending" in html

    def test_crystal_sprite_texture(self, html: str) -> None:
        """Preview HTML contains createRadialGradient and CanvasTexture."""
        assert "createRadialGradient" in html
        assert "CanvasTexture" in html


class TestCrystalBloomPostProcessing:
    """Tests for the bloom post-processing code."""

    def test_effect_composer_imported(self, html: str) -> None:
        """Preview HTML contains EffectComposer."""
        assert "EffectComposer" in html

    def test_render_pass_imported(self, html: str) -> None:
        """Preview HTML contains RenderPass."""
        assert "RenderPass" in html

    def test_unreal_bloom_pass_imported(self, html: str) -> None:
        """Preview HTML contains UnrealBloomPass."""
        assert "UnrealBloomPass" in html

    def test_bloom_setup_in_crystal(self, html: str) -> None:
        """Preview HTML contains setupCrystalComposer and new
        UnrealBloomPass(."""
        assert "setupCrystalComposer" in html
        assert re.search(
            r"new UnrealBloomPass\(", html
        ), "UnrealBloomPass instantiation not found"

    def test_bloom_renders_in_crystal(self, html: str) -> None:
        """Preview HTML calls crystalComposer.render()."""
        assert re.search(
            r"crystalComposer\.render\(\)", html
        ), "Crystal composer render call not found"


class TestCrystalModeExit:
    """Tests for the crystal mode exit and cleanup code."""

    def test_exit_crystal_mode_exists(self, html: str) -> None:
        """exitCrystalMode function is defined."""
        assert "function exitCrystalMode" in html

    def test_exit_removes_glass_block(self, html: str) -> None:
        """Preview HTML calls scene.remove(state.crystalGlassBlock)."""
        assert re.search(
            r"scene\.remove\(state\.crystalGlassBlock\)", html
        ), "Glass block removal not found in exitCrystalMode"

    def test_exit_disposes_composer(self, html: str) -> None:
        """Preview HTML calls crystalComposer.dispose()."""
        assert re.search(
            r"crystalComposer\.dispose\(\)", html
        ), "Composer dispose not found in exitCrystalMode"

    def test_exit_restores_materials(self, html: str) -> None:
        """Preview HTML contains originalMaterial."""
        assert "originalMaterial" in html

    def test_exit_disposes_env_render_target(self, html: str) -> None:
        """Preview HTML calls crystalEnvRT.dispose()."""
        assert "crystalEnvRT" in html
        assert re.search(
            r"crystalEnvRT\.dispose\(\)", html
        ), "Environment map RT disposal not found"

    def test_exit_disposes_canvas_textures(self, html: str) -> None:
        """Preview HTML has crystalTemplateMat.map and .dispose() on one
        line."""
        assert re.search(
            r"crystalTemplateMat\.map.*\.dispose\(\)", html
        ), "Canvas texture disposal not found in exitCrystalMode"

    def test_template_material_disposed_on_exit(self, html: str) -> None:
        """Preview HTML calls crystalTemplateMat.dispose() and stores
        crystalMat in state."""
        assert re.search(
            r"crystalTemplateMat\.dispose\(\)", html
        ), "Template material disposal not found in exitCrystalMode"
        assert re.search(
            r"state\.crystalTemplateMat\s*=\s*crystalMat", html
        ), "Template material not stored in state"


class TestCrystalLedBase:
    """Tests for the LED base checkbox, light and color picker."""

    def test_led_base_checkbox(self, html: str) -> None:
        """Crystal controls include an LED base checkbox."""
        assert 'id="crystal-led-base"' in html

    def test_led_base_creates_light(self, html: str) -> None:
        """LED base adds a PointLight below the glass block."""
        assert "addCrystalLedLight" in html
        assert re.search(
            r"new THREE\.PointLight\(", html
        ), "PointLight for LED base not found"

    def test_led_base_positioned_below(self, html: str) -> None:
        """LED base light is positioned below the block (negative Y)."""
        assert re.search(
            r"light\.position\.set\(0,\s*-", html
        ), "LED light not positioned below block"

    def test_led_base_removable(self, html: str) -> None:
        """Preview HTML contains removeCrystalLedLight."""
        assert "removeCrystalLedLight" in html

    def test_led_color_picker(self, html: str) -> None:
        """Crystal controls include an LED color picker."""
        assert 'id="crystal-led-color"' in html


class TestCrystalDarkBackground:
    """Crystal mode forces dark background."""

    def test_crystal_forces_black_background(self, html: str) -> None:
        """Crystal mode sets background to black (0x000000)."""
        assert re.search(
            r"scene\.background\s*=\s*new THREE\.Color\(0x000000\)", html
        ), "Crystal mode dark background not found"

    def test_exit_restores_bg_from_darkbg_state(self, html: str) -> None:
        """Preview HTML contains state.darkBg ? DARK_COLOR : LIGHT_COLOR."""
        assert re.search(
            r"state\.darkBg\s*\?\s*DARK_COLOR\s*:\s*LIGHT_COLOR", html
        ), "exitCrystalMode should restore bg from state.darkBg"

    def test_bg_toggle_respects_crystal_mode(self, html: str) -> None:
        """Preview HTML contains if (state.crystalActive)."""
        assert re.search(
            r"if\s*\(state\.crystalActive\)", html
        ), "Background toggle crystal check not found"


class TestCrystalCompareModeConflict:
    """Each mode's enter function name is followed later by the other mode's
    exit call."""

    def test_entering_crystal_exits_compare(self, html: str) -> None:
        """Preview HTML has exitCompareMode somewhere after enterCrystalMode."""
        assert re.search(
            r"enterCrystalMode.*exitCompareMode", html, re.DOTALL
        ), "enterCrystalMode should exit compare mode"

    def test_entering_compare_exits_crystal(self, html: str) -> None:
        """Preview HTML has exitCrystalMode somewhere after enterCompareMode."""
        assert re.search(
            r"enterCompareMode.*exitCrystalMode", html, re.DOTALL
        ), "enterCompareMode should exit crystal mode"


class TestCrystalControls:
    """Crystal mode exposes glass tint, bloom, LED, and brightness controls."""

    def test_glass_tint_control(self, html: str) -> None:
        """Glass tint color picker is present."""
        assert 'id="crystal-glass-tint"' in html

    def test_bloom_intensity_slider(self, html: str) -> None:
        """Bloom intensity slider is present."""
        assert 'id="crystal-bloom"' in html

    def test_point_brightness_slider(self, html: str) -> None:
        """Point brightness slider is present."""
        assert 'id="crystal-brightness"' in html

    def test_controls_hidden_by_default(self, html: str) -> None:
        """Crystal controls are hidden when not in crystal mode."""
        assert re.search(
            r'id="crystal-controls"[^>]*style="display:none"', html
        ), "Crystal controls should be hidden by default"
