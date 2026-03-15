"""Tests for the wave interference pattern generator.

Verifies that WaveInterferenceGenerator produces valid meshes with
wave-like structure, that more sources increase complexity, and that
registration and rendering work correctly.
"""

import numpy as np
import pytest

from mathviz.core.generator import clear_registry, get_generator, register
from mathviz.core.representation import RepresentationType
from mathviz.generators.physics.wave_interference import (
    WaveInterferenceGenerator,
)

_TEST_VOXEL_RES = 32


@pytest.fixture(autouse=True)
def _clean_registry():
    """Reset registry and register only this generator."""
    clear_registry(suppress_discovery=True)
    register(WaveInterferenceGenerator)
    yield
    clear_registry(suppress_discovery=True)


@pytest.fixture
def gen() -> WaveInterferenceGenerator:
    """Return a WaveInterferenceGenerator instance."""
    return WaveInterferenceGenerator()


class TestValidMesh:
    """Produces a valid mesh with wave-like structure."""

    def test_produces_valid_mesh(
        self, gen: WaveInterferenceGenerator
    ) -> None:
        """Default parameters produce a valid mesh."""
        obj = gen.generate(voxel_resolution=_TEST_VOXEL_RES)
        obj.validate_or_raise()
        assert obj.mesh is not None
        assert len(obj.mesh.vertices) > 0
        assert len(obj.mesh.faces) > 0

    def test_no_nan_in_mesh(self, gen: WaveInterferenceGenerator) -> None:
        """No NaN values in output mesh vertices."""
        obj = gen.generate(voxel_resolution=_TEST_VOXEL_RES)
        assert not np.any(np.isnan(obj.mesh.vertices))
        assert not np.any(np.isnan(obj.mesh.normals))

    def test_all_vertices_finite(
        self, gen: WaveInterferenceGenerator
    ) -> None:
        """All mesh vertices are finite."""
        obj = gen.generate(voxel_resolution=_TEST_VOXEL_RES)
        assert np.all(np.isfinite(obj.mesh.vertices))

    def test_single_source_produces_mesh(
        self, gen: WaveInterferenceGenerator
    ) -> None:
        """A single source produces a valid mesh."""
        obj = gen.generate(
            params={"num_sources": 1},
            voxel_resolution=_TEST_VOXEL_RES,
        )
        obj.validate_or_raise()
        assert obj.mesh is not None
        assert len(obj.mesh.vertices) > 0


class TestComplexity:
    """More sources produce more complex patterns."""

    def test_more_sources_more_faces(
        self, gen: WaveInterferenceGenerator
    ) -> None:
        """Increasing source count increases mesh complexity."""
        obj_few = gen.generate(
            params={"num_sources": 2},
            voxel_resolution=_TEST_VOXEL_RES,
        )
        obj_many = gen.generate(
            params={"num_sources": 5},
            voxel_resolution=_TEST_VOXEL_RES,
        )
        assert obj_few.mesh is not None and obj_many.mesh is not None
        # More sources create more interference fringes → more faces
        assert len(obj_many.mesh.faces) > len(obj_few.mesh.faces), (
            f"5 sources ({len(obj_many.mesh.faces)} faces) should produce "
            f"more faces than 2 sources ({len(obj_few.mesh.faces)} faces)"
        )


class TestRegistration:
    """Generator registers and renders successfully."""

    def test_registered_by_name(self) -> None:
        """Can look up generator by canonical name."""
        cls = get_generator("wave_interference")
        assert cls is WaveInterferenceGenerator

    def test_registered_by_alias_wave_pattern(self) -> None:
        """Can look up generator by alias wave_pattern."""
        cls = get_generator("wave_pattern")
        assert cls is WaveInterferenceGenerator

    def test_registered_by_alias_interference(self) -> None:
        """Can look up generator by alias interference."""
        cls = get_generator("interference")
        assert cls is WaveInterferenceGenerator

    def test_registers_and_renders(
        self, gen: WaveInterferenceGenerator
    ) -> None:
        """Generator registers and produces valid renderable output."""
        gen_cls = get_generator("wave_interference")
        assert gen_cls is WaveInterferenceGenerator

        obj = gen.generate(voxel_resolution=_TEST_VOXEL_RES)
        obj.validate_or_raise()
        assert obj.mesh is not None
        assert len(obj.mesh.vertices) > 0


class TestRepresentation:
    """Default representation is SURFACE_SHELL."""

    def test_default_representation(
        self, gen: WaveInterferenceGenerator
    ) -> None:
        """Default representation should be SURFACE_SHELL."""
        rep = gen.get_default_representation()
        assert rep.type == RepresentationType.SURFACE_SHELL


class TestMetadata:
    """Generated MathObject has correct metadata."""

    def test_metadata_fields(self, gen: WaveInterferenceGenerator) -> None:
        """MathObject contains generator name, category, and params."""
        obj = gen.generate(
            params={"num_sources": 4},
            voxel_resolution=_TEST_VOXEL_RES,
        )
        assert obj.generator_name == "wave_interference"
        assert obj.category == "physics"
        assert obj.parameters["num_sources"] == 4
        assert obj.bounding_box is not None

    def test_bounding_box_finite(
        self, gen: WaveInterferenceGenerator
    ) -> None:
        """Bounding box is finite."""
        obj = gen.generate(voxel_resolution=_TEST_VOXEL_RES)
        assert obj.bounding_box is not None
        min_c = np.array(obj.bounding_box.min_corner)
        max_c = np.array(obj.bounding_box.max_corner)
        assert np.all(np.isfinite(min_c))
        assert np.all(np.isfinite(max_c))


class TestDeterminism:
    """Same seed produces identical output."""

    def test_same_seed_same_mesh(
        self, gen: WaveInterferenceGenerator
    ) -> None:
        """Two runs with identical seed yield the same mesh."""
        kwargs = dict(
            params={"num_sources": 3},
            seed=42,
            voxel_resolution=_TEST_VOXEL_RES,
        )
        obj1 = gen.generate(**kwargs)
        obj2 = gen.generate(**kwargs)
        np.testing.assert_array_equal(obj1.mesh.vertices, obj2.mesh.vertices)
        np.testing.assert_array_equal(obj1.mesh.faces, obj2.mesh.faces)

    def test_different_seed_different_mesh(
        self, gen: WaveInterferenceGenerator
    ) -> None:
        """Different seeds produce different meshes (source jitter differs)."""
        obj1 = gen.generate(
            params={"num_sources": 3}, seed=42,
            voxel_resolution=_TEST_VOXEL_RES,
        )
        obj2 = gen.generate(
            params={"num_sources": 3}, seed=99,
            voxel_resolution=_TEST_VOXEL_RES,
        )
        assert not np.array_equal(obj1.mesh.vertices, obj2.mesh.vertices)


class TestValidation:
    """Parameter validation raises on invalid input."""

    def test_zero_sources_raises(
        self, gen: WaveInterferenceGenerator
    ) -> None:
        """num_sources=0 raises ValueError."""
        with pytest.raises(ValueError, match="num_sources must be >="):
            gen.generate(params={"num_sources": 0})

    def test_negative_wavelength_raises(
        self, gen: WaveInterferenceGenerator
    ) -> None:
        """Negative wavelength raises ValueError."""
        with pytest.raises(ValueError, match="wavelength must be >="):
            gen.generate(params={"wavelength": -1.0})

    def test_zero_spacing_raises(
        self, gen: WaveInterferenceGenerator
    ) -> None:
        """Zero source_spacing raises ValueError."""
        with pytest.raises(ValueError, match="source_spacing must be > 0"):
            gen.generate(params={"source_spacing": 0.0})

    def test_negative_iso_level_raises(
        self, gen: WaveInterferenceGenerator
    ) -> None:
        """Negative iso_level raises ValueError."""
        with pytest.raises(ValueError, match="iso_level must be > 0"):
            gen.generate(params={"iso_level": -0.1})

    def test_too_many_sources_raises(
        self, gen: WaveInterferenceGenerator
    ) -> None:
        """num_sources above maximum raises ValueError."""
        with pytest.raises(ValueError, match="num_sources must be <="):
            gen.generate(params={"num_sources": 100})

    def test_iso_level_ge_one_raises(
        self, gen: WaveInterferenceGenerator
    ) -> None:
        """iso_level >= 1.0 raises ValueError (field normalized to [-1,1])."""
        with pytest.raises(ValueError, match="iso_level must be < 1.0"):
            gen.generate(params={"iso_level": 1.0})
        with pytest.raises(ValueError, match="iso_level must be < 1.0"):
            gen.generate(params={"iso_level": 2.0})
