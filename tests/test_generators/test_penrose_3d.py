"""Tests for the 3D Penrose tiling generator.

Verifies that PenroseTiling3D produces valid meshes with aperiodic
structure, that higher generations produce more tiles, and that
registration and rendering work correctly.
"""

from typing import Any

import numpy as np
import pytest

from mathviz.core.generator import clear_registry, get_generator, register
from mathviz.core.representation import RepresentationType
from mathviz.generators.procedural.penrose_3d import PenroseTiling3D

_TEST_GENERATIONS = 3


@pytest.fixture(autouse=True)
def _clean_registry():
    """Reset registry and register only this generator."""
    clear_registry(suppress_discovery=True)
    register(PenroseTiling3D)
    yield
    clear_registry(suppress_discovery=True)


@pytest.fixture
def gen() -> PenroseTiling3D:
    """Return a PenroseTiling3D instance."""
    return PenroseTiling3D()


class TestValidMesh:
    """Produces a valid mesh with non-periodic structure."""

    def test_produces_valid_mesh(self, gen: PenroseTiling3D) -> None:
        """Default parameters produce a valid mesh."""
        obj = gen.generate(params={"generations": _TEST_GENERATIONS})
        obj.validate_or_raise()
        assert obj.mesh is not None
        assert len(obj.mesh.vertices) > 0
        assert len(obj.mesh.faces) > 0

    def test_no_nan_in_mesh(self, gen: PenroseTiling3D) -> None:
        """No NaN values in output mesh vertices."""
        obj = gen.generate(params={"generations": _TEST_GENERATIONS})
        assert not np.any(np.isnan(obj.mesh.vertices))

    def test_all_vertices_finite(self, gen: PenroseTiling3D) -> None:
        """All mesh vertices are finite."""
        obj = gen.generate(params={"generations": _TEST_GENERATIONS})
        assert np.all(np.isfinite(obj.mesh.vertices))

    def test_has_two_distinct_heights(self, gen: PenroseTiling3D) -> None:
        """Mesh has two distinct extrusion heights (thick vs thin tiles)."""
        obj = gen.generate(
            params={"generations": _TEST_GENERATIONS, "tile_height_ratio": 0.3},
        )
        z_values = obj.mesh.vertices[:, 2]
        unique_z = np.unique(np.round(z_values, decimals=6))
        # Should have at least 3 unique z: 0.0, thin_height, thick_height
        assert len(unique_z) >= 3

    def test_aperiodic_tile_ratio(self, gen: PenroseTiling3D) -> None:
        """Thick/thin tile ratio approximates golden ratio (aperiodicity)."""
        obj = gen.generate(params={"generations": 5, "extent": 10.0})
        z_top = obj.mesh.vertices[:, 2]
        # Each prism has 6 vertices; top 3 have z > 0
        # Thick tiles have z=1.0, thin have z=0.3
        top_z = z_top[z_top > 0]
        thick_count = np.sum(np.abs(top_z - 1.0) < 0.01)
        thin_count = np.sum(np.abs(top_z - 0.3) < 0.01)
        # Each tile contributes 3 top vertices
        thick_tiles = thick_count / 3
        thin_tiles = thin_count / 3
        assert thick_tiles > 0 and thin_tiles > 0
        ratio = thin_tiles / thick_tiles
        # Ratio should approximate golden ratio (1.618...)
        assert 1.2 < ratio < 2.1, f"Thin/thick ratio {ratio} not near phi"


class TestGenerationsScaling:
    """Higher generations produce more tiles."""

    def test_more_generations_more_tiles(self, gen: PenroseTiling3D) -> None:
        """Increasing generations increases tile count."""
        obj_few = gen.generate(
            params={"generations": 2, "extent": 3.0}, seed=42,
        )
        obj_many = gen.generate(
            params={"generations": 4, "extent": 3.0}, seed=42,
        )
        assert obj_few.mesh is not None and obj_many.mesh is not None
        assert len(obj_many.mesh.faces) > len(obj_few.mesh.faces), (
            f"4 generations ({len(obj_many.mesh.faces)} faces) should "
            f"produce more faces than 2 ({len(obj_few.mesh.faces)} faces)"
        )


class TestRegistration:
    """Generator registers and renders successfully."""

    def test_registered_by_name(self) -> None:
        """Can look up generator by canonical name."""
        cls = get_generator("penrose_3d")
        assert cls is PenroseTiling3D

    def test_registered_by_alias_penrose_tiling(self) -> None:
        """Can look up generator by alias penrose_tiling."""
        cls = get_generator("penrose_tiling")
        assert cls is PenroseTiling3D

    def test_registered_by_alias_penrose(self) -> None:
        """Can look up generator by alias penrose."""
        cls = get_generator("penrose")
        assert cls is PenroseTiling3D

    def test_registers_and_renders(self, gen: PenroseTiling3D) -> None:
        """Generator registers and produces valid renderable output."""
        gen_cls = get_generator("penrose_3d")
        assert gen_cls is PenroseTiling3D

        obj = gen.generate(params={"generations": _TEST_GENERATIONS})
        obj.validate_or_raise()
        assert obj.mesh is not None
        assert len(obj.mesh.vertices) > 0


class TestRepresentation:
    """Default representation is SURFACE_SHELL."""

    def test_default_representation(self, gen: PenroseTiling3D) -> None:
        """Default representation should be SURFACE_SHELL."""
        rep = gen.get_default_representation()
        assert rep.type == RepresentationType.SURFACE_SHELL


class TestMetadata:
    """Generated MathObject has correct metadata."""

    def test_metadata_fields(self, gen: PenroseTiling3D) -> None:
        """MathObject contains generator name, category, and params."""
        obj = gen.generate(
            params={"generations": 3, "tile_height_ratio": 0.5},
        )
        assert obj.generator_name == "penrose_3d"
        assert obj.category == "procedural"
        assert obj.parameters["generations"] == 3
        assert obj.parameters["tile_height_ratio"] == 0.5
        assert obj.bounding_box is not None

    def test_bounding_box_finite(self, gen: PenroseTiling3D) -> None:
        """Bounding box is finite."""
        obj = gen.generate(params={"generations": _TEST_GENERATIONS})
        assert obj.bounding_box is not None
        min_c = np.array(obj.bounding_box.min_corner)
        max_c = np.array(obj.bounding_box.max_corner)
        assert np.all(np.isfinite(min_c))
        assert np.all(np.isfinite(max_c))


class TestDeterminism:
    """Same seed produces identical output."""

    def test_same_seed_same_mesh(self, gen: PenroseTiling3D) -> None:
        """Two runs with identical seed yield the same mesh."""
        kwargs: dict[str, Any] = dict(
            params={"generations": _TEST_GENERATIONS}, seed=42,
        )
        obj1 = gen.generate(**kwargs)
        obj2 = gen.generate(**kwargs)
        np.testing.assert_array_equal(obj1.mesh.vertices, obj2.mesh.vertices)
        np.testing.assert_array_equal(obj1.mesh.faces, obj2.mesh.faces)

    def test_different_seed_different_mesh(
        self, gen: PenroseTiling3D,
    ) -> None:
        """Different seeds produce different meshes (rotation differs)."""
        obj1 = gen.generate(
            params={"generations": _TEST_GENERATIONS}, seed=42,
        )
        obj2 = gen.generate(
            params={"generations": _TEST_GENERATIONS}, seed=99,
        )
        assert not np.array_equal(obj1.mesh.vertices, obj2.mesh.vertices)


class TestValidation:
    """Parameter validation raises on invalid input."""

    def test_zero_generations_raises(self, gen: PenroseTiling3D) -> None:
        """generations=0 raises ValueError."""
        with pytest.raises(ValueError, match="generations must be >="):
            gen.generate(params={"generations": 0})

    def test_too_many_generations_raises(
        self, gen: PenroseTiling3D,
    ) -> None:
        """generations above max raises ValueError."""
        with pytest.raises(ValueError, match="generations must be <="):
            gen.generate(params={"generations": 10})

    def test_negative_tile_height_ratio_raises(
        self, gen: PenroseTiling3D,
    ) -> None:
        """Negative tile_height_ratio raises ValueError."""
        with pytest.raises(ValueError, match="tile_height_ratio must be >="):
            gen.generate(params={"tile_height_ratio": -0.5})

    def test_zero_extent_raises(self, gen: PenroseTiling3D) -> None:
        """Extent below minimum raises ValueError."""
        with pytest.raises(ValueError, match="extent must be >="):
            gen.generate(params={"extent": 0.0})
