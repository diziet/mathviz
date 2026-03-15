"""Tests for hydrogen atom electron orbital generator.

Verifies that ElectronOrbitalGenerator produces correct orbital shapes
for various quantum numbers, validates input, and integrates with
the registry and representation pipeline.
"""

import numpy as np
import pytest

from mathviz.core.generator import clear_registry, get_generator, register
from mathviz.core.representation import RepresentationType
from mathviz.generators.physics.electron_orbital import ElectronOrbitalGenerator

# Low resolution for fast tests
_TEST_VOXEL_RES = 32


@pytest.fixture(autouse=True)
def _clean_registry():
    """Reset registry and register only this generator."""
    clear_registry(suppress_discovery=True)
    register(ElectronOrbitalGenerator)
    yield
    clear_registry(suppress_discovery=True)


@pytest.fixture
def gen() -> ElectronOrbitalGenerator:
    """Return an ElectronOrbitalGenerator instance."""
    return ElectronOrbitalGenerator()


class TestSphereOrbital:
    """(1,0,0) s-orbital produces a roughly spherical isosurface."""

    def test_s_orbital_is_roughly_spherical(
        self, gen: ElectronOrbitalGenerator
    ) -> None:
        """The 1s orbital should be close to a sphere."""
        obj = gen.generate(
            params={"n": 1, "l": 0, "m": 0},
            voxel_resolution=_TEST_VOXEL_RES,
        )
        obj.validate_or_raise()
        assert obj.mesh is not None
        assert len(obj.mesh.vertices) > 0

        # Check sphericity: std/mean of radial distances should be small
        dists = np.linalg.norm(obj.mesh.vertices, axis=1)
        relative_spread = dists.std() / dists.mean()
        assert relative_spread < 0.05, (
            f"1s orbital not spherical enough: std/mean={relative_spread:.3f}"
        )

    def test_s_orbital_centered_at_origin(
        self, gen: ElectronOrbitalGenerator
    ) -> None:
        """The 1s orbital centroid should be near the origin."""
        obj = gen.generate(
            params={"n": 1, "l": 0, "m": 0},
            voxel_resolution=_TEST_VOXEL_RES,
        )
        centroid = obj.mesh.vertices.mean(axis=0)
        assert np.allclose(centroid, 0, atol=0.5), (
            f"1s orbital centroid {centroid} not near origin"
        )


class TestDumbbellOrbital:
    """(2,1,0) p-orbital produces a dumbbell shape with two lobes."""

    def test_p_orbital_has_two_lobes(
        self, gen: ElectronOrbitalGenerator
    ) -> None:
        """The 2p0 orbital should have vertices above and below z=0."""
        obj = gen.generate(
            params={"n": 2, "l": 1, "m": 0},
            voxel_resolution=_TEST_VOXEL_RES,
        )
        obj.validate_or_raise()
        assert obj.mesh is not None

        verts = obj.mesh.vertices
        above = verts[verts[:, 2] > 0.5]
        below = verts[verts[:, 2] < -0.5]
        assert len(above) > 10, "Expected vertices in upper lobe"
        assert len(below) > 10, "Expected vertices in lower lobe"

    def test_p_orbital_is_not_spherical(
        self, gen: ElectronOrbitalGenerator
    ) -> None:
        """The 2p0 orbital should be elongated along z, not spherical."""
        obj = gen.generate(
            params={"n": 2, "l": 1, "m": 0},
            voxel_resolution=_TEST_VOXEL_RES,
        )
        verts = obj.mesh.vertices
        z_range = verts[:, 2].max() - verts[:, 2].min()
        x_range = verts[:, 0].max() - verts[:, 0].min()
        assert z_range > x_range, (
            f"p orbital should be elongated along z: z_range={z_range}, "
            f"x_range={x_range}"
        )


class TestInvalidQuantumNumbers:
    """Invalid quantum numbers raise ValueError."""

    def test_l_ge_n_raises(self, gen: ElectronOrbitalGenerator) -> None:
        """l >= n is invalid and must raise ValueError."""
        with pytest.raises(ValueError, match="l must be < n"):
            gen.generate(
                params={"n": 1, "l": 1, "m": 0},
                voxel_resolution=_TEST_VOXEL_RES,
            )

    def test_l_equals_n_raises(self, gen: ElectronOrbitalGenerator) -> None:
        """l == n is invalid."""
        with pytest.raises(ValueError, match="l must be < n"):
            gen.generate(
                params={"n": 2, "l": 2, "m": 0},
                voxel_resolution=_TEST_VOXEL_RES,
            )

    def test_abs_m_gt_l_raises(self, gen: ElectronOrbitalGenerator) -> None:
        """|m| > l is invalid."""
        with pytest.raises(ValueError, match=r"\|m\| must be <= l"):
            gen.generate(
                params={"n": 3, "l": 1, "m": 2},
                voxel_resolution=_TEST_VOXEL_RES,
            )

    def test_negative_n_raises(self, gen: ElectronOrbitalGenerator) -> None:
        """n < 1 is invalid."""
        with pytest.raises(ValueError, match="n must be >= 1"):
            gen.generate(
                params={"n": 0, "l": 0, "m": 0},
                voxel_resolution=_TEST_VOXEL_RES,
            )


class TestRegistration:
    """Generator registers and is discoverable."""

    def test_registered_by_name(self) -> None:
        """Can look up generator by canonical name."""
        cls = get_generator("electron_orbital")
        assert cls is ElectronOrbitalGenerator

    def test_registered_by_alias(self) -> None:
        """Can look up generator by alias."""
        cls = get_generator("hydrogen_orbital")
        assert cls is ElectronOrbitalGenerator


class TestRepresentation:
    """Default representation is SURFACE_SHELL."""

    def test_default_representation(
        self, gen: ElectronOrbitalGenerator
    ) -> None:
        """Default representation should be SURFACE_SHELL."""
        rep = gen.get_default_representation()
        assert rep.type == RepresentationType.SURFACE_SHELL


class TestMetadata:
    """Generated MathObject has correct metadata."""

    def test_metadata_fields(self, gen: ElectronOrbitalGenerator) -> None:
        """MathObject contains generator name, category, and params."""
        obj = gen.generate(
            params={"n": 2, "l": 1, "m": 0},
            voxel_resolution=_TEST_VOXEL_RES,
        )
        assert obj.generator_name == "electron_orbital"
        assert obj.category == "physics"
        assert obj.parameters["n"] == 2
        assert obj.parameters["l"] == 1
        assert obj.parameters["m"] == 0
        assert obj.bounding_box is not None


class TestDeterminism:
    """Same parameters produce identical output."""

    def test_same_params_same_mesh(
        self, gen: ElectronOrbitalGenerator
    ) -> None:
        """Two runs with identical params yield the same mesh."""
        kwargs = dict(
            params={"n": 2, "l": 1, "m": 0},
            voxel_resolution=_TEST_VOXEL_RES,
        )
        obj1 = gen.generate(**kwargs)
        obj2 = gen.generate(**kwargs)
        np.testing.assert_array_equal(obj1.mesh.vertices, obj2.mesh.vertices)
        np.testing.assert_array_equal(obj1.mesh.faces, obj2.mesh.faces)


class TestCloverleafOrbital:
    """(3,2,0) d-orbital produces a cloverleaf shape."""

    def test_d_orbital_has_mesh(self, gen: ElectronOrbitalGenerator) -> None:
        """The 3d0 orbital generates a valid mesh."""
        obj = gen.generate(
            params={"n": 3, "l": 2, "m": 0},
            voxel_resolution=_TEST_VOXEL_RES,
        )
        obj.validate_or_raise()
        assert obj.mesh is not None
        assert len(obj.mesh.vertices) > 100
        assert len(obj.mesh.faces) > 100
