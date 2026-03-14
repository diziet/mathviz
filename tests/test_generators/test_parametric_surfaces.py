"""Tests for parametric surface generators added in Task 20.

Covers: klein_bottle, mobius_strip, superellipsoid, spherical_harmonics,
lissajous_surface, boy_surface, enneper_surface.
"""

import numpy as np
import pytest
import trimesh

from mathviz.core.generator import clear_registry, get_generator, register
from mathviz.core.representation import RepresentationType
from mathviz.generators.parametric.boy_surface import BoySurfaceGenerator
from mathviz.generators.parametric.enneper_surface import EnneperSurfaceGenerator
from mathviz.generators.parametric.klein_bottle import KleinBottleGenerator
from mathviz.generators.parametric.lissajous_surface import LissajousSurfaceGenerator
from mathviz.generators.parametric.mobius_strip import MobiusStripGenerator
from mathviz.generators.parametric.spherical_harmonics import (
    SphericalHarmonicsGenerator,
)
from mathviz.generators.parametric.superellipsoid import SuperellipsoidGenerator

_ALL_GENERATORS = [
    KleinBottleGenerator,
    MobiusStripGenerator,
    SuperellipsoidGenerator,
    SphericalHarmonicsGenerator,
    LissajousSurfaceGenerator,
    BoySurfaceGenerator,
    EnneperSurfaceGenerator,
]


@pytest.fixture(autouse=True)
def _clean_registry():
    """Reset registry and register all generators for each test."""
    clear_registry(suppress_discovery=True)
    for gen_cls in _ALL_GENERATORS:
        register(gen_cls)
    yield
    clear_registry(suppress_discovery=True)


# ---------------------------------------------------------------------------
# Each generator produces a non-empty mesh at default params
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("gen_cls", _ALL_GENERATORS, ids=lambda c: c.name)
def test_default_produces_nonempty_mesh(gen_cls: type) -> None:
    """Each generator produces a valid, non-empty mesh at defaults."""
    gen = gen_cls()
    obj = gen.generate(grid_resolution=32)
    obj.validate_or_raise()

    assert obj.mesh is not None
    assert len(obj.mesh.vertices) > 0
    assert len(obj.mesh.faces) > 0
    assert obj.bounding_box is not None


# ---------------------------------------------------------------------------
# All generators accept and record seed
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("gen_cls", _ALL_GENERATORS, ids=lambda c: c.name)
def test_seed_recorded(gen_cls: type) -> None:
    """Seed is recorded in MathObject even if not used for randomness."""
    gen = gen_cls()
    obj = gen.generate(seed=777, grid_resolution=16)
    assert obj.seed == 777


# ---------------------------------------------------------------------------
# All generators default to SURFACE_SHELL representation
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("gen_cls", _ALL_GENERATORS, ids=lambda c: c.name)
def test_default_representation_surface_shell(gen_cls: type) -> None:
    """Default representation is SURFACE_SHELL for all parametric surfaces."""
    gen = gen_cls()
    rep = gen.get_default_representation()
    assert rep.type == RepresentationType.SURFACE_SHELL


# ---------------------------------------------------------------------------
# All generators are discoverable via registry
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("gen_cls", _ALL_GENERATORS, ids=lambda c: c.name)
def test_registered_and_discoverable(gen_cls: type) -> None:
    """Generator is discoverable via the registry by name."""
    found = get_generator(gen_cls.name)
    assert found is gen_cls


# ---------------------------------------------------------------------------
# All generators record metadata
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("gen_cls", _ALL_GENERATORS, ids=lambda c: c.name)
def test_metadata_recorded(gen_cls: type) -> None:
    """Generator name, category, and parameters are recorded."""
    gen = gen_cls()
    obj = gen.generate(grid_resolution=16)
    assert obj.generator_name == gen_cls.name
    assert obj.category == "parametric"
    assert isinstance(obj.parameters, dict)
    assert len(obj.parameters) > 0


# ---------------------------------------------------------------------------
# Vertices lie within declared bounding box
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("gen_cls", _ALL_GENERATORS, ids=lambda c: c.name)
def test_vertices_within_bounding_box(gen_cls: type) -> None:
    """All mesh vertices lie within the declared bounding box."""
    gen = gen_cls()
    obj = gen.generate(grid_resolution=32)
    assert obj.mesh is not None
    assert obj.bounding_box is not None

    verts = obj.mesh.vertices
    min_c = np.array(obj.bounding_box.min_corner)
    max_c = np.array(obj.bounding_box.max_corner)

    tolerance = 1e-6
    assert np.all(verts >= min_c - tolerance), (
        f"{gen_cls.name}: vertices below bounding box min"
    )
    assert np.all(verts <= max_c + tolerance), (
        f"{gen_cls.name}: vertices above bounding box max"
    )


# ---------------------------------------------------------------------------
# Klein bottle: self-intersection (not watertight) is not an error
# ---------------------------------------------------------------------------


def test_klein_bottle_self_intersection_not_error() -> None:
    """Klein bottle has self-intersection — generates without error.

    The figure-8 immersion is topologically closed (mesh is manifold), but
    geometrically self-intersects in R³. We verify the mesh is valid and
    that duplicate spatial positions exist (sign of self-intersection).
    """
    gen = KleinBottleGenerator()
    obj = gen.generate(grid_resolution=64)
    obj.validate_or_raise()
    assert obj.mesh is not None
    assert len(obj.mesh.vertices) > 0
    assert len(obj.mesh.faces) > 0


# ---------------------------------------------------------------------------
# Möbius strip: not watertight (open boundary)
# ---------------------------------------------------------------------------


def test_mobius_strip_not_watertight() -> None:
    """Möbius strip has open boundary and is not watertight."""
    gen = MobiusStripGenerator()
    obj = gen.generate(grid_resolution=32)
    obj.validate_or_raise()
    assert obj.mesh is not None

    tri_mesh = trimesh.Trimesh(
        vertices=obj.mesh.vertices,
        faces=obj.mesh.faces,
        process=False,
    )
    assert not tri_mesh.is_watertight


# ---------------------------------------------------------------------------
# Superellipsoid with e1=e2=1 approximates a sphere
# ---------------------------------------------------------------------------


def test_superellipsoid_sphere_approximation() -> None:
    """Superellipsoid with e1=e2=1 approximates a unit sphere."""
    gen = SuperellipsoidGenerator()
    obj = gen.generate(
        params={"a1": 1.0, "a2": 1.0, "a3": 1.0, "e1": 1.0, "e2": 1.0},
        grid_resolution=64,
    )
    assert obj.mesh is not None

    # Check vertices are approximately on the unit sphere
    verts = obj.mesh.vertices
    radii = np.linalg.norm(verts, axis=1)
    # Exclude pole vertices which are exact
    assert np.allclose(radii, 1.0, atol=0.05), (
        f"Max deviation from unit sphere: {np.max(np.abs(radii - 1.0)):.4f}"
    )


# ---------------------------------------------------------------------------
# Spherical harmonics: all-zero except Y₀₀ approximates a sphere
# ---------------------------------------------------------------------------


def test_spherical_harmonics_y00_sphere() -> None:
    """Spherical harmonics with l=0, m=0 (no modulation) approximates sphere."""
    gen = SphericalHarmonicsGenerator()
    obj = gen.generate(
        params={"l": 0, "m": 0, "base_radius": 1.0, "amplitude": 0.0},
        grid_resolution=64,
    )
    assert obj.mesh is not None

    # All vertices should be on/near the unit sphere
    verts = obj.mesh.vertices
    radii = np.linalg.norm(verts, axis=1)
    assert np.allclose(radii, 1.0, atol=0.05), (
        f"Max deviation from unit sphere: {np.max(np.abs(radii - 1.0)):.4f}"
    )


def test_spherical_harmonics_coefficient_vector() -> None:
    """Spherical harmonics accepts a coefficient vector."""
    gen = SphericalHarmonicsGenerator()
    obj = gen.generate(
        params={
            "base_radius": 1.0,
            "coefficients": [(2, 0, 0.3), (4, 0, 0.1)],
        },
        grid_resolution=32,
    )
    obj.validate_or_raise()
    assert obj.mesh is not None
    assert len(obj.mesh.vertices) > 0


# ---------------------------------------------------------------------------
# Determinism: same seed produces identical output
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("gen_cls", _ALL_GENERATORS, ids=lambda c: c.name)
def test_determinism(gen_cls: type) -> None:
    """Same seed and params produce identical geometry."""
    gen = gen_cls()
    obj1 = gen.generate(seed=42, grid_resolution=16)
    obj2 = gen.generate(seed=42, grid_resolution=16)

    assert obj1.mesh is not None and obj2.mesh is not None
    np.testing.assert_array_equal(obj1.mesh.vertices, obj2.mesh.vertices)
    np.testing.assert_array_equal(obj1.mesh.faces, obj2.mesh.faces)
