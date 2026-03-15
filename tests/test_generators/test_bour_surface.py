"""Tests for Bour's minimal surface generator.

Covers: mesh validity, different n values produce distinct surfaces,
registration, representation, determinism, parameter validation,
and bounding box.
"""

import numpy as np
import pytest

from mathviz.core.generator import clear_registry, get_generator, register
from mathviz.core.representation import RepresentationType
from mathviz.generators.parametric.bour_surface import BourSurfaceGenerator


@pytest.fixture(autouse=True)
def _clean_registry():
    """Reset registry and register the Bour surface generator."""
    clear_registry(suppress_discovery=True)
    register(BourSurfaceGenerator)
    yield
    clear_registry(suppress_discovery=True)


# --- Mesh validity ---


def test_default_produces_nonempty_mesh() -> None:
    """Generator produces a valid, non-empty mesh at defaults."""
    gen = BourSurfaceGenerator()
    obj = gen.generate(grid_resolution=32)
    obj.validate_or_raise()

    assert obj.mesh is not None
    assert len(obj.mesh.vertices) > 0
    assert len(obj.mesh.faces) > 0
    assert obj.bounding_box is not None


def test_no_nan_or_inf_in_vertices() -> None:
    """Mesh vertices contain no NaN or Inf values."""
    gen = BourSurfaceGenerator()
    obj = gen.generate(grid_resolution=32)
    assert obj.mesh is not None
    assert np.all(np.isfinite(obj.mesh.vertices))


# --- Different n values produce distinct surfaces ---


def test_different_n_values_produce_distinct_surfaces() -> None:
    """Different n values produce geometrically distinct surfaces."""
    gen = BourSurfaceGenerator()
    obj_n2 = gen.generate(params={"n": 2}, grid_resolution=32)
    obj_n3 = gen.generate(params={"n": 3}, grid_resolution=32)
    obj_n5 = gen.generate(params={"n": 5}, grid_resolution=32)

    assert obj_n2.mesh is not None
    assert obj_n3.mesh is not None
    assert obj_n5.mesh is not None

    # Vertices should differ between different n values
    assert not np.allclose(obj_n2.mesh.vertices, obj_n3.mesh.vertices)
    assert not np.allclose(obj_n3.mesh.vertices, obj_n5.mesh.vertices)


def test_n_affects_z_range() -> None:
    """Different n values produce different z extents."""
    gen = BourSurfaceGenerator()
    obj_n2 = gen.generate(params={"n": 2}, grid_resolution=32)
    obj_n5 = gen.generate(params={"n": 5}, grid_resolution=32)

    assert obj_n2.mesh is not None and obj_n5.mesh is not None
    z_range_n2 = np.ptp(obj_n2.mesh.vertices[:, 2])
    z_range_n5 = np.ptp(obj_n5.mesh.vertices[:, 2])
    assert z_range_n2 != pytest.approx(z_range_n5, rel=0.01)


# --- Registration ---


def test_registered_and_discoverable() -> None:
    """Generator is discoverable via the registry by name."""
    found = get_generator("bour_surface")
    assert found is BourSurfaceGenerator


def test_alias_discoverable() -> None:
    """Generator is discoverable via the bour alias."""
    found = get_generator("bour")
    assert found is BourSurfaceGenerator


# --- Representation ---


def test_default_representation_surface_shell() -> None:
    """Default representation is SURFACE_SHELL."""
    gen = BourSurfaceGenerator()
    rep = gen.get_default_representation()
    assert rep.type == RepresentationType.SURFACE_SHELL


# --- Metadata ---


def test_metadata_recorded() -> None:
    """Generator name, category, and parameters are recorded."""
    gen = BourSurfaceGenerator()
    obj = gen.generate(grid_resolution=16)
    assert obj.generator_name == "bour_surface"
    assert obj.category == "parametric"
    assert isinstance(obj.parameters, dict)
    assert "n" in obj.parameters
    assert "r_max" in obj.parameters


def test_seed_recorded() -> None:
    """Seed is recorded in MathObject."""
    gen = BourSurfaceGenerator()
    obj = gen.generate(seed=777, grid_resolution=16)
    assert obj.seed == 777


# --- Determinism ---


def test_determinism() -> None:
    """Same seed and params produce identical geometry."""
    gen = BourSurfaceGenerator()
    obj1 = gen.generate(seed=42, grid_resolution=16)
    obj2 = gen.generate(seed=42, grid_resolution=16)

    assert obj1.mesh is not None and obj2.mesh is not None
    np.testing.assert_array_equal(obj1.mesh.vertices, obj2.mesh.vertices)
    np.testing.assert_array_equal(obj1.mesh.faces, obj2.mesh.faces)


# --- Bounding box ---


def test_vertices_within_bounding_box() -> None:
    """All mesh vertices lie within the declared bounding box."""
    gen = BourSurfaceGenerator()
    obj = gen.generate(grid_resolution=32)
    assert obj.mesh is not None
    assert obj.bounding_box is not None

    verts = obj.mesh.vertices
    min_c = np.array(obj.bounding_box.min_corner)
    max_c = np.array(obj.bounding_box.max_corner)

    tolerance = 1e-6
    assert np.all(verts >= min_c - tolerance), "vertices below bounding box min"
    assert np.all(verts <= max_c + tolerance), "vertices above bounding box max"


# --- Parameter validation ---


def test_invalid_n_raises() -> None:
    """n below minimum raises ValueError."""
    gen = BourSurfaceGenerator()
    with pytest.raises(ValueError, match="n must be"):
        gen.generate(params={"n": 0})


def test_invalid_r_max_raises() -> None:
    """Non-positive r_max raises ValueError."""
    gen = BourSurfaceGenerator()
    with pytest.raises(ValueError, match="r_max must be positive"):
        gen.generate(params={"r_max": 0.0})
    with pytest.raises(ValueError, match="r_max must be positive"):
        gen.generate(params={"r_max": -1.0})


def test_invalid_grid_resolution_raises() -> None:
    """Grid resolution below minimum raises ValueError."""
    gen = BourSurfaceGenerator()
    with pytest.raises(ValueError, match="grid_resolution must be"):
        gen.generate(grid_resolution=2)


# --- r_max affects extent ---


def test_r_max_affects_extent() -> None:
    """Larger r_max produces larger mesh extent."""
    gen = BourSurfaceGenerator()
    obj_small = gen.generate(params={"r_max": 0.5}, grid_resolution=16)
    obj_large = gen.generate(params={"r_max": 2.0}, grid_resolution=16)

    assert obj_small.mesh is not None and obj_large.mesh is not None
    extent_small = np.ptp(obj_small.mesh.vertices, axis=0).max()
    extent_large = np.ptp(obj_large.mesh.vertices, axis=0).max()
    assert extent_large > extent_small
