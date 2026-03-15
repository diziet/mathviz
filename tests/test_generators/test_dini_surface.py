"""Tests for Dini's surface generator.

Covers: mesh validity, spiral length scaling with turns, registration,
representation, determinism, parameter validation, and bounding box.
"""

import numpy as np
import pytest

from mathviz.core.generator import clear_registry, get_generator, register
from mathviz.core.representation import RepresentationType
from mathviz.generators.parametric.dini_surface import DiniSurfaceGenerator


@pytest.fixture(autouse=True)
def _clean_registry():
    """Reset registry and register the Dini surface generator."""
    clear_registry(suppress_discovery=True)
    register(DiniSurfaceGenerator)
    yield
    clear_registry(suppress_discovery=True)


# --- Mesh validity ---


def test_default_produces_nonempty_mesh() -> None:
    """Generator produces a valid, non-empty mesh at defaults."""
    gen = DiniSurfaceGenerator()
    obj = gen.generate(grid_resolution=32)
    obj.validate_or_raise()

    assert obj.mesh is not None
    assert len(obj.mesh.vertices) > 0
    assert len(obj.mesh.faces) > 0
    assert obj.bounding_box is not None


def test_no_nan_or_inf_in_vertices() -> None:
    """Mesh vertices contain no NaN or Inf values."""
    gen = DiniSurfaceGenerator()
    obj = gen.generate(grid_resolution=32)
    assert obj.mesh is not None
    assert np.all(np.isfinite(obj.mesh.vertices))


# --- More turns produce a longer spiral ---


def test_more_turns_longer_spiral() -> None:
    """More turns produce a longer spiral along the z-axis."""
    gen = DiniSurfaceGenerator()
    obj_short = gen.generate(params={"turns": 1}, grid_resolution=32)
    obj_long = gen.generate(params={"turns": 4}, grid_resolution=32)

    assert obj_short.mesh is not None and obj_long.mesh is not None
    z_range_short = np.ptp(obj_short.mesh.vertices[:, 2])
    z_range_long = np.ptp(obj_long.mesh.vertices[:, 2])
    assert z_range_long > z_range_short


def test_turns_scales_u_range() -> None:
    """More turns increase the maximum u value used in generation."""
    gen = DiniSurfaceGenerator()
    obj2 = gen.generate(params={"turns": 2}, grid_resolution=16)
    obj5 = gen.generate(params={"turns": 5}, grid_resolution=16)

    assert obj2.mesh is not None and obj5.mesh is not None
    # With b*u contribution to z, more turns means larger z extent
    extent_2 = np.ptp(obj2.mesh.vertices[:, 2])
    extent_5 = np.ptp(obj5.mesh.vertices[:, 2])
    assert extent_5 > extent_2 * 1.3


# --- Registration ---


def test_registered_and_discoverable() -> None:
    """Generator is discoverable via the registry by name."""
    found = get_generator("dini_surface")
    assert found is DiniSurfaceGenerator


def test_alias_discoverable() -> None:
    """Generator is discoverable via the dini alias."""
    found = get_generator("dini")
    assert found is DiniSurfaceGenerator


# --- Representation ---


def test_default_representation_surface_shell() -> None:
    """Default representation is SURFACE_SHELL."""
    gen = DiniSurfaceGenerator()
    rep = gen.get_default_representation()
    assert rep.type == RepresentationType.SURFACE_SHELL


# --- Metadata ---


def test_metadata_recorded() -> None:
    """Generator name, category, and parameters are recorded."""
    gen = DiniSurfaceGenerator()
    obj = gen.generate(grid_resolution=16)
    assert obj.generator_name == "dini_surface"
    assert obj.category == "parametric"
    assert isinstance(obj.parameters, dict)
    assert "a" in obj.parameters
    assert "b" in obj.parameters
    assert "turns" in obj.parameters


def test_seed_recorded() -> None:
    """Seed is recorded in MathObject."""
    gen = DiniSurfaceGenerator()
    obj = gen.generate(seed=777, grid_resolution=16)
    assert obj.seed == 777


# --- Determinism ---


def test_determinism() -> None:
    """Same seed and params produce identical geometry."""
    gen = DiniSurfaceGenerator()
    obj1 = gen.generate(seed=42, grid_resolution=16)
    obj2 = gen.generate(seed=42, grid_resolution=16)

    assert obj1.mesh is not None and obj2.mesh is not None
    np.testing.assert_array_equal(obj1.mesh.vertices, obj2.mesh.vertices)
    np.testing.assert_array_equal(obj1.mesh.faces, obj2.mesh.faces)


# --- Bounding box ---


def test_vertices_within_bounding_box() -> None:
    """All mesh vertices lie within the declared bounding box."""
    gen = DiniSurfaceGenerator()
    obj = gen.generate(grid_resolution=32)
    assert obj.mesh is not None
    assert obj.bounding_box is not None

    verts = obj.mesh.vertices
    min_c = np.array(obj.bounding_box.min_corner)
    max_c = np.array(obj.bounding_box.max_corner)

    tolerance = 1e-6
    assert np.all(verts >= min_c - tolerance), "vertices below bounding box min"
    assert np.all(verts <= max_c + tolerance), "vertices above bounding box max"


# --- Parameter effects ---


def test_scale_affects_extent() -> None:
    """Larger a produces larger mesh extent in x/y."""
    gen = DiniSurfaceGenerator()
    obj_small = gen.generate(params={"a": 0.5}, grid_resolution=16)
    obj_large = gen.generate(params={"a": 2.0}, grid_resolution=16)

    assert obj_small.mesh is not None and obj_large.mesh is not None
    xy_range_small = np.ptp(obj_small.mesh.vertices[:, :2], axis=0).max()
    xy_range_large = np.ptp(obj_large.mesh.vertices[:, :2], axis=0).max()
    assert xy_range_large > xy_range_small


def test_twist_rate_affects_z() -> None:
    """Larger b (twist rate) stretches the spiral along z."""
    gen = DiniSurfaceGenerator()
    obj_lo = gen.generate(params={"b": 0.1}, grid_resolution=16)
    obj_hi = gen.generate(params={"b": 0.5}, grid_resolution=16)

    assert obj_lo.mesh is not None and obj_hi.mesh is not None
    z_range_lo = np.ptp(obj_lo.mesh.vertices[:, 2])
    z_range_hi = np.ptp(obj_hi.mesh.vertices[:, 2])
    assert z_range_hi > z_range_lo


# --- Parameter validation ---


def test_invalid_a_raises() -> None:
    """Non-positive a raises ValueError."""
    gen = DiniSurfaceGenerator()
    with pytest.raises(ValueError, match="a must be positive"):
        gen.generate(params={"a": 0.0})
    with pytest.raises(ValueError, match="a must be positive"):
        gen.generate(params={"a": -1.0})


def test_invalid_b_raises() -> None:
    """Non-positive b raises ValueError."""
    gen = DiniSurfaceGenerator()
    with pytest.raises(ValueError, match="b must be positive"):
        gen.generate(params={"b": 0.0})


def test_invalid_turns_raises() -> None:
    """Turns below minimum raises ValueError."""
    gen = DiniSurfaceGenerator()
    with pytest.raises(ValueError, match="turns must be"):
        gen.generate(params={"turns": 0})


def test_invalid_grid_resolution_raises() -> None:
    """Grid resolution below minimum raises ValueError."""
    gen = DiniSurfaceGenerator()
    with pytest.raises(ValueError, match="grid_resolution must be"):
        gen.generate(grid_resolution=2)


# --- Pseudospherical property ---


def test_pseudospherical_shape() -> None:
    """Surface has cylindrical xy-extent bounded by a (pseudospherical)."""
    gen = DiniSurfaceGenerator()
    obj = gen.generate(params={"a": 1.0}, grid_resolution=64)
    assert obj.mesh is not None

    verts = obj.mesh.vertices
    xy_radius = np.sqrt(verts[:, 0] ** 2 + verts[:, 1] ** 2)
    # x = a*cos(u)*sin(v), y = a*sin(u)*sin(v) => r = a*sin(v) <= a
    assert np.all(xy_radius <= 1.0 + 1e-6)
