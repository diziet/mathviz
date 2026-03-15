"""Tests for Dupin cyclide generator.

Covers: mesh validity, torus-like vs horn-like shapes, registration,
representation, determinism, parameter validation, and bounding box.
"""

import numpy as np
import pytest

from mathviz.core.generator import clear_registry, get_generator, register
from mathviz.core.representation import RepresentationType
from mathviz.generators.parametric.dupin_cyclide import DupinCyclideGenerator


@pytest.fixture(autouse=True)
def _clean_registry():
    """Reset registry and register the Dupin cyclide generator."""
    clear_registry(suppress_discovery=True)
    register(DupinCyclideGenerator)
    yield
    clear_registry(suppress_discovery=True)


# --- Mesh validity ---


def test_default_produces_nonempty_mesh() -> None:
    """Generator produces a valid, non-empty mesh at defaults."""
    gen = DupinCyclideGenerator()
    obj = gen.generate(grid_resolution=32)
    obj.validate_or_raise()

    assert obj.mesh is not None
    assert len(obj.mesh.vertices) > 0
    assert len(obj.mesh.faces) > 0
    assert obj.bounding_box is not None


def test_no_nan_or_inf_in_vertices() -> None:
    """Mesh vertices contain no NaN or Inf values."""
    gen = DupinCyclideGenerator()
    obj = gen.generate(grid_resolution=32)
    assert obj.mesh is not None
    assert np.all(np.isfinite(obj.mesh.vertices))


# --- Torus-like vs horn-like shapes ---


def test_torus_like_shape() -> None:
    """Small d relative to b produces a torus-like shape (symmetric)."""
    gen = DupinCyclideGenerator()
    obj = gen.generate(
        params={"a": 2.0, "b": 1.0, "c": 0.5, "d": 0.3},
        grid_resolution=32,
    )
    assert obj.mesh is not None
    verts = obj.mesh.vertices

    # A torus-like cyclide has comparable x and y extents (near-symmetric)
    x_range = np.ptp(verts[:, 0])
    y_range = np.ptp(verts[:, 1])
    ratio = min(x_range, y_range) / max(x_range, y_range)
    assert ratio > 0.5, "torus-like shape should be roughly symmetric in x/y"


def test_horn_like_shape() -> None:
    """Larger d shifts the cyclide further from a standard torus."""
    gen = DupinCyclideGenerator()
    obj_small_d = gen.generate(
        params={"a": 2.0, "b": 1.0, "c": 0.5, "d": 0.3},
        grid_resolution=32,
    )
    obj_large_d = gen.generate(
        params={"a": 2.0, "b": 1.0, "c": 0.5, "d": 1.8},
        grid_resolution=32,
    )
    assert obj_small_d.mesh is not None and obj_large_d.mesh is not None

    # Different d values produce meaningfully different geometry
    assert not np.allclose(
        obj_small_d.mesh.vertices, obj_large_d.mesh.vertices,
    )
    # Larger d shifts the center of mass along x
    cx_small = np.mean(obj_small_d.mesh.vertices[:, 0])
    cx_large = np.mean(obj_large_d.mesh.vertices[:, 0])
    assert abs(cx_small - cx_large) > 0.1, (
        "different d values should shift the shape"
    )


def test_different_d_produces_different_geometry() -> None:
    """Changing d parameter produces measurably different geometry."""
    gen = DupinCyclideGenerator()
    obj1 = gen.generate(params={"d": 0.3}, grid_resolution=16)
    obj2 = gen.generate(params={"d": 0.9}, grid_resolution=16)

    assert obj1.mesh is not None and obj2.mesh is not None
    assert not np.allclose(obj1.mesh.vertices, obj2.mesh.vertices)


# --- Registration ---


def test_registered_and_discoverable() -> None:
    """Generator is discoverable via the registry by name."""
    found = get_generator("dupin_cyclide")
    assert found is DupinCyclideGenerator


def test_alias_discoverable() -> None:
    """Generator is discoverable via the cyclide alias."""
    found = get_generator("cyclide")
    assert found is DupinCyclideGenerator


# --- Representation ---


def test_default_representation_surface_shell() -> None:
    """Default representation is SURFACE_SHELL."""
    gen = DupinCyclideGenerator()
    rep = gen.get_default_representation()
    assert rep.type == RepresentationType.SURFACE_SHELL


# --- Metadata ---


def test_metadata_recorded() -> None:
    """Generator name, category, and parameters are recorded."""
    gen = DupinCyclideGenerator()
    obj = gen.generate(grid_resolution=16)
    assert obj.generator_name == "dupin_cyclide"
    assert obj.category == "parametric"
    assert isinstance(obj.parameters, dict)
    assert "a" in obj.parameters
    assert "b" in obj.parameters
    assert "c" in obj.parameters
    assert "d" in obj.parameters


def test_seed_recorded() -> None:
    """Seed is recorded in MathObject."""
    gen = DupinCyclideGenerator()
    obj = gen.generate(seed=777, grid_resolution=16)
    assert obj.seed == 777


# --- Determinism ---


def test_determinism() -> None:
    """Same seed and params produce identical geometry."""
    gen = DupinCyclideGenerator()
    obj1 = gen.generate(seed=42, grid_resolution=16)
    obj2 = gen.generate(seed=42, grid_resolution=16)

    assert obj1.mesh is not None and obj2.mesh is not None
    np.testing.assert_array_equal(obj1.mesh.vertices, obj2.mesh.vertices)
    np.testing.assert_array_equal(obj1.mesh.faces, obj2.mesh.faces)


# --- Bounding box ---


def test_vertices_within_bounding_box() -> None:
    """All mesh vertices lie within the declared bounding box."""
    gen = DupinCyclideGenerator()
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


def test_a_affects_scale() -> None:
    """Larger a produces different mesh extent."""
    gen = DupinCyclideGenerator()
    obj_small = gen.generate(
        params={"a": 1.0, "c": 0.3}, grid_resolution=16,
    )
    obj_large = gen.generate(
        params={"a": 3.0, "c": 0.3}, grid_resolution=16,
    )
    assert obj_small.mesh is not None and obj_large.mesh is not None
    assert not np.allclose(obj_small.mesh.vertices, obj_large.mesh.vertices)


def test_b_affects_tube_size() -> None:
    """Changing b alters the tube cross-section size."""
    gen = DupinCyclideGenerator()
    obj_thin = gen.generate(params={"b": 0.3, "c": 0.3}, grid_resolution=16)
    obj_thick = gen.generate(params={"b": 0.9, "c": 0.3}, grid_resolution=16)

    assert obj_thin.mesh is not None and obj_thick.mesh is not None
    z_range_thin = np.ptp(obj_thin.mesh.vertices[:, 2])
    z_range_thick = np.ptp(obj_thick.mesh.vertices[:, 2])
    assert z_range_thick > z_range_thin


# --- Parameter validation ---


def test_invalid_a_raises() -> None:
    """Non-positive a raises ValueError."""
    gen = DupinCyclideGenerator()
    with pytest.raises(ValueError, match="a must be positive"):
        gen.generate(params={"a": 0.0})
    with pytest.raises(ValueError, match="a must be positive"):
        gen.generate(params={"a": -1.0})


def test_invalid_b_raises() -> None:
    """Non-positive b raises ValueError."""
    gen = DupinCyclideGenerator()
    with pytest.raises(ValueError, match="b must be positive"):
        gen.generate(params={"b": 0.0})


def test_invalid_c_raises() -> None:
    """Non-positive c raises ValueError."""
    gen = DupinCyclideGenerator()
    with pytest.raises(ValueError, match="c must be positive"):
        gen.generate(params={"c": 0.0})


def test_c_ge_a_raises() -> None:
    """c >= a raises ValueError (would cause singularity)."""
    gen = DupinCyclideGenerator()
    with pytest.raises(ValueError, match="c must be less than a"):
        gen.generate(params={"a": 1.0, "c": 1.0})
    with pytest.raises(ValueError, match="c must be less than a"):
        gen.generate(params={"a": 1.0, "c": 1.5})


def test_invalid_d_raises() -> None:
    """Non-positive d raises ValueError."""
    gen = DupinCyclideGenerator()
    with pytest.raises(ValueError, match="d must be positive"):
        gen.generate(params={"d": 0.0})


def test_invalid_grid_resolution_raises() -> None:
    """Grid resolution below minimum raises ValueError."""
    gen = DupinCyclideGenerator()
    with pytest.raises(ValueError, match="grid_resolution must be"):
        gen.generate(grid_resolution=2)
