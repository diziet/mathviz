"""Tests for the Menger sponge fractal generator.

Covers: level 0 single cube, level 1 sub-cube count, level 3 vertex count,
registration, representation, determinism, parameter validation, and
bounding box.
"""

import numpy as np
import pytest

from mathviz.core.generator import clear_registry, get_generator, register
from mathviz.core.representation import RepresentationType
from mathviz.generators.fractals.menger_sponge import (
    MengerSpongeGenerator,
    _build_cube_set,
)


@pytest.fixture(autouse=True)
def _clean_registry():
    """Reset registry and register the Menger sponge generator."""
    clear_registry(suppress_discovery=True)
    register(MengerSpongeGenerator)
    yield
    clear_registry(suppress_discovery=True)


# --- Level 0: single cube ---


def test_level_0_single_cube() -> None:
    """Level 0 produces a single cube with 8 vertices and 12 triangles."""
    gen = MengerSpongeGenerator()
    obj = gen.generate(params={"level": 0})
    obj.validate_or_raise()

    assert obj.mesh is not None
    # A single cube: 6 faces * 4 verts = 24 vertices (unshared per face)
    # 6 faces * 2 triangles = 12 triangles
    assert len(obj.mesh.faces) == 12
    assert len(obj.mesh.vertices) == 24


# --- Level 1: 20 sub-cubes ---


def test_level_1_produces_20_subcubes() -> None:
    """Level 1 produces 20 sub-cubes."""
    cubes = _build_cube_set(level=1, size=1.0)
    assert len(cubes) == 20


def test_level_1_valid_mesh() -> None:
    """Level 1 produces a valid mesh."""
    gen = MengerSpongeGenerator()
    obj = gen.generate(params={"level": 1})
    obj.validate_or_raise()

    assert obj.mesh is not None
    assert len(obj.mesh.vertices) > 0
    assert len(obj.mesh.faces) > 0


# --- Level 3: expected counts ---


def test_level_3_cube_count() -> None:
    """Level 3 produces 20^3 = 8000 sub-cubes."""
    cubes = _build_cube_set(level=3, size=1.0)
    assert len(cubes) == 20 ** 3


def test_level_3_valid_mesh() -> None:
    """Level 3 produces a valid mesh with expected vertex count."""
    gen = MengerSpongeGenerator()
    obj = gen.generate(params={"level": 3})
    obj.validate_or_raise()

    assert obj.mesh is not None
    # Each exposed face contributes 4 vertices
    assert len(obj.mesh.vertices) > 0
    assert len(obj.mesh.vertices) % 4 == 0
    # Each exposed face contributes 2 triangles
    assert len(obj.mesh.faces) == len(obj.mesh.vertices) // 4 * 2


# --- No NaN/Inf ---


def test_no_nan_or_inf_in_vertices() -> None:
    """Mesh vertices contain no NaN or Inf values."""
    gen = MengerSpongeGenerator()
    obj = gen.generate(params={"level": 2})
    assert obj.mesh is not None
    assert np.all(np.isfinite(obj.mesh.vertices))


# --- Registration ---


def test_registered_by_name() -> None:
    """Generator is discoverable via the registry by name."""
    found = get_generator("menger_sponge")
    assert found is MengerSpongeGenerator


def test_alias_discoverable() -> None:
    """Generator is discoverable via the menger alias."""
    found = get_generator("menger")
    assert found is MengerSpongeGenerator


# --- Representation ---


def test_default_representation_surface_shell() -> None:
    """Default representation is SURFACE_SHELL."""
    gen = MengerSpongeGenerator()
    rep = gen.get_default_representation()
    assert rep.type == RepresentationType.SURFACE_SHELL


# --- Metadata ---


def test_metadata_recorded() -> None:
    """Generator name, category, and parameters are recorded."""
    gen = MengerSpongeGenerator()
    obj = gen.generate(params={"level": 1})
    assert obj.generator_name == "menger_sponge"
    assert obj.category == "fractals"
    assert isinstance(obj.parameters, dict)
    assert "level" in obj.parameters
    assert "size" in obj.parameters


# --- Determinism ---


def test_determinism() -> None:
    """Same params produce identical geometry."""
    gen = MengerSpongeGenerator()
    obj1 = gen.generate(params={"level": 2}, seed=42)
    obj2 = gen.generate(params={"level": 2}, seed=42)

    assert obj1.mesh is not None and obj2.mesh is not None
    np.testing.assert_array_equal(obj1.mesh.vertices, obj2.mesh.vertices)
    np.testing.assert_array_equal(obj1.mesh.faces, obj2.mesh.faces)


# --- Bounding box ---


def test_vertices_within_bounding_box() -> None:
    """All mesh vertices lie within the declared bounding box."""
    gen = MengerSpongeGenerator()
    obj = gen.generate(params={"level": 2})
    assert obj.mesh is not None
    assert obj.bounding_box is not None

    verts = obj.mesh.vertices
    min_c = np.array(obj.bounding_box.min_corner)
    max_c = np.array(obj.bounding_box.max_corner)

    tolerance = 1e-6
    assert np.all(verts >= min_c - tolerance), "vertices below bounding box min"
    assert np.all(verts <= max_c + tolerance), "vertices above bounding box max"


# --- Parameter validation ---


def test_negative_level_raises() -> None:
    """Negative level raises ValueError."""
    gen = MengerSpongeGenerator()
    with pytest.raises(ValueError, match="level must be >= 0"):
        gen.generate(params={"level": -1})


def test_level_exceeds_max_raises() -> None:
    """Level above maximum raises ValueError."""
    gen = MengerSpongeGenerator()
    with pytest.raises(ValueError, match="level must be <="):
        gen.generate(params={"level": 5})


def test_nonpositive_size_raises() -> None:
    """Non-positive or too-small size raises ValueError."""
    gen = MengerSpongeGenerator()
    with pytest.raises(ValueError, match="size must be"):
        gen.generate(params={"size": 0.0})
    with pytest.raises(ValueError, match="size must be"):
        gen.generate(params={"size": -1.0})


# --- Size parameter ---


def test_size_affects_extent() -> None:
    """Larger size produces larger mesh extent."""
    gen = MengerSpongeGenerator()
    obj_small = gen.generate(params={"level": 1, "size": 0.5})
    obj_large = gen.generate(params={"level": 1, "size": 2.0})

    assert obj_small.mesh is not None and obj_large.mesh is not None
    extent_small = np.ptp(obj_small.mesh.vertices, axis=0).max()
    extent_large = np.ptp(obj_large.mesh.vertices, axis=0).max()
    assert extent_large > extent_small
