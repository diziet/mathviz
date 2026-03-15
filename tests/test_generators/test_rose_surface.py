"""Tests for the rose surface generator.

Covers symmetric petal patterns for integer k1, distinct geometries for
different k1/k2 combinations, registration, and rendering.
"""

import numpy as np
import pytest

from mathviz.core.generator import clear_registry, get_generator, register
from mathviz.core.representation import RepresentationType
from mathviz.generators.parametric.rose_surface import RoseSurfaceGenerator


@pytest.fixture(autouse=True)
def _clean_registry():
    """Reset registry and register the rose surface generator for each test."""
    clear_registry(suppress_discovery=True)
    register(RoseSurfaceGenerator)
    yield
    clear_registry(suppress_discovery=True)


@pytest.fixture()
def default_obj():
    """Generate rose surface with default parameters at low resolution."""
    gen = RoseSurfaceGenerator()
    return gen.generate(grid_resolution=32)


# ---------------------------------------------------------------------------
# Integer k1 values produce symmetric petal patterns
# ---------------------------------------------------------------------------


def test_integer_k1_produces_valid_mesh(default_obj) -> None:
    """Integer k1 produces a valid, non-empty mesh."""
    default_obj.validate_or_raise()
    assert default_obj.mesh is not None
    assert len(default_obj.mesh.vertices) > 0
    assert len(default_obj.mesh.faces) > 0


def test_integer_k1_symmetric_petals() -> None:
    """Integer k1 values produce geometry symmetric about the z-axis.

    The rose surface should be centered at the origin and have equal
    positive and negative extents along each horizontal axis.
    """
    gen = RoseSurfaceGenerator()
    for k1 in (2, 3, 5):
        obj = gen.generate(params={"k1": k1, "k2": 2}, grid_resolution=64)
        verts = obj.mesh.vertices
        # Centroid xy should be near origin (symmetric pattern)
        centroid_x = np.mean(verts[:, 0])
        centroid_y = np.mean(verts[:, 1])
        assert abs(centroid_x) < 0.05, f"k1={k1}: centroid_x={centroid_x:.4f}"
        assert abs(centroid_y) < 0.05, f"k1={k1}: centroid_y={centroid_y:.4f}"


def test_different_k1_produce_different_vertex_counts() -> None:
    """Different k1 values produce different geometries (vertex positions)."""
    gen = RoseSurfaceGenerator()
    obj_2 = gen.generate(params={"k1": 2}, grid_resolution=32)
    obj_5 = gen.generate(params={"k1": 5}, grid_resolution=32)
    assert not np.allclose(obj_2.mesh.vertices, obj_5.mesh.vertices)


# ---------------------------------------------------------------------------
# Different k1/k2 combinations produce distinct geometries
# ---------------------------------------------------------------------------


def test_different_k2_produce_distinct_geometry() -> None:
    """Different k2 values produce different vertex sets."""
    gen = RoseSurfaceGenerator()
    obj_a = gen.generate(params={"k1": 3, "k2": 1}, grid_resolution=32)
    obj_b = gen.generate(params={"k1": 3, "k2": 4}, grid_resolution=32)
    assert not np.allclose(obj_a.mesh.vertices, obj_b.mesh.vertices)


def test_different_k1_k2_combinations_distinct() -> None:
    """Various k1/k2 combinations produce mutually distinct geometries."""
    gen = RoseSurfaceGenerator()
    combos = [(2, 3), (3, 2), (5, 1), (4, 4)]
    objects = [
        gen.generate(params={"k1": k1, "k2": k2}, grid_resolution=32)
        for k1, k2 in combos
    ]
    for i in range(len(objects)):
        for j in range(i + 1, len(objects)):
            assert not np.allclose(
                objects[i].mesh.vertices, objects[j].mesh.vertices,
            ), f"combos {combos[i]} and {combos[j]} should differ"


# ---------------------------------------------------------------------------
# Registers and renders successfully
# ---------------------------------------------------------------------------


def test_registered_and_discoverable() -> None:
    """Generator is discoverable via the registry by name."""
    found = get_generator("rose_surface")
    assert found is RoseSurfaceGenerator


def test_default_representation_surface_shell() -> None:
    """Default representation is SURFACE_SHELL."""
    gen = RoseSurfaceGenerator()
    rep = gen.get_default_representation()
    assert rep.type == RepresentationType.SURFACE_SHELL


def test_metadata_recorded(default_obj) -> None:
    """Generator name, category, and parameters are recorded."""
    assert default_obj.generator_name == "rose_surface"
    assert default_obj.category == "parametric"
    assert isinstance(default_obj.parameters, dict)
    assert "k1" in default_obj.parameters
    assert "k2" in default_obj.parameters


def test_seed_recorded() -> None:
    """Seed is recorded in MathObject."""
    gen = RoseSurfaceGenerator()
    obj = gen.generate(seed=777, grid_resolution=16)
    assert obj.seed == 777


def test_determinism() -> None:
    """Same seed and params produce identical geometry."""
    gen = RoseSurfaceGenerator()
    obj1 = gen.generate(seed=42, grid_resolution=16)
    obj2 = gen.generate(seed=42, grid_resolution=16)
    np.testing.assert_array_equal(obj1.mesh.vertices, obj2.mesh.vertices)
    np.testing.assert_array_equal(obj1.mesh.faces, obj2.mesh.faces)


# ---------------------------------------------------------------------------
# Mesh validity
# ---------------------------------------------------------------------------


def test_no_nan_or_inf_vertices(default_obj) -> None:
    """Mesh vertices contain no NaN or infinite values."""
    assert np.all(np.isfinite(default_obj.mesh.vertices))


def test_face_indices_in_range(default_obj) -> None:
    """All face indices reference valid vertex positions."""
    assert np.all(default_obj.mesh.faces >= 0)
    assert np.all(default_obj.mesh.faces < len(default_obj.mesh.vertices))


def test_vertices_within_bounding_box(default_obj) -> None:
    """All mesh vertices lie within the declared bounding box."""
    verts = default_obj.mesh.vertices
    min_c = np.array(default_obj.bounding_box.min_corner)
    max_c = np.array(default_obj.bounding_box.max_corner)
    tolerance = 1e-6
    assert np.all(verts >= min_c - tolerance)
    assert np.all(verts <= max_c + tolerance)


# ---------------------------------------------------------------------------
# Parameter validation
# ---------------------------------------------------------------------------


def test_k1_below_minimum_raises() -> None:
    """k1 below minimum raises ValueError."""
    gen = RoseSurfaceGenerator()
    with pytest.raises(ValueError, match="k1"):
        gen.generate(params={"k1": 0})


def test_k2_below_minimum_raises() -> None:
    """k2 below minimum raises ValueError."""
    gen = RoseSurfaceGenerator()
    with pytest.raises(ValueError, match="k2"):
        gen.generate(params={"k2": 0})


def test_grid_resolution_below_minimum_raises() -> None:
    """Too low grid resolution raises ValueError."""
    gen = RoseSurfaceGenerator()
    with pytest.raises(ValueError, match="grid_resolution"):
        gen.generate(grid_resolution=3)
