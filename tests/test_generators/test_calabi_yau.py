"""Tests for the Calabi-Yau manifold cross-section generator."""

import numpy as np
import pytest

from mathviz.core.generator import clear_registry, get_generator, register
from mathviz.core.representation import RepresentationType
from mathviz.generators.parametric.calabi_yau import CalabiYauGenerator


@pytest.fixture(autouse=True)
def _clean_registry():
    """Reset registry and register Calabi-Yau for each test."""
    clear_registry(suppress_discovery=True)
    register(CalabiYauGenerator)
    yield
    clear_registry(suppress_discovery=True)


# ---------------------------------------------------------------------------
# Produces a valid mesh
# ---------------------------------------------------------------------------


def test_default_produces_valid_mesh() -> None:
    """Default parameters produce a valid, non-empty mesh."""
    gen = CalabiYauGenerator()
    obj = gen.generate(grid_resolution=32)
    obj.validate_or_raise()

    assert obj.mesh is not None
    assert len(obj.mesh.vertices) > 0
    assert len(obj.mesh.faces) > 0
    assert obj.bounding_box is not None


def test_mesh_vertex_face_counts() -> None:
    """Verify expected vertex and face counts for multi-patch surface."""
    gen = CalabiYauGenerator()
    n = 5
    res = 16
    obj = gen.generate(params={"n": n}, grid_resolution=res)

    assert obj.mesh is not None
    # n patches, each with res*res vertices
    assert len(obj.mesh.vertices) == n * res * res
    # n patches, each with 2*(res-1)*(res-1) faces (open grid)
    expected_faces = n * 2 * (res - 1) * (res - 1)
    assert len(obj.mesh.faces) == expected_faces


# ---------------------------------------------------------------------------
# Different n values produce distinct geometries
# ---------------------------------------------------------------------------


def test_different_n_values_produce_distinct_geometries() -> None:
    """Different n values produce geometrically distinct meshes."""
    gen = CalabiYauGenerator()
    obj3 = gen.generate(params={"n": 3}, grid_resolution=16)
    obj5 = gen.generate(params={"n": 5}, grid_resolution=16)
    obj7 = gen.generate(params={"n": 7}, grid_resolution=16)

    assert obj3.mesh is not None
    assert obj5.mesh is not None
    assert obj7.mesh is not None

    # Different n → different vertex counts (n * res^2)
    assert len(obj3.mesh.vertices) != len(obj5.mesh.vertices)
    assert len(obj5.mesh.vertices) != len(obj7.mesh.vertices)

    # Even at matching resolution, shapes differ in vertex positions
    # Compare using bounding box extents or vertex stats
    extent3 = np.ptp(obj3.mesh.vertices, axis=0)
    extent5 = np.ptp(obj5.mesh.vertices, axis=0)
    assert not np.allclose(extent3, extent5, atol=0.01)


# ---------------------------------------------------------------------------
# Registers and renders successfully
# ---------------------------------------------------------------------------


def test_registered_and_discoverable() -> None:
    """Generator is discoverable via the registry by name."""
    found = get_generator("calabi_yau")
    assert found is CalabiYauGenerator


def test_default_representation_surface_shell() -> None:
    """Default representation is SURFACE_SHELL."""
    gen = CalabiYauGenerator()
    rep = gen.get_default_representation()
    assert rep.type == RepresentationType.SURFACE_SHELL


def test_metadata_recorded() -> None:
    """Generator name, category, and parameters are recorded."""
    gen = CalabiYauGenerator()
    obj = gen.generate(grid_resolution=16)
    assert obj.generator_name == "calabi_yau"
    assert obj.category == "parametric"
    assert isinstance(obj.parameters, dict)
    assert "n" in obj.parameters
    assert "alpha" in obj.parameters


# ---------------------------------------------------------------------------
# Seed and determinism
# ---------------------------------------------------------------------------


def test_seed_recorded() -> None:
    """Seed is recorded in MathObject."""
    gen = CalabiYauGenerator()
    obj = gen.generate(seed=999, grid_resolution=16)
    assert obj.seed == 999


def test_determinism() -> None:
    """Same seed and params produce identical geometry."""
    gen = CalabiYauGenerator()
    obj1 = gen.generate(seed=42, grid_resolution=16)
    obj2 = gen.generate(seed=42, grid_resolution=16)

    assert obj1.mesh is not None and obj2.mesh is not None
    np.testing.assert_array_equal(obj1.mesh.vertices, obj2.mesh.vertices)
    np.testing.assert_array_equal(obj1.mesh.faces, obj2.mesh.faces)


# ---------------------------------------------------------------------------
# Vertices within bounding box
# ---------------------------------------------------------------------------


def test_vertices_within_bounding_box() -> None:
    """All mesh vertices lie within the declared bounding box."""
    gen = CalabiYauGenerator()
    obj = gen.generate(grid_resolution=32)
    assert obj.mesh is not None
    assert obj.bounding_box is not None

    verts = obj.mesh.vertices
    min_c = np.array(obj.bounding_box.min_corner)
    max_c = np.array(obj.bounding_box.max_corner)

    tolerance = 1e-6
    assert np.all(verts >= min_c - tolerance), "vertices below bounding box min"
    assert np.all(verts <= max_c + tolerance), "vertices above bounding box max"


# ---------------------------------------------------------------------------
# Parameter validation
# ---------------------------------------------------------------------------


def test_invalid_n_raises_value_error() -> None:
    """n < 2 raises ValueError."""
    gen = CalabiYauGenerator()
    with pytest.raises(ValueError, match="n must be >= 2"):
        gen.generate(params={"n": 1}, grid_resolution=16)


def test_invalid_grid_resolution_raises_value_error() -> None:
    """grid_resolution below minimum raises ValueError."""
    gen = CalabiYauGenerator()
    with pytest.raises(ValueError, match="grid_resolution must be >= 4"):
        gen.generate(grid_resolution=2)


# ---------------------------------------------------------------------------
# Alpha parameter affects geometry
# ---------------------------------------------------------------------------


def test_alpha_affects_geometry() -> None:
    """Different alpha values produce different projections."""
    gen = CalabiYauGenerator()
    obj1 = gen.generate(params={"alpha": 0.0}, grid_resolution=16)
    obj2 = gen.generate(params={"alpha": np.pi / 2}, grid_resolution=16)

    assert obj1.mesh is not None and obj2.mesh is not None
    assert not np.allclose(obj1.mesh.vertices, obj2.mesh.vertices)
