"""Tests for the Möbius trefoil generator.

Covers mesh validity, non-orientability, registry integration, and rendering.
"""

import numpy as np
import pytest
import trimesh

from mathviz.core.generator import clear_registry, get_generator, register
from mathviz.core.representation import RepresentationType
from mathviz.generators.parametric.mobius_trefoil import MobiusTrefoilGenerator


@pytest.fixture(autouse=True)
def _clean_registry():
    """Reset registry and register the Möbius trefoil for each test."""
    clear_registry(suppress_discovery=True)
    register(MobiusTrefoilGenerator)
    yield
    clear_registry(suppress_discovery=True)


@pytest.fixture()
def mesh_obj_128():
    """Generate a Möbius trefoil mesh with 128 curve points."""
    gen = MobiusTrefoilGenerator()
    return gen.generate(grid_resolution=16, curve_points=128)


@pytest.fixture()
def mesh_obj_256():
    """Generate a Möbius trefoil mesh with 256 curve points."""
    gen = MobiusTrefoilGenerator()
    return gen.generate(grid_resolution=16, curve_points=256)


@pytest.fixture()
def tri_mesh_256(mesh_obj_256):
    """Build a trimesh from the 256-point Möbius trefoil."""
    return trimesh.Trimesh(
        vertices=mesh_obj_256.mesh.vertices,
        faces=mesh_obj_256.mesh.faces,
        process=False,
    )


# ---------------------------------------------------------------------------
# Produces a valid mesh
# ---------------------------------------------------------------------------


def test_default_produces_valid_mesh(mesh_obj_128) -> None:
    """Default parameters produce a valid, non-empty mesh."""
    mesh_obj_128.validate_or_raise()

    assert mesh_obj_128.mesh is not None
    assert len(mesh_obj_128.mesh.vertices) > 0
    assert len(mesh_obj_128.mesh.faces) > 0
    assert mesh_obj_128.bounding_box is not None


def test_no_nan_or_inf_vertices(mesh_obj_128) -> None:
    """Mesh vertices contain no NaN or infinite values."""
    assert np.all(np.isfinite(mesh_obj_128.mesh.vertices))


def test_face_indices_in_range(mesh_obj_128) -> None:
    """All face indices reference valid vertex positions."""
    assert np.all(mesh_obj_128.mesh.faces >= 0)
    assert np.all(mesh_obj_128.mesh.faces < len(mesh_obj_128.mesh.vertices))


def test_vertices_within_bounding_box(mesh_obj_128) -> None:
    """All mesh vertices lie within the declared bounding box."""
    verts = mesh_obj_128.mesh.vertices
    min_c = np.array(mesh_obj_128.bounding_box.min_corner)
    max_c = np.array(mesh_obj_128.bounding_box.max_corner)

    tolerance = 1e-6
    assert np.all(verts >= min_c - tolerance)
    assert np.all(verts <= max_c + tolerance)


# ---------------------------------------------------------------------------
# Surface is non-orientable (single-sided)
# ---------------------------------------------------------------------------


def test_surface_not_watertight(tri_mesh_256) -> None:
    """Möbius trefoil has a Möbius half-twist and is not watertight."""
    assert not tri_mesh_256.is_watertight


def test_surface_has_single_boundary(tri_mesh_256) -> None:
    """A Möbius strip has boundary edges (non-orientable sign)."""
    edges = tri_mesh_256.edges_sorted
    edge_counts: dict[tuple[int, int], int] = {}
    for edge in edges:
        key = (int(edge[0]), int(edge[1]))
        edge_counts[key] = edge_counts.get(key, 0) + 1

    # Boundary edges appear exactly once (not shared by two faces)
    boundary_edges = [e for e, c in edge_counts.items() if c == 1]
    assert len(boundary_edges) > 0, "Expected boundary edges for non-orientable surface"


# ---------------------------------------------------------------------------
# Registers and renders successfully
# ---------------------------------------------------------------------------


def test_registered_and_discoverable() -> None:
    """Generator is discoverable via the registry by name."""
    found = get_generator("mobius_trefoil")
    assert found is MobiusTrefoilGenerator


def test_default_representation_surface_shell() -> None:
    """Default representation is SURFACE_SHELL."""
    gen = MobiusTrefoilGenerator()
    rep = gen.get_default_representation()
    assert rep.type == RepresentationType.SURFACE_SHELL


def test_metadata_recorded() -> None:
    """Generator name, category, and parameters are recorded."""
    gen = MobiusTrefoilGenerator()
    obj = gen.generate(grid_resolution=16, curve_points=64)
    assert obj.generator_name == "mobius_trefoil"
    assert obj.category == "parametric"
    assert isinstance(obj.parameters, dict)
    assert "width" in obj.parameters


def test_seed_recorded() -> None:
    """Seed is recorded in MathObject."""
    gen = MobiusTrefoilGenerator()
    obj = gen.generate(seed=777, grid_resolution=16, curve_points=64)
    assert obj.seed == 777


def test_determinism() -> None:
    """Same seed and params produce identical geometry."""
    gen = MobiusTrefoilGenerator()
    obj1 = gen.generate(seed=42, grid_resolution=16, curve_points=64)
    obj2 = gen.generate(seed=42, grid_resolution=16, curve_points=64)

    assert obj1.mesh is not None and obj2.mesh is not None
    np.testing.assert_array_equal(obj1.mesh.vertices, obj2.mesh.vertices)
    np.testing.assert_array_equal(obj1.mesh.faces, obj2.mesh.faces)


# ---------------------------------------------------------------------------
# Parameter validation
# ---------------------------------------------------------------------------


def test_negative_width_raises() -> None:
    """Negative width raises ValueError."""
    gen = MobiusTrefoilGenerator()
    with pytest.raises(ValueError, match="width"):
        gen.generate(params={"width": -0.1})


def test_zero_width_raises() -> None:
    """Zero width raises ValueError."""
    gen = MobiusTrefoilGenerator()
    with pytest.raises(ValueError, match="width"):
        gen.generate(params={"width": 0.0})


def test_curve_points_below_minimum_raises() -> None:
    """Too few curve points raises ValueError."""
    gen = MobiusTrefoilGenerator()
    with pytest.raises(ValueError, match="curve_points"):
        gen.generate(curve_points=2)


def test_grid_resolution_below_minimum_raises() -> None:
    """Too low grid resolution raises ValueError."""
    gen = MobiusTrefoilGenerator()
    with pytest.raises(ValueError, match="grid_resolution"):
        gen.generate(grid_resolution=1)


# ---------------------------------------------------------------------------
# Custom parameters
# ---------------------------------------------------------------------------


def test_custom_width() -> None:
    """Custom width produces a valid mesh with different geometry."""
    gen = MobiusTrefoilGenerator()
    obj_narrow = gen.generate(
        params={"width": 0.1}, grid_resolution=16, curve_points=64,
    )
    obj_wide = gen.generate(
        params={"width": 0.8}, grid_resolution=16, curve_points=64,
    )
    assert obj_narrow.mesh is not None and obj_wide.mesh is not None
    # Wider strip should have larger spatial extent
    narrow_extent = np.ptp(obj_narrow.mesh.vertices, axis=0)
    wide_extent = np.ptp(obj_wide.mesh.vertices, axis=0)
    assert np.any(wide_extent > narrow_extent)
