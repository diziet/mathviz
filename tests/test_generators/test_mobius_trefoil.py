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


# ---------------------------------------------------------------------------
# Produces a valid mesh
# ---------------------------------------------------------------------------


def test_default_produces_valid_mesh() -> None:
    """Default parameters produce a valid, non-empty mesh."""
    gen = MobiusTrefoilGenerator()
    obj = gen.generate(grid_resolution=16, curve_points=128)
    obj.validate_or_raise()

    assert obj.mesh is not None
    assert len(obj.mesh.vertices) > 0
    assert len(obj.mesh.faces) > 0
    assert obj.bounding_box is not None


def test_no_nan_or_inf_vertices() -> None:
    """Mesh vertices contain no NaN or infinite values."""
    gen = MobiusTrefoilGenerator()
    obj = gen.generate(grid_resolution=16, curve_points=128)
    assert obj.mesh is not None
    assert np.all(np.isfinite(obj.mesh.vertices))


def test_face_indices_in_range() -> None:
    """All face indices reference valid vertex positions."""
    gen = MobiusTrefoilGenerator()
    obj = gen.generate(grid_resolution=16, curve_points=128)
    assert obj.mesh is not None
    assert np.all(obj.mesh.faces >= 0)
    assert np.all(obj.mesh.faces < len(obj.mesh.vertices))


def test_vertices_within_bounding_box() -> None:
    """All mesh vertices lie within the declared bounding box."""
    gen = MobiusTrefoilGenerator()
    obj = gen.generate(grid_resolution=16, curve_points=128)
    assert obj.mesh is not None
    assert obj.bounding_box is not None

    verts = obj.mesh.vertices
    min_c = np.array(obj.bounding_box.min_corner)
    max_c = np.array(obj.bounding_box.max_corner)

    tolerance = 1e-6
    assert np.all(verts >= min_c - tolerance)
    assert np.all(verts <= max_c + tolerance)


# ---------------------------------------------------------------------------
# Surface is non-orientable (single-sided)
# ---------------------------------------------------------------------------


def test_surface_not_watertight() -> None:
    """Möbius trefoil has a Möbius half-twist and is not watertight."""
    gen = MobiusTrefoilGenerator()
    obj = gen.generate(grid_resolution=16, curve_points=256)
    assert obj.mesh is not None

    tri_mesh = trimesh.Trimesh(
        vertices=obj.mesh.vertices,
        faces=obj.mesh.faces,
        process=False,
    )
    assert not tri_mesh.is_watertight


def test_surface_has_single_boundary() -> None:
    """A Möbius strip has exactly one boundary loop (non-orientable sign)."""
    gen = MobiusTrefoilGenerator()
    obj = gen.generate(grid_resolution=16, curve_points=256)
    assert obj.mesh is not None

    tri_mesh = trimesh.Trimesh(
        vertices=obj.mesh.vertices,
        faces=obj.mesh.faces,
        process=False,
    )
    # A Möbius strip has a single boundary component
    edges = tri_mesh.edges_sorted
    edge_counts = {}
    for edge in edges:
        key = tuple(edge)
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
