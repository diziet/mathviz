"""Tests for the Seifert surface generator.

Covers: mesh validity, boundary approximation, distinct knot types,
registration, representation, determinism, parameter validation,
and bounding box correctness.
"""

import numpy as np
import pytest

from mathviz.core.generator import clear_registry, get_generator, register
from mathviz.core.representation import RepresentationType
from mathviz.generators.parametric.seifert_surface import (
    SeifertSurfaceGenerator,
    _figure_eight_reference_knot,
    _trefoil_reference_knot,
)


@pytest.fixture(autouse=True)
def _clean_registry():
    """Reset registry and register the Seifert surface generator."""
    clear_registry(suppress_discovery=True)
    register(SeifertSurfaceGenerator)
    yield
    clear_registry(suppress_discovery=True)


# --- Mesh validity ---


@pytest.mark.parametrize("knot_type", ["trefoil", "figure_eight"])
def test_produces_nonempty_mesh(knot_type: str) -> None:
    """Seifert surface produces a valid, non-empty mesh."""
    gen = SeifertSurfaceGenerator()
    obj = gen.generate(params={"knot_type": knot_type}, grid_resolution=32)
    obj.validate_or_raise()

    assert obj.mesh is not None
    assert len(obj.mesh.vertices) > 0
    assert len(obj.mesh.faces) > 0
    assert obj.bounding_box is not None


# --- Boundary approximation ---


def test_trefoil_boundary_near_knot() -> None:
    """Mesh vertices include points close to the trefoil knot curve."""
    gen = SeifertSurfaceGenerator()
    obj = gen.generate(params={"knot_type": "trefoil"}, grid_resolution=64)
    assert obj.mesh is not None

    knot = _trefoil_reference_knot(200)
    verts = obj.mesh.vertices

    # For each knot sample, find nearest mesh vertex
    min_dists = np.array([
        np.min(np.linalg.norm(verts - kp, axis=1)) for kp in knot
    ])
    # Most knot points should have a nearby mesh vertex
    # Tolerance accounts for Milnor fiber clamping near boundary
    assert np.median(min_dists) < 0.6, (
        f"Median distance from knot to mesh: {np.median(min_dists):.3f}"
    )


def test_figure_eight_boundary_matches_knot() -> None:
    """Last row of figure-eight mesh matches the knot parameterization."""
    gen = SeifertSurfaceGenerator()
    n = 64
    obj = gen.generate(params={"knot_type": "figure_eight"}, grid_resolution=n)
    assert obj.mesh is not None

    # Last row of the grid (r=1) should be the knot
    boundary_start = (n - 1) * n
    boundary_verts = obj.mesh.vertices[boundary_start : boundary_start + n]

    ref = _figure_eight_reference_knot(n)
    np.testing.assert_allclose(boundary_verts, ref, atol=1e-10)


# --- Distinct surfaces ---


def test_different_knot_types_produce_distinct_surfaces() -> None:
    """Trefoil and figure-eight produce geometrically distinct meshes."""
    gen = SeifertSurfaceGenerator()
    obj_trefoil = gen.generate(
        params={"knot_type": "trefoil"}, grid_resolution=32,
    )
    obj_fig8 = gen.generate(
        params={"knot_type": "figure_eight"}, grid_resolution=32,
    )

    assert obj_trefoil.mesh is not None and obj_fig8.mesh is not None

    # Different extents or vertex distributions
    extent_trefoil = np.ptp(obj_trefoil.mesh.vertices, axis=0)
    extent_fig8 = np.ptp(obj_fig8.mesh.vertices, axis=0)

    assert not np.allclose(extent_trefoil, extent_fig8, atol=0.1), (
        "Trefoil and figure-eight should have distinct spatial extents"
    )


def test_different_theta_produces_distinct_mesh() -> None:
    """Different theta values produce different geometry."""
    gen = SeifertSurfaceGenerator()
    obj1 = gen.generate(params={"theta": 0.0}, grid_resolution=32)
    obj2 = gen.generate(params={"theta": 1.0}, grid_resolution=32)

    assert obj1.mesh is not None and obj2.mesh is not None
    assert not np.allclose(obj1.mesh.vertices, obj2.mesh.vertices, atol=1e-6)


# --- Registration ---


def test_registered_and_discoverable() -> None:
    """Generator is discoverable via the registry by name."""
    found = get_generator("seifert_surface")
    assert found is SeifertSurfaceGenerator


def test_alias_discoverable() -> None:
    """Generator is discoverable via the seifert alias."""
    found = get_generator("seifert")
    assert found is SeifertSurfaceGenerator


def test_default_representation_surface_shell() -> None:
    """Default representation is SURFACE_SHELL."""
    gen = SeifertSurfaceGenerator()
    rep = gen.get_default_representation()
    assert rep.type == RepresentationType.SURFACE_SHELL


# --- Metadata ---


def test_metadata_recorded() -> None:
    """Generator name, category, and parameters are recorded."""
    gen = SeifertSurfaceGenerator()
    obj = gen.generate(grid_resolution=16)
    assert obj.generator_name == "seifert_surface"
    assert obj.category == "parametric"
    assert isinstance(obj.parameters, dict)
    assert "knot_type" in obj.parameters
    assert "theta" in obj.parameters


def test_seed_recorded() -> None:
    """Seed is recorded in MathObject."""
    gen = SeifertSurfaceGenerator()
    obj = gen.generate(seed=777, grid_resolution=16)
    assert obj.seed == 777


# --- Determinism ---


@pytest.mark.parametrize("knot_type", ["trefoil", "figure_eight"])
def test_determinism(knot_type: str) -> None:
    """Same params produce identical geometry."""
    gen = SeifertSurfaceGenerator()
    obj1 = gen.generate(
        params={"knot_type": knot_type}, seed=42, grid_resolution=16,
    )
    obj2 = gen.generate(
        params={"knot_type": knot_type}, seed=42, grid_resolution=16,
    )

    assert obj1.mesh is not None and obj2.mesh is not None
    np.testing.assert_array_equal(obj1.mesh.vertices, obj2.mesh.vertices)
    np.testing.assert_array_equal(obj1.mesh.faces, obj2.mesh.faces)


# --- Bounding box ---


def test_vertices_within_bounding_box() -> None:
    """All mesh vertices lie within the declared bounding box."""
    gen = SeifertSurfaceGenerator()
    for knot_type in ("trefoil", "figure_eight"):
        obj = gen.generate(
            params={"knot_type": knot_type}, grid_resolution=32,
        )
        assert obj.mesh is not None
        assert obj.bounding_box is not None

        verts = obj.mesh.vertices
        min_c = np.array(obj.bounding_box.min_corner)
        max_c = np.array(obj.bounding_box.max_corner)

        tolerance = 1e-6
        assert np.all(verts >= min_c - tolerance), (
            f"{knot_type}: vertices below bounding box min"
        )
        assert np.all(verts <= max_c + tolerance), (
            f"{knot_type}: vertices above bounding box max"
        )


# --- Parameter validation ---


def test_invalid_knot_type_raises() -> None:
    """Invalid knot_type raises ValueError."""
    gen = SeifertSurfaceGenerator()
    with pytest.raises(ValueError, match="knot_type must be one of"):
        gen.generate(params={"knot_type": "cinquefoil"})


def test_invalid_theta_raises() -> None:
    """Out-of-range theta raises ValueError."""
    gen = SeifertSurfaceGenerator()
    with pytest.raises(ValueError, match="theta must be in"):
        gen.generate(params={"theta": -0.1})
    with pytest.raises(ValueError, match="theta must be in"):
        gen.generate(params={"theta": 7.0})


def test_invalid_grid_resolution_raises() -> None:
    """Grid resolution below minimum raises ValueError."""
    gen = SeifertSurfaceGenerator()
    with pytest.raises(ValueError, match="grid_resolution must be"):
        gen.generate(grid_resolution=2)
