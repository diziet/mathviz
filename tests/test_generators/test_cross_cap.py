"""Tests for the cross-cap surface generator.

Covers: mesh validity, non-orientability, registration, representation,
determinism, parameter validation, and bounding box correctness.
"""

import numpy as np
import pytest

from mathviz.core.generator import clear_registry, get_generator, register
from mathviz.core.representation import RepresentationType
from mathviz.generators.parametric.cross_cap import CrossCapGenerator


@pytest.fixture(autouse=True)
def _clean_registry():
    """Reset registry and register the cross-cap generator."""
    clear_registry(suppress_discovery=True)
    register(CrossCapGenerator)
    yield
    clear_registry(suppress_discovery=True)


def test_default_produces_nonempty_mesh() -> None:
    """Generator produces a valid, non-empty mesh at defaults."""
    gen = CrossCapGenerator()
    obj = gen.generate(grid_resolution=32)
    obj.validate_or_raise()

    assert obj.mesh is not None
    assert len(obj.mesh.vertices) > 0
    assert len(obj.mesh.faces) > 0
    assert obj.bounding_box is not None


def test_no_nan_or_inf_in_vertices() -> None:
    """Mesh vertices contain no NaN or Inf values."""
    gen = CrossCapGenerator()
    obj = gen.generate(grid_resolution=32)
    assert obj.mesh is not None

    assert np.all(np.isfinite(obj.mesh.vertices))


def test_self_intersections_present() -> None:
    """Cross-cap has self-intersections in R³.

    Multiple topologically distant vertices collapse near the origin,
    confirming the self-intersecting immersion.
    """
    gen = CrossCapGenerator()
    obj = gen.generate(grid_resolution=64)
    assert obj.mesh is not None

    verts = obj.mesh.vertices
    origin_dist = np.linalg.norm(verts, axis=1)
    near_origin = np.where(origin_dist < 0.05)[0]
    assert len(near_origin) > 1, "Expected multiple vertices near origin"


def test_registered_and_discoverable() -> None:
    """Generator is discoverable via the registry by name."""
    found = get_generator("cross_cap")
    assert found is CrossCapGenerator


def test_alias_discoverable() -> None:
    """Generator is discoverable via the crosscap alias."""
    found = get_generator("crosscap")
    assert found is CrossCapGenerator


def test_default_representation_surface_shell() -> None:
    """Default representation is SURFACE_SHELL."""
    gen = CrossCapGenerator()
    rep = gen.get_default_representation()
    assert rep.type == RepresentationType.SURFACE_SHELL


def test_metadata_recorded() -> None:
    """Generator name, category, and parameters are recorded."""
    gen = CrossCapGenerator()
    obj = gen.generate(grid_resolution=16)
    assert obj.generator_name == "cross_cap"
    assert obj.category == "parametric"
    assert isinstance(obj.parameters, dict)
    assert "scale" in obj.parameters


def test_seed_recorded() -> None:
    """Seed is recorded in MathObject."""
    gen = CrossCapGenerator()
    obj = gen.generate(seed=777, grid_resolution=16)
    assert obj.seed == 777


def test_determinism() -> None:
    """Same seed and params produce identical geometry."""
    gen = CrossCapGenerator()
    obj1 = gen.generate(seed=42, grid_resolution=16)
    obj2 = gen.generate(seed=42, grid_resolution=16)

    assert obj1.mesh is not None and obj2.mesh is not None
    np.testing.assert_array_equal(obj1.mesh.vertices, obj2.mesh.vertices)
    np.testing.assert_array_equal(obj1.mesh.faces, obj2.mesh.faces)


def test_vertices_within_bounding_box() -> None:
    """All mesh vertices lie within the declared bounding box."""
    gen = CrossCapGenerator()
    obj = gen.generate(grid_resolution=32)
    assert obj.mesh is not None
    assert obj.bounding_box is not None

    verts = obj.mesh.vertices
    min_c = np.array(obj.bounding_box.min_corner)
    max_c = np.array(obj.bounding_box.max_corner)

    tolerance = 1e-6
    assert np.all(verts >= min_c - tolerance), "vertices below bounding box min"
    assert np.all(verts <= max_c + tolerance), "vertices above bounding box max"


def test_scale_affects_extent() -> None:
    """Larger scale produces larger mesh extent."""
    gen = CrossCapGenerator()
    obj_small = gen.generate(params={"scale": 0.5}, grid_resolution=16)
    obj_large = gen.generate(params={"scale": 2.0}, grid_resolution=16)

    assert obj_small.mesh is not None and obj_large.mesh is not None
    extent_small = np.ptp(obj_small.mesh.vertices, axis=0).max()
    extent_large = np.ptp(obj_large.mesh.vertices, axis=0).max()
    assert extent_large > extent_small


def test_grid_resolution_affects_vertex_count() -> None:
    """Higher grid resolution produces more vertices."""
    gen = CrossCapGenerator()
    obj_low = gen.generate(grid_resolution=16)
    obj_high = gen.generate(grid_resolution=32)

    assert obj_low.mesh is not None and obj_high.mesh is not None
    assert len(obj_high.mesh.vertices) > len(obj_low.mesh.vertices)


def test_invalid_scale_raises() -> None:
    """Negative or zero scale raises ValueError."""
    gen = CrossCapGenerator()
    with pytest.raises(ValueError, match="scale must be positive"):
        gen.generate(params={"scale": 0.0})
    with pytest.raises(ValueError, match="scale must be positive"):
        gen.generate(params={"scale": -1.0})


def test_invalid_grid_resolution_raises() -> None:
    """Grid resolution below minimum raises ValueError."""
    gen = CrossCapGenerator()
    with pytest.raises(ValueError, match="grid_resolution must be"):
        gen.generate(grid_resolution=2)


def test_parametric_equations() -> None:
    """Vertices satisfy the cross-cap parametric equations.

    Verify that the generated vertices match the expected parametric
    formulas for a subset of (u, v) sample points.

    NOTE: Equations are intentionally duplicated here as an independent
    oracle rather than importing _evaluate_cross_cap from the module.
    """
    gen = CrossCapGenerator()
    obj = gen.generate(grid_resolution=32)
    assert obj.mesh is not None

    n = 32
    u_vals = np.linspace(0, np.pi, n, endpoint=True)
    v_vals = np.linspace(0, np.pi, n, endpoint=True)
    uu, vv = np.meshgrid(u_vals, v_vals, indexing="ij")

    expected_x = np.sin(uu) * np.sin(2.0 * vv) / 2.0
    expected_y = np.sin(2.0 * uu) * np.cos(vv) ** 2
    expected_z = np.cos(2.0 * uu) * np.cos(vv) ** 2

    expected = np.column_stack([
        expected_x.ravel(), expected_y.ravel(), expected_z.ravel(),
    ])

    np.testing.assert_allclose(obj.mesh.vertices, expected, atol=1e-12)
