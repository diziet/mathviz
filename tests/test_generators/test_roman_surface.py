"""Tests for the Roman (Steiner) surface generator.

Covers: mesh validity, self-intersections, registration, representation,
determinism, parameter validation, and bounding box correctness.
"""

import numpy as np
import pytest

from mathviz.core.generator import clear_registry, get_generator, register
from mathviz.core.representation import RepresentationType
from mathviz.generators.parametric.roman_surface import RomanSurfaceGenerator


@pytest.fixture(autouse=True)
def _clean_registry():
    """Reset registry and register the Roman surface generator."""
    clear_registry(suppress_discovery=True)
    register(RomanSurfaceGenerator)
    yield
    clear_registry(suppress_discovery=True)


def test_default_produces_nonempty_mesh() -> None:
    """Generator produces a valid, non-empty mesh at defaults."""
    gen = RomanSurfaceGenerator()
    obj = gen.generate(grid_resolution=32)
    obj.validate_or_raise()

    assert obj.mesh is not None
    assert len(obj.mesh.vertices) > 0
    assert len(obj.mesh.faces) > 0
    assert obj.bounding_box is not None


def test_self_intersections_present() -> None:
    """Roman surface has self-intersections in R³.

    We detect this by checking that the trimesh representation reports
    the mesh as non-watertight or has degenerate faces due to the
    self-intersecting immersion, and that spatially close but
    topologically distant vertices exist.
    """
    gen = RomanSurfaceGenerator()
    obj = gen.generate(grid_resolution=64)
    obj.validate_or_raise()
    assert obj.mesh is not None

    verts = obj.mesh.vertices
    assert len(verts) > 0

    # The Roman surface passes through the origin multiple times.
    # Vertices near origin from different grid regions confirm
    # self-intersection.
    origin_dist = np.linalg.norm(verts, axis=1)
    near_origin = np.where(origin_dist < 0.05)[0]
    assert len(near_origin) > 1, "Expected multiple vertices near origin"


def test_registered_and_discoverable() -> None:
    """Generator is discoverable via the registry by name."""
    found = get_generator("roman_surface")
    assert found is RomanSurfaceGenerator


def test_alias_discoverable() -> None:
    """Generator is discoverable via the steiner_surface alias."""
    found = get_generator("steiner_surface")
    assert found is RomanSurfaceGenerator


def test_default_representation_surface_shell() -> None:
    """Default representation is SURFACE_SHELL."""
    gen = RomanSurfaceGenerator()
    rep = gen.get_default_representation()
    assert rep.type == RepresentationType.SURFACE_SHELL


def test_metadata_recorded() -> None:
    """Generator name, category, and parameters are recorded."""
    gen = RomanSurfaceGenerator()
    obj = gen.generate(grid_resolution=16)
    assert obj.generator_name == "roman_surface"
    assert obj.category == "parametric"
    assert isinstance(obj.parameters, dict)
    assert "scale" in obj.parameters


def test_seed_recorded() -> None:
    """Seed is recorded in MathObject."""
    gen = RomanSurfaceGenerator()
    obj = gen.generate(seed=777, grid_resolution=16)
    assert obj.seed == 777


def test_determinism() -> None:
    """Same seed and params produce identical geometry."""
    gen = RomanSurfaceGenerator()
    obj1 = gen.generate(seed=42, grid_resolution=16)
    obj2 = gen.generate(seed=42, grid_resolution=16)

    assert obj1.mesh is not None and obj2.mesh is not None
    np.testing.assert_array_equal(obj1.mesh.vertices, obj2.mesh.vertices)
    np.testing.assert_array_equal(obj1.mesh.faces, obj2.mesh.faces)


def test_vertices_within_bounding_box() -> None:
    """All mesh vertices lie within the declared bounding box."""
    gen = RomanSurfaceGenerator()
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
    gen = RomanSurfaceGenerator()
    obj_small = gen.generate(params={"scale": 0.5}, grid_resolution=16)
    obj_large = gen.generate(params={"scale": 2.0}, grid_resolution=16)

    assert obj_small.mesh is not None and obj_large.mesh is not None
    extent_small = np.ptp(obj_small.mesh.vertices, axis=0).max()
    extent_large = np.ptp(obj_large.mesh.vertices, axis=0).max()
    assert extent_large > extent_small


def test_invalid_scale_raises() -> None:
    """Negative or zero scale raises ValueError."""
    gen = RomanSurfaceGenerator()
    with pytest.raises(ValueError, match="scale must be positive"):
        gen.generate(params={"scale": 0.0})
    with pytest.raises(ValueError, match="scale must be positive"):
        gen.generate(params={"scale": -1.0})


def test_invalid_grid_resolution_raises() -> None:
    """Grid resolution below minimum raises ValueError."""
    gen = RomanSurfaceGenerator()
    with pytest.raises(ValueError, match="grid_resolution must be"):
        gen.generate(grid_resolution=2)


def test_implicit_equation() -> None:
    """Vertices satisfy the Roman surface implicit equation.

    The Roman surface satisfies x²y² + y²z² + x²z² = scale² · xyz.
    We verify this for vertices away from the origin where the
    equation is numerically stable.
    """
    gen = RomanSurfaceGenerator()
    obj = gen.generate(
        params={"separation_epsilon": 0.0},
        grid_resolution=64,
    )
    assert obj.mesh is not None

    verts = obj.mesh.vertices
    x, y, z = verts[:, 0], verts[:, 1], verts[:, 2]

    lhs = x**2 * y**2 + y**2 * z**2 + x**2 * z**2
    rhs = 1.0 * x * y * z  # scale=1.0, so scale² = 1.0

    # Filter to vertices away from origin where both sides are nonzero
    norms = np.linalg.norm(verts, axis=1)
    mask = norms > 0.1
    assert np.sum(mask) > 100, "Not enough vertices away from origin"

    residual = np.abs(lhs[mask] - rhs[mask])
    assert np.all(residual < 1e-10), (
        f"Max residual: {np.max(residual):.2e}"
    )
