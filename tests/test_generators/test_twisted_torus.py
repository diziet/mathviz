"""Tests for the twisted torus generator.

Covers standard torus equivalence (twist=0), non-orientability (twist=1),
distinct geometries for different twists, registration, and rendering.
"""

import numpy as np
import pytest
import trimesh

from mathviz.core.generator import clear_registry, get_generator, register
from mathviz.core.representation import RepresentationType
from mathviz.generators.parametric.twisted_torus import TwistedTorusGenerator


@pytest.fixture(autouse=True)
def _clean_registry():
    """Reset registry and register the twisted torus generator for each test."""
    clear_registry(suppress_discovery=True)
    register(TwistedTorusGenerator)
    yield
    clear_registry(suppress_discovery=True)


@pytest.fixture()
def default_obj():
    """Generate twisted torus with default parameters at low resolution."""
    gen = TwistedTorusGenerator()
    return gen.generate(grid_resolution=32)


@pytest.fixture()
def twist_zero_obj():
    """Generate twist=0 torus (standard torus) at low resolution."""
    gen = TwistedTorusGenerator()
    return gen.generate(params={"twist": 0}, grid_resolution=32)


@pytest.fixture()
def twist_one_obj():
    """Generate twist=1 torus (Möbius-torus) at low resolution."""
    gen = TwistedTorusGenerator()
    return gen.generate(params={"twist": 1}, grid_resolution=32)


# ---------------------------------------------------------------------------
# twist=0 produces a standard torus
# ---------------------------------------------------------------------------


def test_twist_zero_produces_valid_mesh(twist_zero_obj) -> None:
    """twist=0 produces a valid, non-empty mesh."""
    twist_zero_obj.validate_or_raise()
    assert twist_zero_obj.mesh is not None
    assert len(twist_zero_obj.mesh.vertices) > 0
    assert len(twist_zero_obj.mesh.faces) > 0


def test_twist_zero_is_orientable(twist_zero_obj) -> None:
    """twist=0 surface is orientable (standard torus)."""
    tm = trimesh.Trimesh(
        vertices=twist_zero_obj.mesh.vertices,
        faces=twist_zero_obj.mesh.faces,
        process=False,
    )
    assert tm.is_winding_consistent


def test_twist_zero_geometry_matches_standard_torus() -> None:
    """twist=0 vertices match a standard torus parameterization."""
    gen = TwistedTorusGenerator()
    obj = gen.generate(
        params={"twist": 0, "major_radius": 1.0, "minor_radius": 0.3},
        grid_resolution=16,
    )
    verts = obj.mesh.vertices

    # For a standard torus, z range should be [-minor_radius, minor_radius]
    z_range = verts[:, 2].max() - verts[:, 2].min()
    assert abs(z_range - 2 * 0.3) < 0.01


# ---------------------------------------------------------------------------
# twist=1 produces a non-orientable surface
# ---------------------------------------------------------------------------


def test_twist_one_produces_valid_mesh(twist_one_obj) -> None:
    """twist=1 produces a valid, non-empty mesh."""
    twist_one_obj.validate_or_raise()
    assert twist_one_obj.mesh is not None
    assert len(twist_one_obj.mesh.vertices) > 0


def test_twist_one_is_non_orientable(twist_one_obj) -> None:
    """twist=1 surface is non-orientable (Möbius-torus)."""
    tm = trimesh.Trimesh(
        vertices=twist_one_obj.mesh.vertices,
        faces=twist_one_obj.mesh.faces,
        process=False,
    )
    assert not tm.is_winding_consistent


# ---------------------------------------------------------------------------
# Different twist values produce distinct geometries
# ---------------------------------------------------------------------------


def test_different_twists_produce_distinct_geometry() -> None:
    """Different twist values produce different vertex sets."""
    gen = TwistedTorusGenerator()
    obj_0 = gen.generate(params={"twist": 0}, grid_resolution=16)
    obj_1 = gen.generate(params={"twist": 1}, grid_resolution=16)
    obj_3 = gen.generate(params={"twist": 3}, grid_resolution=16)

    # Vertices should differ for different twist values
    assert not np.allclose(obj_0.mesh.vertices, obj_1.mesh.vertices)
    assert not np.allclose(obj_0.mesh.vertices, obj_3.mesh.vertices)
    assert not np.allclose(obj_1.mesh.vertices, obj_3.mesh.vertices)


def test_even_twist_is_orientable() -> None:
    """Even twist values produce orientable surfaces."""
    gen = TwistedTorusGenerator()
    obj = gen.generate(params={"twist": 2}, grid_resolution=32)
    tm = trimesh.Trimesh(
        vertices=obj.mesh.vertices,
        faces=obj.mesh.faces,
        process=False,
    )
    assert tm.is_winding_consistent


def test_odd_twist_is_non_orientable() -> None:
    """Odd twist values produce non-orientable surfaces."""
    gen = TwistedTorusGenerator()
    for twist in (1, 3, 5):
        obj = gen.generate(params={"twist": twist}, grid_resolution=32)
        tm = trimesh.Trimesh(
            vertices=obj.mesh.vertices,
            faces=obj.mesh.faces,
            process=False,
        )
        assert not tm.is_winding_consistent, (
            f"twist={twist} should be non-orientable"
        )


# ---------------------------------------------------------------------------
# Registers and renders successfully
# ---------------------------------------------------------------------------


def test_registered_and_discoverable() -> None:
    """Generator is discoverable via the registry by name."""
    found = get_generator("twisted_torus")
    assert found is TwistedTorusGenerator


def test_default_representation_surface_shell() -> None:
    """Default representation is SURFACE_SHELL."""
    gen = TwistedTorusGenerator()
    rep = gen.get_default_representation()
    assert rep.type == RepresentationType.SURFACE_SHELL


def test_metadata_recorded(default_obj) -> None:
    """Generator name, category, and parameters are recorded."""
    assert default_obj.generator_name == "twisted_torus"
    assert default_obj.category == "parametric"
    assert isinstance(default_obj.parameters, dict)
    assert "twist" in default_obj.parameters
    assert "major_radius" in default_obj.parameters
    assert "minor_radius" in default_obj.parameters


def test_seed_recorded() -> None:
    """Seed is recorded in MathObject."""
    gen = TwistedTorusGenerator()
    obj = gen.generate(seed=777, grid_resolution=16)
    assert obj.seed == 777


def test_determinism() -> None:
    """Same seed and params produce identical geometry."""
    gen = TwistedTorusGenerator()
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


def test_negative_major_radius_raises() -> None:
    """Negative major_radius raises ValueError."""
    gen = TwistedTorusGenerator()
    with pytest.raises(ValueError, match="major_radius"):
        gen.generate(params={"major_radius": -1.0})


def test_negative_minor_radius_raises() -> None:
    """Negative minor_radius raises ValueError."""
    gen = TwistedTorusGenerator()
    with pytest.raises(ValueError, match="minor_radius"):
        gen.generate(params={"minor_radius": -0.1})


def test_negative_twist_raises() -> None:
    """Negative twist raises ValueError."""
    gen = TwistedTorusGenerator()
    with pytest.raises(ValueError, match="twist"):
        gen.generate(params={"twist": -1})


def test_grid_resolution_below_minimum_raises() -> None:
    """Too low grid resolution raises ValueError."""
    gen = TwistedTorusGenerator()
    with pytest.raises(ValueError, match="grid_resolution"):
        gen.generate(grid_resolution=3)


def test_odd_grid_resolution_with_odd_twist_raises() -> None:
    """Odd grid resolution with odd twist raises ValueError."""
    gen = TwistedTorusGenerator()
    with pytest.raises(ValueError, match="grid_resolution must be even"):
        gen.generate(params={"twist": 1}, grid_resolution=33)
