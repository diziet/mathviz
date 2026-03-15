"""Tests for the shell spiral generator.

Covers valid mesh production, spiral structure, growth rate effects,
turn count effects, registration, and rendering.
"""

import numpy as np
import pytest

from mathviz.core.generator import clear_registry, get_generator, register
from mathviz.core.representation import RepresentationType
from mathviz.generators.parametric.shell_spiral import ShellSpiralGenerator


@pytest.fixture(autouse=True)
def _clean_registry():
    """Reset registry and register the shell spiral generator for each test."""
    clear_registry(suppress_discovery=True)
    register(ShellSpiralGenerator)
    yield
    clear_registry(suppress_discovery=True)


@pytest.fixture()
def default_obj():
    """Generate shell spiral with default parameters at low resolution."""
    gen = ShellSpiralGenerator()
    return gen.generate(curve_points=64, radial_segments=16)


# ---------------------------------------------------------------------------
# Produces a valid mesh with spiral structure
# ---------------------------------------------------------------------------


def test_produces_valid_mesh(default_obj) -> None:
    """Default parameters produce a valid, non-empty mesh."""
    default_obj.validate_or_raise()
    assert default_obj.mesh is not None
    assert len(default_obj.mesh.vertices) > 0
    assert len(default_obj.mesh.faces) > 0


def test_spiral_structure_vertices_spread_radially(default_obj) -> None:
    """Vertices spread outward from the origin — spiral structure."""
    verts = default_obj.mesh.vertices
    distances = np.sqrt(verts[:, 0] ** 2 + verts[:, 1] ** 2)
    # A spiral should have vertices at various distances
    assert distances.max() > distances.min() * 2


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
# More turns produce a longer shell
# ---------------------------------------------------------------------------


def test_more_turns_produce_longer_shell() -> None:
    """Increasing turns extends the spiral, producing more vertices."""
    gen = ShellSpiralGenerator()
    obj_2 = gen.generate(
        params={"turns": 2}, curve_points=64, radial_segments=8,
    )
    obj_5 = gen.generate(
        params={"turns": 5}, curve_points=64, radial_segments=8,
    )
    # More turns → the spiral reaches farther from origin
    dist_2 = np.max(np.linalg.norm(obj_2.mesh.vertices, axis=1))
    dist_5 = np.max(np.linalg.norm(obj_5.mesh.vertices, axis=1))
    assert dist_5 > dist_2


def test_more_turns_different_geometry() -> None:
    """Different turn counts produce distinct vertex positions."""
    gen = ShellSpiralGenerator()
    obj_a = gen.generate(
        params={"turns": 2}, curve_points=64, radial_segments=8,
    )
    obj_b = gen.generate(
        params={"turns": 4}, curve_points=64, radial_segments=8,
    )
    assert not np.allclose(obj_a.mesh.vertices, obj_b.mesh.vertices)


# ---------------------------------------------------------------------------
# Growth rate affects how quickly the spiral expands
# ---------------------------------------------------------------------------


def test_growth_rate_affects_expansion() -> None:
    """Higher growth rate produces a more rapidly expanding spiral."""
    gen = ShellSpiralGenerator()
    obj_slow = gen.generate(
        params={"growth_rate": 0.05}, curve_points=64, radial_segments=8,
    )
    obj_fast = gen.generate(
        params={"growth_rate": 0.3}, curve_points=64, radial_segments=8,
    )
    extent_slow = np.max(np.linalg.norm(obj_slow.mesh.vertices, axis=1))
    extent_fast = np.max(np.linalg.norm(obj_fast.mesh.vertices, axis=1))
    assert extent_fast > extent_slow


def test_different_growth_rates_distinct_geometry() -> None:
    """Different growth rates produce distinct vertex positions."""
    gen = ShellSpiralGenerator()
    obj_a = gen.generate(
        params={"growth_rate": 0.05}, curve_points=64, radial_segments=8,
    )
    obj_b = gen.generate(
        params={"growth_rate": 0.2}, curve_points=64, radial_segments=8,
    )
    assert not np.allclose(obj_a.mesh.vertices, obj_b.mesh.vertices)


# ---------------------------------------------------------------------------
# Registers and renders successfully
# ---------------------------------------------------------------------------


def test_registered_and_discoverable() -> None:
    """Generator is discoverable via the registry by name."""
    found = get_generator("shell_spiral")
    assert found is ShellSpiralGenerator


def test_default_representation_surface_shell() -> None:
    """Default representation is SURFACE_SHELL."""
    gen = ShellSpiralGenerator()
    rep = gen.get_default_representation()
    assert rep.type == RepresentationType.SURFACE_SHELL


def test_metadata_recorded(default_obj) -> None:
    """Generator name, category, and parameters are recorded."""
    assert default_obj.generator_name == "shell_spiral"
    assert default_obj.category == "parametric"
    assert isinstance(default_obj.parameters, dict)
    assert "growth_rate" in default_obj.parameters
    assert "turns" in default_obj.parameters
    assert "opening_rate" in default_obj.parameters
    assert "ellipticity" in default_obj.parameters


def test_seed_recorded() -> None:
    """Seed is recorded in MathObject."""
    gen = ShellSpiralGenerator()
    obj = gen.generate(seed=777, curve_points=16, radial_segments=8)
    assert obj.seed == 777


def test_determinism() -> None:
    """Same seed and params produce identical geometry."""
    gen = ShellSpiralGenerator()
    obj1 = gen.generate(seed=42, curve_points=32, radial_segments=8)
    obj2 = gen.generate(seed=42, curve_points=32, radial_segments=8)
    np.testing.assert_array_equal(obj1.mesh.vertices, obj2.mesh.vertices)
    np.testing.assert_array_equal(obj1.mesh.faces, obj2.mesh.faces)


# ---------------------------------------------------------------------------
# Ellipticity affects cross-section shape
# ---------------------------------------------------------------------------


def test_ellipticity_changes_geometry() -> None:
    """Non-unit ellipticity produces different geometry than circular."""
    gen = ShellSpiralGenerator()
    obj_round = gen.generate(
        params={"ellipticity": 1.0}, curve_points=32, radial_segments=8,
    )
    obj_flat = gen.generate(
        params={"ellipticity": 2.0}, curve_points=32, radial_segments=8,
    )
    assert not np.allclose(obj_round.mesh.vertices, obj_flat.mesh.vertices)


# ---------------------------------------------------------------------------
# Parameter validation
# ---------------------------------------------------------------------------


def test_growth_rate_below_minimum_raises() -> None:
    """growth_rate below minimum raises ValueError."""
    gen = ShellSpiralGenerator()
    with pytest.raises(ValueError, match="growth_rate"):
        gen.generate(params={"growth_rate": 0.0})


def test_turns_below_minimum_raises() -> None:
    """turns below minimum raises ValueError."""
    gen = ShellSpiralGenerator()
    with pytest.raises(ValueError, match="turns"):
        gen.generate(params={"turns": 0.1})


def test_curve_points_below_minimum_raises() -> None:
    """Too low curve_points raises ValueError."""
    gen = ShellSpiralGenerator()
    with pytest.raises(ValueError, match="curve_points"):
        gen.generate(curve_points=3)


def test_radial_segments_below_minimum_raises() -> None:
    """Too low radial_segments raises ValueError."""
    gen = ShellSpiralGenerator()
    with pytest.raises(ValueError, match="radial_segments"):
        gen.generate(radial_segments=2)
