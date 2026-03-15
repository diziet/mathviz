"""Tests for the involute gear generator."""

import numpy as np
import pytest

from mathviz.core.generator import clear_registry, get_generator, register
from mathviz.core.representation import RepresentationType
from mathviz.generators.geometry.gear import GearGenerator


@pytest.fixture(autouse=True)
def _clean_registry():
    """Reset registry for each test."""
    clear_registry(suppress_discovery=True)
    register(GearGenerator)
    yield
    clear_registry(suppress_discovery=True)


# ---------------------------------------------------------------------------
# Valid mesh with expected number of teeth
# ---------------------------------------------------------------------------


def test_produces_valid_mesh_default_params() -> None:
    """Default parameters produce a valid mesh."""
    gen = GearGenerator()
    obj = gen.generate()
    obj.validate_or_raise()

    assert obj.mesh is not None
    assert len(obj.mesh.vertices) > 0
    assert len(obj.mesh.faces) > 0
    assert obj.generator_name == "gear"
    assert obj.category == "geometry"
    assert obj.bounding_box is not None


def test_mesh_extent_matches_tooth_count() -> None:
    """Gear with more teeth produces more vertices and a larger mesh."""
    gen = GearGenerator()
    obj_small = gen.generate(params={"num_teeth": 8, "module": 1.0})
    obj_large = gen.generate(params={"num_teeth": 30, "module": 1.0})
    obj_small.validate_or_raise()
    obj_large.validate_or_raise()

    assert len(obj_large.mesh.vertices) > len(obj_small.mesh.vertices)


def test_gear_radius_scales_with_module() -> None:
    """Larger module produces a larger gear (larger bounding box)."""
    gen = GearGenerator()
    obj_small = gen.generate(params={"num_teeth": 12, "module": 0.5})
    obj_large = gen.generate(params={"num_teeth": 12, "module": 2.0})
    obj_small.validate_or_raise()
    obj_large.validate_or_raise()

    small_extent = max(obj_small.bounding_box.size[:2])
    large_extent = max(obj_large.bounding_box.size[:2])
    assert large_extent > small_extent * 3.0


def test_vertex_count_scales_with_teeth() -> None:
    """Vertex count scales linearly with number of teeth."""
    gen = GearGenerator()
    obj_10 = gen.generate(params={"num_teeth": 10, "curve_points": 16})
    obj_20 = gen.generate(params={"num_teeth": 20, "curve_points": 16})
    obj_10.validate_or_raise()
    obj_20.validate_or_raise()

    # Expect roughly 2x vertices (minus the 2 center caps)
    v10 = len(obj_10.mesh.vertices) - 2
    v20 = len(obj_20.mesh.vertices) - 2
    ratio = v20 / v10
    assert 1.8 < ratio < 2.2


# ---------------------------------------------------------------------------
# Spur gear (helix_angle=0) produces straight teeth
# ---------------------------------------------------------------------------


def test_spur_gear_straight_extrusion() -> None:
    """helix_angle=0 produces a straight spur gear (no twist between layers)."""
    gen = GearGenerator()
    obj = gen.generate(params={"num_teeth": 12, "helix_angle": 0})
    obj.validate_or_raise()

    verts = obj.mesh.vertices
    # Exclude the 2 center vertices (last two)
    profile_verts = verts[:-2]

    # For a spur gear with 2 layers, bottom (z=0) and top (z=face_width)
    bottom = profile_verts[profile_verts[:, 2] < 0.01]
    top = profile_verts[profile_verts[:, 2] > 0.01]

    # Bottom and top profiles should have identical xy coordinates
    assert len(bottom) == len(top)
    np.testing.assert_allclose(bottom[:, :2], top[:, :2], atol=1e-10)


def test_spur_gear_has_two_z_layers() -> None:
    """Spur gear extrusion uses exactly 2 layers."""
    gen = GearGenerator()
    obj = gen.generate(params={"num_teeth": 10, "helix_angle": 0})
    obj.validate_or_raise()

    profile_verts = obj.mesh.vertices[:-2]
    unique_z = np.unique(np.round(profile_verts[:, 2], decimals=10))
    assert len(unique_z) == 2


# ---------------------------------------------------------------------------
# Helical gear (helix_angle > 0) produces twisted teeth
# ---------------------------------------------------------------------------


def test_helical_gear_twisted_teeth() -> None:
    """helix_angle > 0 produces twisted teeth (top rotated vs bottom)."""
    gen = GearGenerator()
    obj = gen.generate(params={"num_teeth": 12, "helix_angle": 15})
    obj.validate_or_raise()

    verts = obj.mesh.vertices[:-2]
    z_vals = np.unique(np.round(verts[:, 2], decimals=8))

    # Should have more than 2 z-layers for helical gear
    assert len(z_vals) >= 2

    # Bottom and top profiles should differ in xy (twisted)
    num_per_layer = len(verts) // len(z_vals)
    bottom = verts[:num_per_layer]
    top = verts[-num_per_layer:]

    # At least some xy coordinates must differ significantly
    xy_diff = np.linalg.norm(bottom[:, :2] - top[:, :2], axis=1)
    assert np.max(xy_diff) > 1e-4, "Top and bottom should be twisted"


def test_helical_gear_multiple_layers() -> None:
    """Helical gear with large twist uses more intermediate layers."""
    gen = GearGenerator()
    # Large face_width + helix_angle to ensure significant twist
    obj = gen.generate(
        params={"num_teeth": 8, "helix_angle": 30, "face_width": 5.0}
    )
    obj.validate_or_raise()

    verts = obj.mesh.vertices[:-2]
    unique_z = np.unique(np.round(verts[:, 2], decimals=8))
    assert len(unique_z) > 2, "Helical gear should use multiple layers"


# ---------------------------------------------------------------------------
# Registration and rendering
# ---------------------------------------------------------------------------


def test_registers_and_renders() -> None:
    """Generator registers and can produce valid output via registry lookup."""
    gen_cls = get_generator("gear")
    assert gen_cls is GearGenerator

    gen = gen_cls()
    obj = gen.generate()
    obj.validate_or_raise()

    assert obj.mesh is not None
    assert obj.generator_name == "gear"
    assert obj.category == "geometry"
    assert obj.bounding_box is not None


def test_default_representation_is_surface_shell() -> None:
    """Default representation is SURFACE_SHELL."""
    gen = GearGenerator()
    rep = gen.get_default_representation()
    assert rep.type == RepresentationType.SURFACE_SHELL


# ---------------------------------------------------------------------------
# Determinism
# ---------------------------------------------------------------------------


def test_deterministic_output() -> None:
    """Same parameters produce identical output."""
    gen = GearGenerator()
    obj1 = gen.generate(params={"num_teeth": 15})
    obj2 = gen.generate(params={"num_teeth": 15})

    assert obj1.mesh is not None
    assert obj2.mesh is not None
    np.testing.assert_array_equal(obj1.mesh.vertices, obj2.mesh.vertices)
    np.testing.assert_array_equal(obj1.mesh.faces, obj2.mesh.faces)


# ---------------------------------------------------------------------------
# Parameter validation
# ---------------------------------------------------------------------------


def test_too_few_teeth_rejected() -> None:
    """num_teeth below minimum raises ValueError."""
    gen = GearGenerator()
    with pytest.raises(ValueError, match="num_teeth must be >= 6"):
        gen.generate(params={"num_teeth": 3})


def test_too_many_teeth_rejected() -> None:
    """num_teeth above maximum raises ValueError."""
    gen = GearGenerator()
    with pytest.raises(ValueError, match="num_teeth must be <= 200"):
        gen.generate(params={"num_teeth": 300})


def test_negative_module_rejected() -> None:
    """Negative module raises ValueError."""
    gen = GearGenerator()
    with pytest.raises(ValueError, match="module must be > 0"):
        gen.generate(params={"module": -1.0})


def test_invalid_pressure_angle_rejected() -> None:
    """Pressure angle out of range raises ValueError."""
    gen = GearGenerator()
    with pytest.raises(ValueError, match="pressure_angle must be in"):
        gen.generate(params={"pressure_angle": 50})


def test_negative_face_width_rejected() -> None:
    """Negative face_width raises ValueError."""
    gen = GearGenerator()
    with pytest.raises(ValueError, match="face_width must be > 0"):
        gen.generate(params={"face_width": -1.0})


def test_too_few_curve_points_rejected() -> None:
    """curve_points below minimum raises ValueError."""
    gen = GearGenerator()
    with pytest.raises(ValueError, match="curve_points must be >= 8"):
        gen.generate(params={"curve_points": 4})


# ---------------------------------------------------------------------------
# Default params
# ---------------------------------------------------------------------------


def test_default_params() -> None:
    """Default params match specification."""
    gen = GearGenerator()
    defaults = gen.get_default_params()
    assert defaults["num_teeth"] == 20
    assert defaults["module"] == 1.0
    assert defaults["pressure_angle"] == 20.0
    assert defaults["face_width"] == 0.5
    assert defaults["helix_angle"] == 0.0
    assert defaults["curve_points"] == 32


# ---------------------------------------------------------------------------
# Face width
# ---------------------------------------------------------------------------


def test_face_width_determines_z_extent() -> None:
    """Gear z-extent matches face_width parameter."""
    gen = GearGenerator()
    obj = gen.generate(params={"face_width": 2.0})
    obj.validate_or_raise()

    z_min = obj.mesh.vertices[:, 2].min()
    z_max = obj.mesh.vertices[:, 2].max()
    np.testing.assert_allclose(z_min, 0.0, atol=1e-10)
    np.testing.assert_allclose(z_max, 2.0, atol=1e-10)
