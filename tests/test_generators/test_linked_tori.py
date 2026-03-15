"""Tests for the linked tori generator.

Covers mesh validity, interlocking geometry, registry integration, and rendering.
"""

import numpy as np
import pytest
import trimesh

from mathviz.core.generator import clear_registry, get_generator, register
from mathviz.core.representation import RepresentationType
from mathviz.generators.parametric.linked_tori import LinkedToriGenerator


def _split_into_components(math_obj):
    """Split a MathObject mesh into connected components."""
    tm = trimesh.Trimesh(
        vertices=math_obj.mesh.vertices,
        faces=math_obj.mesh.faces,
        process=False,
    )
    return tm.split(only_watertight=False)


@pytest.fixture(autouse=True)
def _clean_registry():
    """Reset registry and register the linked tori generator for each test."""
    clear_registry(suppress_discovery=True)
    register(LinkedToriGenerator)
    yield
    clear_registry(suppress_discovery=True)


@pytest.fixture()
def default_obj():
    """Generate linked tori with default parameters at low resolution."""
    gen = LinkedToriGenerator()
    return gen.generate(grid_resolution=16)


@pytest.fixture()
def three_tori_obj():
    """Generate three linked tori at low resolution."""
    gen = LinkedToriGenerator()
    return gen.generate(params={"num_tori": 3}, grid_resolution=16)


# ---------------------------------------------------------------------------
# 2 linked tori produce two distinct mesh components
# ---------------------------------------------------------------------------


def test_default_produces_valid_mesh(default_obj) -> None:
    """Default parameters produce a valid, non-empty mesh."""
    default_obj.validate_or_raise()
    assert default_obj.mesh is not None
    assert len(default_obj.mesh.vertices) > 0
    assert len(default_obj.mesh.faces) > 0
    assert default_obj.bounding_box is not None


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


def test_two_tori_have_two_components(default_obj) -> None:
    """Two linked tori produce exactly two distinct mesh components."""
    components = _split_into_components(default_obj)
    assert len(components) == 2


def test_three_tori_have_three_components(three_tori_obj) -> None:
    """Three linked tori produce exactly three distinct mesh components."""
    components = _split_into_components(three_tori_obj)
    assert len(components) == 3


def test_vertex_count_scales_with_num_tori() -> None:
    """Vertex count scales linearly with number of tori."""
    gen = LinkedToriGenerator()
    obj2 = gen.generate(params={"num_tori": 2}, grid_resolution=16)
    obj4 = gen.generate(params={"num_tori": 4}, grid_resolution=16)
    assert len(obj4.mesh.vertices) == 2 * len(obj2.mesh.vertices)


# ---------------------------------------------------------------------------
# Tori geometrically interlock (bounding boxes overlap)
# ---------------------------------------------------------------------------


def test_component_bounding_boxes_overlap(default_obj) -> None:
    """Adjacent tori bounding boxes overlap, confirming interlocking."""
    components = _split_into_components(default_obj)
    assert len(components) == 2

    bbox_a = components[0].bounds  # (2, 3) array: [min, max]
    bbox_b = components[1].bounds

    # Check overlap: for each axis, min of one < max of the other
    has_overlap = True
    for axis in range(3):
        if bbox_a[1, axis] < bbox_b[0, axis] or bbox_b[1, axis] < bbox_a[0, axis]:
            has_overlap = False
            break

    assert has_overlap, "Linked tori bounding boxes should overlap"


def test_tori_centers_are_spaced_along_x() -> None:
    """Tori centers are offset along the X axis by link_spacing."""
    gen = LinkedToriGenerator()
    spacing = 2.0
    obj = gen.generate(
        params={"num_tori": 3, "link_spacing": spacing},
        grid_resolution=16,
    )
    components = _split_into_components(obj)
    centers = sorted([c.centroid[0] for c in components])

    for i in range(len(centers) - 1):
        assert abs(centers[i + 1] - centers[i] - spacing) < 0.2


# ---------------------------------------------------------------------------
# Registers and renders successfully
# ---------------------------------------------------------------------------


def test_registered_and_discoverable() -> None:
    """Generator is discoverable via the registry by name."""
    found = get_generator("linked_tori")
    assert found is LinkedToriGenerator


def test_default_representation_surface_shell() -> None:
    """Default representation is SURFACE_SHELL."""
    gen = LinkedToriGenerator()
    rep = gen.get_default_representation()
    assert rep.type == RepresentationType.SURFACE_SHELL


def test_metadata_recorded(default_obj) -> None:
    """Generator name, category, and parameters are recorded."""
    assert default_obj.generator_name == "linked_tori"
    assert default_obj.category == "parametric"
    assert isinstance(default_obj.parameters, dict)
    assert "num_tori" in default_obj.parameters
    assert "major_radius" in default_obj.parameters


def test_seed_recorded() -> None:
    """Seed is recorded in MathObject."""
    gen = LinkedToriGenerator()
    obj = gen.generate(seed=777, grid_resolution=16)
    assert obj.seed == 777


def test_determinism() -> None:
    """Same seed and params produce identical geometry."""
    gen = LinkedToriGenerator()
    obj1 = gen.generate(seed=42, grid_resolution=16)
    obj2 = gen.generate(seed=42, grid_resolution=16)
    np.testing.assert_array_equal(obj1.mesh.vertices, obj2.mesh.vertices)
    np.testing.assert_array_equal(obj1.mesh.faces, obj2.mesh.faces)


# ---------------------------------------------------------------------------
# Parameter validation
# ---------------------------------------------------------------------------


def test_num_tori_below_minimum_raises() -> None:
    """num_tori below minimum raises ValueError."""
    gen = LinkedToriGenerator()
    with pytest.raises(ValueError, match="num_tori"):
        gen.generate(params={"num_tori": 1})


def test_num_tori_above_maximum_raises() -> None:
    """num_tori above maximum raises ValueError."""
    gen = LinkedToriGenerator()
    with pytest.raises(ValueError, match="num_tori"):
        gen.generate(params={"num_tori": 25})


def test_negative_major_radius_raises() -> None:
    """Negative major_radius raises ValueError."""
    gen = LinkedToriGenerator()
    with pytest.raises(ValueError, match="major_radius"):
        gen.generate(params={"major_radius": -1.0})


def test_negative_minor_radius_raises() -> None:
    """Negative minor_radius raises ValueError."""
    gen = LinkedToriGenerator()
    with pytest.raises(ValueError, match="minor_radius"):
        gen.generate(params={"minor_radius": -0.1})


def test_negative_link_spacing_raises() -> None:
    """Negative link_spacing raises ValueError."""
    gen = LinkedToriGenerator()
    with pytest.raises(ValueError, match="link_spacing"):
        gen.generate(params={"link_spacing": -1.0})


def test_grid_resolution_below_minimum_raises() -> None:
    """Too low grid resolution raises ValueError."""
    gen = LinkedToriGenerator()
    with pytest.raises(ValueError, match="grid_resolution"):
        gen.generate(grid_resolution=2)


# ---------------------------------------------------------------------------
# Custom parameters
# ---------------------------------------------------------------------------


def test_custom_radii_change_geometry() -> None:
    """Different radii produce different geometry extents."""
    gen = LinkedToriGenerator()
    obj_small = gen.generate(
        params={"major_radius": 0.5, "minor_radius": 0.1},
        grid_resolution=16,
    )
    obj_large = gen.generate(
        params={"major_radius": 2.0, "minor_radius": 0.5},
        grid_resolution=16,
    )
    small_extent = np.ptp(obj_small.mesh.vertices, axis=0)
    large_extent = np.ptp(obj_large.mesh.vertices, axis=0)
    assert np.all(large_extent > small_extent)
