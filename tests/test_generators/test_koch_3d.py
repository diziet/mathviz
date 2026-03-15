"""Tests for the Koch snowflake 3D generator."""

from pathlib import Path

import numpy as np
import pytest

from mathviz.core.container import Container, PlacementPolicy
from mathviz.core.generator import clear_registry, get_generator, register
from mathviz.core.representation import RepresentationConfig, RepresentationType
from mathviz.generators.fractals.koch_3d import Koch3DGenerator


@pytest.fixture(autouse=True)
def _clean_registry():
    """Reset registry and register Koch 3D for each test."""
    clear_registry(suppress_discovery=True)
    register(Koch3DGenerator)
    yield
    clear_registry(suppress_discovery=True)


@pytest.fixture
def gen() -> Koch3DGenerator:
    """Return a Koch3DGenerator instance."""
    return Koch3DGenerator()


# ---------------------------------------------------------------------------
# Level 0 produces an equilateral triangle
# ---------------------------------------------------------------------------


def test_level_0_produces_equilateral_triangle(gen: Koch3DGenerator) -> None:
    """Level 0 extrusion is based on an equilateral triangle (3 curve pts)."""
    obj = gen.generate(params={"level": 0, "mode": "extrude"})
    obj.validate_or_raise()
    assert obj.mesh is not None
    # Level 0: 3 curve points -> 6 vertices (top + bottom rings)
    # plus side faces and caps
    assert len(obj.mesh.vertices) == 6
    assert len(obj.mesh.faces) > 0


def test_level_0_revolve(gen: Koch3DGenerator) -> None:
    """Level 0 revolve is based on an equilateral triangle."""
    obj = gen.generate(params={"level": 0, "mode": "revolve"})
    obj.validate_or_raise()
    assert obj.mesh is not None
    # Level 0: 3 curve points * 64 revolution segments = 192 vertices
    assert len(obj.mesh.vertices) == 3 * 64


# ---------------------------------------------------------------------------
# Higher levels produce more vertices (4^n scaling)
# ---------------------------------------------------------------------------


def test_vertex_count_scales_with_level(gen: Koch3DGenerator) -> None:
    """Each level quadruples the number of curve segments."""
    counts = []
    for level in range(4):
        obj = gen.generate(params={"level": level, "mode": "extrude"})
        counts.append(len(obj.mesh.vertices))

    # Level 0: 3 pts -> 6 verts (extrude doubles for top/bottom)
    # Level n: 3 * 4^n pts -> 6 * 4^n verts
    for i in range(1, len(counts)):
        assert counts[i] == counts[i - 1] * 4


def test_curve_point_count_formula(gen: Koch3DGenerator) -> None:
    """Koch snowflake at level n has 3 * 4^n curve points."""
    for level in range(5):
        obj = gen.generate(params={"level": level, "mode": "extrude"})
        expected_curve_pts = 3 * (4 ** level)
        # Extrusion has 2x curve points (top + bottom)
        assert len(obj.mesh.vertices) == 2 * expected_curve_pts


# ---------------------------------------------------------------------------
# Extrude and revolve modes produce distinct geometries
# ---------------------------------------------------------------------------


def test_extrude_and_revolve_produce_distinct_geometry(
    gen: Koch3DGenerator,
) -> None:
    """Extrude and revolve modes produce meshes with different vertex counts."""
    obj_extrude = gen.generate(params={"level": 2, "mode": "extrude"})
    obj_revolve = gen.generate(params={"level": 2, "mode": "revolve"})

    assert obj_extrude.mesh is not None
    assert obj_revolve.mesh is not None

    # Different number of vertices
    assert len(obj_extrude.mesh.vertices) != len(obj_revolve.mesh.vertices)

    # Different bounding boxes
    ext_bbox = obj_extrude.bounding_box
    rev_bbox = obj_revolve.bounding_box
    assert ext_bbox is not None
    assert rev_bbox is not None
    # Revolved geometry has rotational symmetry, so z-extent should be
    # comparable to x-extent; extruded geometry has small z-extent
    ext_z = ext_bbox.max_corner[2] - ext_bbox.min_corner[2]
    rev_z = rev_bbox.max_corner[2] - rev_bbox.min_corner[2]
    assert rev_z > ext_z


def test_extrude_z_extent_matches_height(gen: Koch3DGenerator) -> None:
    """Extruded geometry z-range equals the height parameter."""
    height = 0.5
    obj = gen.generate(params={"level": 1, "mode": "extrude", "height": height})
    bbox = obj.bounding_box
    assert bbox is not None
    z_range = bbox.max_corner[2] - bbox.min_corner[2]
    np.testing.assert_allclose(z_range, height, atol=1e-10)


# ---------------------------------------------------------------------------
# Registers and renders successfully
# ---------------------------------------------------------------------------


def test_registered() -> None:
    """koch_3d is discoverable via the registry."""
    gen_cls = get_generator("koch_3d")
    assert gen_cls is Koch3DGenerator


def test_alias_registered() -> None:
    """Alias is discoverable via the registry."""
    assert get_generator("koch_snowflake_3d") is Koch3DGenerator


def test_full_pipeline_renders(tmp_path: Path) -> None:
    """Full pipeline with SURFACE_SHELL produces a valid STL."""
    from mathviz.pipeline.runner import ExportConfig, run

    out_path = tmp_path / "koch_3d.stl"
    result = run(
        "koch_3d",
        params={"level": 2},
        container=Container.with_uniform_margin(),
        placement=PlacementPolicy(),
        representation_config=RepresentationConfig(
            type=RepresentationType.SURFACE_SHELL,
        ),
        export_config=ExportConfig(
            path=out_path, export_type="mesh",
        ),
    )
    assert result.export_path == out_path
    assert out_path.exists()
    assert out_path.stat().st_size > 0


# ---------------------------------------------------------------------------
# Default representation
# ---------------------------------------------------------------------------


def test_default_representation(gen: Koch3DGenerator) -> None:
    """Default representation is SURFACE_SHELL."""
    rep = gen.get_default_representation()
    assert rep.type == RepresentationType.SURFACE_SHELL


# ---------------------------------------------------------------------------
# Metadata
# ---------------------------------------------------------------------------


def test_metadata_recorded(gen: Koch3DGenerator) -> None:
    """Generator name, category, and parameters are recorded."""
    obj = gen.generate(params={"level": 2})
    assert obj.generator_name == "koch_3d"
    assert obj.category == "fractals"
    assert obj.parameters["level"] == 2
    assert obj.parameters["mode"] == "extrude"
    assert obj.parameters["height"] == 0.3


# ---------------------------------------------------------------------------
# Determinism
# ---------------------------------------------------------------------------


def test_determinism(gen: Koch3DGenerator) -> None:
    """Same params produce identical output (deterministic, no seed dep)."""
    obj1 = gen.generate(params={"level": 3})
    obj2 = gen.generate(params={"level": 3})
    np.testing.assert_array_equal(obj1.mesh.vertices, obj2.mesh.vertices)
    np.testing.assert_array_equal(obj1.mesh.faces, obj2.mesh.faces)


# ---------------------------------------------------------------------------
# Input validation
# ---------------------------------------------------------------------------


def test_negative_level_raises(gen: Koch3DGenerator) -> None:
    """Negative level raises ValueError."""
    with pytest.raises(ValueError, match="level must be >= 0"):
        gen.generate(params={"level": -1})


def test_level_exceeds_max_raises(gen: Koch3DGenerator) -> None:
    """Level above max raises ValueError."""
    with pytest.raises(ValueError, match="level must be <= 6"):
        gen.generate(params={"level": 7})


def test_invalid_mode_raises(gen: Koch3DGenerator) -> None:
    """Invalid mode raises ValueError."""
    with pytest.raises(ValueError, match="mode must be one of"):
        gen.generate(params={"mode": "twist"})


def test_tiny_height_raises(gen: Koch3DGenerator) -> None:
    """Near-zero height raises ValueError."""
    with pytest.raises(ValueError, match="height must be"):
        gen.generate(params={"height": 0.0})


# ---------------------------------------------------------------------------
# No NaN in output
# ---------------------------------------------------------------------------


def test_no_nan_in_extrude(gen: Koch3DGenerator) -> None:
    """Extruded output contains no NaN values."""
    obj = gen.generate(params={"level": 3, "mode": "extrude"})
    assert not np.any(np.isnan(obj.mesh.vertices))


def test_no_nan_in_revolve(gen: Koch3DGenerator) -> None:
    """Revolved output contains no NaN values."""
    obj = gen.generate(params={"level": 3, "mode": "revolve"})
    assert not np.any(np.isnan(obj.mesh.vertices))
