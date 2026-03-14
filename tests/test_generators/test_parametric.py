"""Tests for parametric surface generators, starting with the torus."""

from pathlib import Path

import numpy as np
import pytest
import trimesh

from mathviz.core.container import Container, PlacementPolicy
from mathviz.core.generator import clear_registry, get_generator, register
from mathviz.core.representation import RepresentationConfig, RepresentationType
from mathviz.generators.parametric.torus import TorusGenerator
from mathviz.pipeline.runner import ExportConfig, run


@pytest.fixture(autouse=True)
def _clean_registry():
    """Reset registry and re-register torus for each test."""
    clear_registry(suppress_discovery=True)
    register(TorusGenerator)
    yield
    clear_registry(suppress_discovery=True)


@pytest.fixture
def torus() -> TorusGenerator:
    """Return a TorusGenerator instance."""
    return TorusGenerator()


# ---------------------------------------------------------------------------
# Default torus produces a watertight mesh
# ---------------------------------------------------------------------------


def test_default_torus_watertight(torus: TorusGenerator) -> None:
    """Default torus produces a watertight mesh."""
    obj = torus.generate()
    obj.validate_or_raise()

    assert obj.mesh is not None
    tri_mesh = trimesh.Trimesh(
        vertices=obj.mesh.vertices,
        faces=obj.mesh.faces,
        process=False,
    )
    assert tri_mesh.is_watertight


# ---------------------------------------------------------------------------
# Mesh face count is consistent with grid_resolution
# ---------------------------------------------------------------------------


def test_face_count_matches_grid_resolution(torus: TorusGenerator) -> None:
    """Mesh face count equals 2 * N^2 for an N x N grid."""
    for resolution in (16, 32, 64):
        obj = torus.generate(grid_resolution=resolution)
        expected_faces = 2 * resolution * resolution
        assert obj.mesh is not None
        assert len(obj.mesh.faces) == expected_faces, (
            f"Expected {expected_faces} faces for resolution={resolution}, "
            f"got {len(obj.mesh.faces)}"
        )


def test_vertex_count_matches_grid_resolution(torus: TorusGenerator) -> None:
    """Vertex count equals N^2 for an N x N grid (no duplicated boundary)."""
    resolution = 32
    obj = torus.generate(grid_resolution=resolution)
    assert obj.mesh is not None
    assert len(obj.mesh.vertices) == resolution * resolution


# ---------------------------------------------------------------------------
# Bounding box is roughly ±(R+r) in x/y and ±r in z
# ---------------------------------------------------------------------------


def test_bounding_box_extents(torus: TorusGenerator) -> None:
    """Bounding box matches expected torus geometry."""
    major_r = 1.0
    minor_r = 0.4
    obj = torus.generate()

    assert obj.bounding_box is not None
    min_c = obj.bounding_box.min_corner
    max_c = obj.bounding_box.max_corner

    xy_extent = major_r + minor_r
    z_extent = minor_r

    assert min_c[0] == pytest.approx(-xy_extent)
    assert max_c[0] == pytest.approx(xy_extent)
    assert min_c[1] == pytest.approx(-xy_extent)
    assert max_c[1] == pytest.approx(xy_extent)
    assert min_c[2] == pytest.approx(-z_extent)
    assert max_c[2] == pytest.approx(z_extent)


def test_actual_vertices_within_bounding_box(torus: TorusGenerator) -> None:
    """All mesh vertices lie within the declared bounding box."""
    obj = torus.generate()
    assert obj.mesh is not None
    assert obj.bounding_box is not None

    verts = obj.mesh.vertices
    min_c = np.array(obj.bounding_box.min_corner)
    max_c = np.array(obj.bounding_box.max_corner)

    tolerance = 1e-10
    assert np.all(verts >= min_c - tolerance)
    assert np.all(verts <= max_c + tolerance)


# ---------------------------------------------------------------------------
# Determinism: same seed produces identical output
# ---------------------------------------------------------------------------


def test_determinism_with_seed(torus: TorusGenerator) -> None:
    """Same seed produces identical geometry (torus is deterministic)."""
    obj1 = torus.generate(seed=123)
    obj2 = torus.generate(seed=123)

    assert obj1.mesh is not None and obj2.mesh is not None
    np.testing.assert_array_equal(obj1.mesh.vertices, obj2.mesh.vertices)
    np.testing.assert_array_equal(obj1.mesh.faces, obj2.mesh.faces)
    assert obj1.seed == 123
    assert obj2.seed == 123


def test_seed_recorded_in_math_object(torus: TorusGenerator) -> None:
    """Seed is recorded in the MathObject even for deterministic generators."""
    obj = torus.generate(seed=999)
    assert obj.seed == 999


# ---------------------------------------------------------------------------
# Full pipeline: generate → transform → export STL → reimport → valid mesh
# ---------------------------------------------------------------------------


def test_full_pipeline_stl_roundtrip(tmp_path: Path) -> None:
    """Full pipeline produces a valid STL that can be reimported."""
    out_path = tmp_path / "torus.stl"

    result = run(
        "torus",
        container=Container.with_uniform_margin(),
        placement=PlacementPolicy(),
        representation_config=RepresentationConfig(
            type=RepresentationType.SURFACE_SHELL,
        ),
        export_config=ExportConfig(path=out_path, export_type="mesh"),
    )

    assert result.export_path == out_path
    assert out_path.exists()
    assert out_path.stat().st_size > 0

    # Reimport and validate
    reimported = trimesh.load(str(out_path), file_type="stl")
    assert reimported.is_watertight
    assert len(reimported.faces) > 0
    assert len(reimported.vertices) > 0


# ---------------------------------------------------------------------------
# Low resolution (grid_resolution=8) still produces valid geometry
# ---------------------------------------------------------------------------


def test_low_resolution_valid(torus: TorusGenerator) -> None:
    """Low resolution grid still produces valid geometry."""
    obj = torus.generate(grid_resolution=8)
    obj.validate_or_raise()

    assert obj.mesh is not None
    assert len(obj.mesh.faces) == 2 * 8 * 8
    assert len(obj.mesh.vertices) == 8 * 8

    tri_mesh = trimesh.Trimesh(
        vertices=obj.mesh.vertices,
        faces=obj.mesh.faces,
        process=False,
    )
    assert tri_mesh.is_watertight


# ---------------------------------------------------------------------------
# Registry: torus is discoverable
# ---------------------------------------------------------------------------


def test_torus_registered_and_discoverable() -> None:
    """Torus generator is discoverable via the registry."""
    gen_cls = get_generator("torus")
    assert gen_cls is TorusGenerator


def test_default_representation_is_surface_shell(
    torus: TorusGenerator,
) -> None:
    """Default representation for torus is SURFACE_SHELL."""
    rep = torus.get_default_representation()
    assert rep.type == RepresentationType.SURFACE_SHELL


# ---------------------------------------------------------------------------
# Custom radii
# ---------------------------------------------------------------------------


def test_custom_radii(torus: TorusGenerator) -> None:
    """Custom major/minor radii produce correctly sized geometry."""
    obj = torus.generate(
        params={"major_radius": 2.0, "minor_radius": 0.5},
        grid_resolution=16,
    )
    assert obj.mesh is not None
    assert obj.bounding_box is not None

    xy_extent = 2.0 + 0.5
    z_extent = 0.5
    assert obj.bounding_box.max_corner[0] == pytest.approx(xy_extent)
    assert obj.bounding_box.max_corner[2] == pytest.approx(z_extent)

    # Verify actual vertex extent matches
    verts = obj.mesh.vertices
    assert np.max(np.abs(verts[:, 0])) == pytest.approx(xy_extent, rel=0.01)
    assert np.max(np.abs(verts[:, 2])) == pytest.approx(z_extent, rel=0.01)


def test_metadata_recorded(torus: TorusGenerator) -> None:
    """Generator name, category, and parameters are recorded."""
    obj = torus.generate()
    assert obj.generator_name == "torus"
    assert obj.category == "parametric"
    assert obj.parameters["major_radius"] == 1.0
    assert obj.parameters["minor_radius"] == 0.4
