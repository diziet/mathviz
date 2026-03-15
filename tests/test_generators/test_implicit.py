"""Tests for implicit surface generators, starting with the gyroid."""

from pathlib import Path

import numpy as np
import pytest
import trimesh

from mathviz.core.container import Container, PlacementPolicy
from mathviz.core.generator import clear_registry, get_generator, register
from mathviz.core.representation import RepresentationConfig, RepresentationType
from mathviz.generators.implicit.gyroid import GyroidGenerator
from mathviz.pipeline.runner import ExportConfig, run


@pytest.fixture(autouse=True)
def _clean_registry():
    """Reset registry and re-register gyroid for each test."""
    clear_registry(suppress_discovery=True)
    register(GyroidGenerator)
    yield
    clear_registry(suppress_discovery=True)


@pytest.fixture
def gyroid() -> GyroidGenerator:
    """Return a GyroidGenerator instance."""
    return GyroidGenerator()


# Use low resolution for test speed
_TEST_VOXEL_RESOLUTION = 32


# ---------------------------------------------------------------------------
# Default gyroid produces a non-empty manifold mesh
# ---------------------------------------------------------------------------


def test_default_gyroid_non_empty_mesh(gyroid: GyroidGenerator) -> None:
    """Default gyroid produces a non-empty valid mesh."""
    obj = gyroid.generate(voxel_resolution=_TEST_VOXEL_RESOLUTION)
    obj.validate_or_raise()

    assert obj.mesh is not None
    assert len(obj.mesh.vertices) > 0
    assert len(obj.mesh.faces) > 0

    tri_mesh = trimesh.Trimesh(
        vertices=obj.mesh.vertices,
        faces=obj.mesh.faces,
        process=False,
    )
    # The gyroid is an open surface (intersects the bounding box edges),
    # so it won't be watertight. Check that it has non-degenerate area.
    assert tri_mesh.area > 0


# ---------------------------------------------------------------------------
# Increasing periods increases face count
# ---------------------------------------------------------------------------


def test_more_periods_increases_face_count(gyroid: GyroidGenerator) -> None:
    """More periods produce more geometry (more unit cells)."""
    obj_1 = gyroid.generate(
        params={"periods": 1}, voxel_resolution=_TEST_VOXEL_RESOLUTION,
    )
    obj_2 = gyroid.generate(
        params={"periods": 2}, voxel_resolution=_TEST_VOXEL_RESOLUTION,
    )

    assert obj_1.mesh is not None and obj_2.mesh is not None
    assert len(obj_2.mesh.faces) > len(obj_1.mesh.faces), (
        f"periods=2 should have more faces ({len(obj_2.mesh.faces)}) "
        f"than periods=1 ({len(obj_1.mesh.faces)})"
    )


# ---------------------------------------------------------------------------
# Bounding box extent is proportional to periods
# ---------------------------------------------------------------------------


def test_bounding_box_proportional_to_periods(
    gyroid: GyroidGenerator,
) -> None:
    """Bounding box extent scales with periods."""
    obj_small = gyroid.generate(
        params={"periods": 1},
        voxel_resolution=_TEST_VOXEL_RESOLUTION,
    )
    obj_large = gyroid.generate(
        params={"periods": 2},
        voxel_resolution=_TEST_VOXEL_RESOLUTION,
    )

    assert obj_small.bounding_box is not None
    assert obj_large.bounding_box is not None

    small_extent = np.array(obj_small.bounding_box.size)
    large_extent = np.array(obj_large.bounding_box.size)

    # periods=2 should be exactly 2x the extent of periods=1
    ratio = large_extent / small_extent
    np.testing.assert_allclose(ratio, 2.0, rtol=1e-10)


# ---------------------------------------------------------------------------
# voxel_resolution=32 still produces valid geometry (low-res sanity check)
# ---------------------------------------------------------------------------


def test_low_resolution_valid(gyroid: GyroidGenerator) -> None:
    """voxel_resolution=32 still produces valid geometry."""
    obj = gyroid.generate(voxel_resolution=32)
    obj.validate_or_raise()

    assert obj.mesh is not None
    assert len(obj.mesh.vertices) > 0
    assert len(obj.mesh.faces) > 0

    # Vertices should have valid float64 coordinates
    assert obj.mesh.vertices.dtype == np.float64
    assert not np.any(np.isnan(obj.mesh.vertices))


# ---------------------------------------------------------------------------
# Full pipeline: generate → transform to container → export STL → valid file
# ---------------------------------------------------------------------------


def test_full_pipeline_stl_roundtrip(tmp_path: Path) -> None:
    """Full pipeline produces a valid STL that can be reimported."""
    out_path = tmp_path / "gyroid.stl"

    result = run(
        "gyroid",
        resolution_kwargs={"voxel_resolution": _TEST_VOXEL_RESOLUTION},
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
    assert len(reimported.faces) > 0
    assert len(reimported.vertices) > 0


# ---------------------------------------------------------------------------
# Metadata recording
# ---------------------------------------------------------------------------


def test_metadata_recorded(gyroid: GyroidGenerator) -> None:
    """Generator name, category, and parameters are recorded."""
    obj = gyroid.generate(voxel_resolution=_TEST_VOXEL_RESOLUTION)
    assert obj.generator_name == "gyroid"
    assert obj.category == "implicit"
    assert "cell_size" not in obj.parameters
    assert obj.parameters["periods"] == 2
    assert obj.parameters["voxel_resolution"] == _TEST_VOXEL_RESOLUTION


def test_seed_recorded(gyroid: GyroidGenerator) -> None:
    """Seed is recorded in the MathObject."""
    obj = gyroid.generate(seed=999, voxel_resolution=_TEST_VOXEL_RESOLUTION)
    assert obj.seed == 999


# ---------------------------------------------------------------------------
# Registry: gyroid is discoverable
# ---------------------------------------------------------------------------


def test_gyroid_registered_and_discoverable() -> None:
    """Gyroid generator is discoverable via the registry."""
    gen_cls = get_generator("gyroid")
    assert gen_cls is GyroidGenerator


def test_default_representation_is_surface_shell(
    gyroid: GyroidGenerator,
) -> None:
    """Default representation for gyroid is SURFACE_SHELL."""
    rep = gyroid.get_default_representation()
    assert rep.type == RepresentationType.SURFACE_SHELL


# ---------------------------------------------------------------------------
# Determinism
# ---------------------------------------------------------------------------


def test_determinism_with_seed(gyroid: GyroidGenerator) -> None:
    """Same parameters produce identical geometry (gyroid is deterministic)."""
    obj1 = gyroid.generate(seed=42, voxel_resolution=_TEST_VOXEL_RESOLUTION)
    obj2 = gyroid.generate(seed=42, voxel_resolution=_TEST_VOXEL_RESOLUTION)

    assert obj1.mesh is not None and obj2.mesh is not None
    np.testing.assert_array_equal(obj1.mesh.vertices, obj2.mesh.vertices)
    np.testing.assert_array_equal(obj1.mesh.faces, obj2.mesh.faces)


# ---------------------------------------------------------------------------
# Input validation
# ---------------------------------------------------------------------------


def test_cell_size_param_ignored_gracefully(gyroid: GyroidGenerator) -> None:
    """Passing cell_size as a parameter is silently ignored."""
    obj = gyroid.generate(
        params={"cell_size": 2.0, "periods": 1},
        voxel_resolution=_TEST_VOXEL_RESOLUTION,
    )
    obj.validate_or_raise()
    assert obj.mesh is not None
    assert len(obj.mesh.vertices) > 0
    assert "cell_size" not in obj.parameters


def test_default_params_no_cell_size(gyroid: GyroidGenerator) -> None:
    """Default params dict does not contain cell_size."""
    defaults = gyroid.get_default_params()
    assert "cell_size" not in defaults
    assert "periods" in defaults


def test_zero_periods_raises(gyroid: GyroidGenerator) -> None:
    """Zero periods raises ValueError."""
    with pytest.raises(ValueError, match="periods must be >= 1"):
        gyroid.generate(params={"periods": 0})


def test_low_voxel_resolution_raises(gyroid: GyroidGenerator) -> None:
    """voxel_resolution below minimum raises ValueError."""
    with pytest.raises(ValueError, match="voxel_resolution must be >= 4"):
        gyroid.generate(voxel_resolution=2)


# ---------------------------------------------------------------------------
# Vertices within bounding box
# ---------------------------------------------------------------------------


def test_vertices_within_bounding_box(gyroid: GyroidGenerator) -> None:
    """All mesh vertices lie within the declared bounding box."""
    obj = gyroid.generate(voxel_resolution=_TEST_VOXEL_RESOLUTION)
    assert obj.mesh is not None
    assert obj.bounding_box is not None

    verts = obj.mesh.vertices
    min_c = np.array(obj.bounding_box.min_corner)
    max_c = np.array(obj.bounding_box.max_corner)

    tolerance = 1e-10
    assert np.all(verts >= min_c - tolerance)
    assert np.all(verts <= max_c + tolerance)
