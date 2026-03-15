"""Tests for the quaternion Julia set generator."""

from pathlib import Path

import numpy as np
import pytest
import trimesh

from mathviz.core.container import Container, PlacementPolicy
from mathviz.core.generator import clear_registry, get_generator, register
from mathviz.core.representation import RepresentationConfig, RepresentationType
from mathviz.generators.fractals.quaternion_julia import QuaternionJuliaGenerator
from mathviz.pipeline.runner import ExportConfig, run


@pytest.fixture(autouse=True)
def _clean_registry():
    """Reset registry and re-register quaternion Julia for each test."""
    clear_registry(suppress_discovery=True)
    register(QuaternionJuliaGenerator)
    yield
    clear_registry(suppress_discovery=True)


@pytest.fixture
def qjulia() -> QuaternionJuliaGenerator:
    """Return a QuaternionJuliaGenerator instance."""
    return QuaternionJuliaGenerator()


_TEST_VOXEL_RESOLUTION = 32


# ---------------------------------------------------------------------------
# Produces a valid mesh
# ---------------------------------------------------------------------------


def test_produces_valid_mesh(qjulia: QuaternionJuliaGenerator) -> None:
    """Default parameters produce a non-empty, valid mesh."""
    obj = qjulia.generate(voxel_resolution=_TEST_VOXEL_RESOLUTION)
    obj.validate_or_raise()
    assert obj.mesh is not None
    assert len(obj.mesh.vertices) > 0
    assert len(obj.mesh.faces) > 0


# ---------------------------------------------------------------------------
# Different c values produce distinct shapes
# ---------------------------------------------------------------------------


def test_different_c_values_produce_distinct_shapes(
    qjulia: QuaternionJuliaGenerator,
) -> None:
    """Different c constants produce meshes with different vertex counts."""
    obj_a = qjulia.generate(
        params={"c_real": -0.2, "c_i": 0.8, "c_j": 0.0, "c_k": 0.0},
        voxel_resolution=_TEST_VOXEL_RESOLUTION,
    )
    obj_b = qjulia.generate(
        params={"c_real": -0.4, "c_i": 0.6, "c_j": 0.1, "c_k": -0.1},
        voxel_resolution=_TEST_VOXEL_RESOLUTION,
    )
    assert obj_a.mesh is not None and obj_b.mesh is not None
    # Shapes should differ — either vertex count or positions
    vertices_differ = len(obj_a.mesh.vertices) != len(obj_b.mesh.vertices)
    if not vertices_differ:
        vertices_differ = not np.allclose(
            obj_a.mesh.vertices, obj_b.mesh.vertices,
        )
    assert vertices_differ, "Different c values should produce distinct shapes"


# ---------------------------------------------------------------------------
# Higher voxel_resolution produces more detailed mesh
# ---------------------------------------------------------------------------


def test_higher_resolution_more_detail(
    qjulia: QuaternionJuliaGenerator,
) -> None:
    """Higher voxel_resolution produces a mesh with more vertices."""
    obj_low = qjulia.generate(voxel_resolution=16)
    obj_high = qjulia.generate(voxel_resolution=48)
    assert obj_low.mesh is not None and obj_high.mesh is not None
    assert len(obj_high.mesh.vertices) > len(obj_low.mesh.vertices)


# ---------------------------------------------------------------------------
# Registers and renders successfully
# ---------------------------------------------------------------------------


def test_registered() -> None:
    """quaternion_julia is discoverable via the registry."""
    gen_cls = get_generator("quaternion_julia")
    assert gen_cls is QuaternionJuliaGenerator


def test_alias_registered() -> None:
    """qjulia alias is discoverable via the registry."""
    gen_cls = get_generator("qjulia")
    assert gen_cls is QuaternionJuliaGenerator


def test_full_pipeline_renders(tmp_path: Path) -> None:
    """Full pipeline with SURFACE_SHELL produces a valid STL."""
    out_path = tmp_path / "qjulia.stl"

    result = run(
        "quaternion_julia",
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

    reimported = trimesh.load(str(out_path), file_type="stl")
    assert len(reimported.faces) > 0
    assert len(reimported.vertices) > 0


# ---------------------------------------------------------------------------
# Determinism
# ---------------------------------------------------------------------------


def test_determinism(qjulia: QuaternionJuliaGenerator) -> None:
    """Same seed + params produces identical output."""
    obj1 = qjulia.generate(seed=42, voxel_resolution=_TEST_VOXEL_RESOLUTION)
    obj2 = qjulia.generate(seed=42, voxel_resolution=_TEST_VOXEL_RESOLUTION)
    assert obj1.mesh is not None and obj2.mesh is not None
    np.testing.assert_array_equal(obj1.mesh.vertices, obj2.mesh.vertices)
    np.testing.assert_array_equal(obj1.mesh.faces, obj2.mesh.faces)


# ---------------------------------------------------------------------------
# Metadata
# ---------------------------------------------------------------------------


def test_metadata_recorded(qjulia: QuaternionJuliaGenerator) -> None:
    """Generator name, category, and parameters are recorded."""
    obj = qjulia.generate(voxel_resolution=_TEST_VOXEL_RESOLUTION)
    assert obj.generator_name == "quaternion_julia"
    assert obj.category == "fractals"
    assert obj.parameters["c_real"] == -0.2
    assert obj.parameters["c_i"] == 0.8
    assert obj.parameters["max_iter"] == 10
    assert obj.parameters["voxel_resolution"] == _TEST_VOXEL_RESOLUTION


# ---------------------------------------------------------------------------
# Default representation
# ---------------------------------------------------------------------------


def test_default_representation(qjulia: QuaternionJuliaGenerator) -> None:
    """Default representation is SURFACE_SHELL."""
    rep = qjulia.get_default_representation()
    assert rep.type == RepresentationType.SURFACE_SHELL


# ---------------------------------------------------------------------------
# Bounding box
# ---------------------------------------------------------------------------


def test_bounding_box_set(qjulia: QuaternionJuliaGenerator) -> None:
    """Bounding box is populated after generation."""
    obj = qjulia.generate(voxel_resolution=_TEST_VOXEL_RESOLUTION)
    assert obj.bounding_box is not None


# ---------------------------------------------------------------------------
# Slice_w variation
# ---------------------------------------------------------------------------


def test_slice_w_changes_shape(qjulia: QuaternionJuliaGenerator) -> None:
    """Different slice_w values produce different meshes."""
    obj_a = qjulia.generate(
        params={"slice_w": 0.0},
        voxel_resolution=_TEST_VOXEL_RESOLUTION,
    )
    obj_b = qjulia.generate(
        params={"slice_w": 0.3},
        voxel_resolution=_TEST_VOXEL_RESOLUTION,
    )
    assert obj_a.mesh is not None and obj_b.mesh is not None
    vertices_differ = len(obj_a.mesh.vertices) != len(obj_b.mesh.vertices)
    if not vertices_differ:
        vertices_differ = not np.allclose(
            obj_a.mesh.vertices, obj_b.mesh.vertices,
        )
    assert vertices_differ, "Different slice_w should produce distinct shapes"
