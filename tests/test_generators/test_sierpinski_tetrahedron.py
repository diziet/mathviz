"""Tests for the Sierpinski tetrahedron (tetrix) fractal generator.

Covers: level 0 single tetrahedron, level 1 produces 4 tetrahedra,
vertex count scaling, registration, representation, determinism,
parameter validation, bounding box, and full pipeline render.
"""

import numpy as np
import pytest

from mathviz.core.generator import clear_registry, get_generator, register
from mathviz.core.representation import RepresentationType
from mathviz.generators.fractals.sierpinski_tetrahedron import (
    SierpinskiTetrahedronGenerator,
    _build_tetrahedra,
)


@pytest.fixture(autouse=True)
def _clean_registry():
    """Reset registry and register the Sierpinski tetrahedron generator."""
    clear_registry(suppress_discovery=True)
    register(SierpinskiTetrahedronGenerator)
    yield
    clear_registry(suppress_discovery=True)


# --- Level 0: single tetrahedron (4 faces) ---


def test_level_0_single_tetrahedron() -> None:
    """Level 0 produces a single tetrahedron with 4 faces."""
    gen = SierpinskiTetrahedronGenerator()
    obj = gen.generate(params={"level": 0})
    obj.validate_or_raise()

    assert obj.mesh is not None
    # 1 tetrahedron: 4 triangular faces, 12 vertices (3 per face, unshared)
    assert len(obj.mesh.faces) == 4
    assert len(obj.mesh.vertices) == 12


# --- Level 1: 4 tetrahedra ---


def test_level_1_produces_4_tetrahedra() -> None:
    """Level 1 produces 4 tetrahedra."""
    tetrahedra = _build_tetrahedra(level=1, size=1.0)
    assert len(tetrahedra) == 4


def test_level_1_valid_mesh() -> None:
    """Level 1 produces a valid mesh with 16 faces."""
    gen = SierpinskiTetrahedronGenerator()
    obj = gen.generate(params={"level": 1})
    obj.validate_or_raise()

    assert obj.mesh is not None
    # 4 tetrahedra * 4 faces = 16 faces
    assert len(obj.mesh.faces) == 16
    assert len(obj.mesh.vertices) == 48


# --- Vertex/face count scales as expected ---


def test_counts_scale_with_level() -> None:
    """Face count scales as 4^level * 4."""
    gen = SierpinskiTetrahedronGenerator()
    for level in range(4):
        obj = gen.generate(params={"level": level})
        assert obj.mesh is not None
        expected_faces = (4 ** level) * 4
        expected_vertices = expected_faces * 3
        assert len(obj.mesh.faces) == expected_faces
        assert len(obj.mesh.vertices) == expected_vertices


# --- No NaN/Inf ---


def test_no_nan_or_inf_in_vertices() -> None:
    """Mesh vertices contain no NaN or Inf values."""
    gen = SierpinskiTetrahedronGenerator()
    obj = gen.generate(params={"level": 3})
    assert obj.mesh is not None
    assert np.all(np.isfinite(obj.mesh.vertices))


# --- Registration ---


def test_registered_by_name() -> None:
    """Generator is discoverable via the registry by name."""
    found = get_generator("sierpinski_tetrahedron")
    assert found is SierpinskiTetrahedronGenerator


def test_alias_tetrix_discoverable() -> None:
    """Generator is discoverable via the tetrix alias."""
    found = get_generator("tetrix")
    assert found is SierpinskiTetrahedronGenerator


def test_alias_sierpinski_tetrix_discoverable() -> None:
    """Generator is discoverable via the sierpinski_tetrix alias."""
    found = get_generator("sierpinski_tetrix")
    assert found is SierpinskiTetrahedronGenerator


# --- Representation ---


def test_default_representation_surface_shell() -> None:
    """Default representation is SURFACE_SHELL."""
    gen = SierpinskiTetrahedronGenerator()
    rep = gen.get_default_representation()
    assert rep.type == RepresentationType.SURFACE_SHELL


# --- Metadata ---


def test_metadata_recorded() -> None:
    """Generator name, category, and parameters are recorded."""
    gen = SierpinskiTetrahedronGenerator()
    obj = gen.generate(params={"level": 2})
    assert obj.generator_name == "sierpinski_tetrahedron"
    assert obj.category == "fractals"
    assert isinstance(obj.parameters, dict)
    assert obj.parameters["level"] == 2
    assert "size" in obj.parameters


# --- Determinism ---


def test_determinism() -> None:
    """Same params produce identical geometry."""
    gen = SierpinskiTetrahedronGenerator()
    obj1 = gen.generate(params={"level": 3}, seed=42)
    obj2 = gen.generate(params={"level": 3}, seed=42)

    assert obj1.mesh is not None and obj2.mesh is not None
    np.testing.assert_array_equal(obj1.mesh.vertices, obj2.mesh.vertices)
    np.testing.assert_array_equal(obj1.mesh.faces, obj2.mesh.faces)


# --- Bounding box ---


def test_vertices_within_bounding_box() -> None:
    """All mesh vertices lie within the declared bounding box."""
    gen = SierpinskiTetrahedronGenerator()
    obj = gen.generate(params={"level": 3})
    assert obj.mesh is not None
    assert obj.bounding_box is not None

    verts = obj.mesh.vertices
    min_c = np.array(obj.bounding_box.min_corner)
    max_c = np.array(obj.bounding_box.max_corner)

    tolerance = 1e-6
    assert np.all(verts >= min_c - tolerance), "vertices below bounding box"
    assert np.all(verts <= max_c + tolerance), "vertices above bounding box"


# --- Parameter validation ---


def test_negative_level_raises() -> None:
    """Negative level raises ValueError."""
    gen = SierpinskiTetrahedronGenerator()
    with pytest.raises(ValueError, match="level must be >= 0"):
        gen.generate(params={"level": -1})


def test_level_exceeds_max_raises() -> None:
    """Level above maximum raises ValueError."""
    gen = SierpinskiTetrahedronGenerator()
    with pytest.raises(ValueError, match="level must be <="):
        gen.generate(params={"level": 9})


def test_nonpositive_size_raises() -> None:
    """Non-positive or too-small size raises ValueError."""
    gen = SierpinskiTetrahedronGenerator()
    with pytest.raises(ValueError, match="size must be"):
        gen.generate(params={"size": 0.0})
    with pytest.raises(ValueError, match="size must be"):
        gen.generate(params={"size": -1.0})


# --- Size parameter ---


def test_size_affects_extent() -> None:
    """Larger size produces larger mesh extent."""
    gen = SierpinskiTetrahedronGenerator()
    obj_small = gen.generate(params={"level": 2, "size": 0.5})
    obj_large = gen.generate(params={"level": 2, "size": 2.0})

    assert obj_small.mesh is not None and obj_large.mesh is not None
    extent_small = np.ptp(obj_small.mesh.vertices, axis=0).max()
    extent_large = np.ptp(obj_large.mesh.vertices, axis=0).max()
    assert extent_large > extent_small


# --- Full pipeline render ---


def test_registers_and_renders(tmp_path) -> None:
    """Generator registers and produces exportable mesh via pipeline."""
    from mathviz.core.container import Container, PlacementPolicy
    from mathviz.core.representation import RepresentationConfig
    from mathviz.pipeline.runner import ExportConfig, run

    result = run(
        "sierpinski_tetrahedron",
        params={"level": 2},
        container=Container.with_uniform_margin(),
        placement=PlacementPolicy(),
        representation_config=RepresentationConfig(
            type=RepresentationType.SURFACE_SHELL,
        ),
        export_config=ExportConfig(
            path=tmp_path / "sierpinski_tet.stl",
            export_type="mesh",
        ),
    )

    assert result.export_path is not None
    assert result.export_path.exists()
    assert result.export_path.stat().st_size > 0
