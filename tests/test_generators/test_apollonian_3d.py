"""Tests for the 3D Apollonian gasket generator."""

from pathlib import Path

import numpy as np
import pytest

from mathviz.core.container import Container, PlacementPolicy
from mathviz.core.generator import clear_registry, get_generator, register
from mathviz.core.representation import RepresentationConfig, RepresentationType
from mathviz.generators.fractals.apollonian_3d import Apollonian3DGenerator
from mathviz.pipeline.runner import ExportConfig, run


@pytest.fixture(autouse=True)
def _clean_registry():
    """Reset registry and re-register for each test."""
    clear_registry(suppress_discovery=True)
    register(Apollonian3DGenerator)
    yield
    clear_registry(suppress_discovery=True)


@pytest.fixture
def gasket() -> Apollonian3DGenerator:
    """Return an Apollonian3DGenerator instance."""
    return Apollonian3DGenerator()


# Low depth for test speed
_TEST_MAX_DEPTH = 2


# ---------------------------------------------------------------------------
# Produces a mesh containing multiple spheres
# ---------------------------------------------------------------------------


def test_produces_mesh_with_multiple_spheres(
    gasket: Apollonian3DGenerator,
) -> None:
    """Default params (low depth) produce a mesh with multiple sphere meshes."""
    obj = gasket.generate(params={"max_depth": _TEST_MAX_DEPTH})
    obj.validate_or_raise()

    assert obj.mesh is not None
    assert len(obj.mesh.vertices) > 0
    assert len(obj.mesh.faces) > 0

    radii = obj.parameters["_sphere_radii"]
    # At depth 2 we should have the 4 initial + recursively found spheres
    assert len(radii) > 4


# ---------------------------------------------------------------------------
# Deeper recursion produces more spheres
# ---------------------------------------------------------------------------


def test_deeper_recursion_produces_more_spheres(
    gasket: Apollonian3DGenerator,
) -> None:
    """Higher max_depth produces more spheres in the gasket."""
    obj_shallow = gasket.generate(params={"max_depth": 1})
    obj_deep = gasket.generate(params={"max_depth": 3})

    count_shallow = len(obj_shallow.parameters["_sphere_radii"])
    count_deep = len(obj_deep.parameters["_sphere_radii"])

    assert count_deep > count_shallow


# ---------------------------------------------------------------------------
# All spheres are non-overlapping
# ---------------------------------------------------------------------------


def test_spheres_non_overlapping(
    gasket: Apollonian3DGenerator,
) -> None:
    """All sphere pairs have centers separated by at least sum of radii."""
    obj = gasket.generate(params={"max_depth": _TEST_MAX_DEPTH})
    centers = np.array(obj.parameters["_sphere_centers"])
    radii = obj.parameters["_sphere_radii"]
    n = len(radii)

    tolerance = 1e-6
    for i in range(n):
        for j in range(i + 1, n):
            dist = float(np.linalg.norm(centers[i] - centers[j]))
            min_dist = radii[i] + radii[j]
            assert dist >= min_dist - tolerance, (
                f"Spheres {i} and {j} overlap: distance={dist:.8f}, "
                f"sum_radii={min_dist:.8f}"
            )


# ---------------------------------------------------------------------------
# Registers and renders successfully
# ---------------------------------------------------------------------------


def test_registered() -> None:
    """apollonian_3d is discoverable via the registry."""
    gen_cls = get_generator("apollonian_3d")
    assert gen_cls is Apollonian3DGenerator


def test_alias_registered() -> None:
    """apollonian_gasket_3d alias is discoverable."""
    gen_cls = get_generator("apollonian_gasket_3d")
    assert gen_cls is Apollonian3DGenerator


def test_full_pipeline_renders(tmp_path: Path) -> None:
    """Full pipeline with SURFACE_SHELL produces a valid STL."""
    out_path = tmp_path / "apollonian.stl"

    result = run(
        "apollonian_3d",
        resolution_kwargs={"icosphere_subdivisions": 1},
        params={"max_depth": 1},
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


# ---------------------------------------------------------------------------
# Default representation
# ---------------------------------------------------------------------------


def test_default_representation(
    gasket: Apollonian3DGenerator,
) -> None:
    """Default representation is SURFACE_SHELL."""
    rep = gasket.get_default_representation()
    assert rep.type == RepresentationType.SURFACE_SHELL


# ---------------------------------------------------------------------------
# Metadata recording
# ---------------------------------------------------------------------------


def test_metadata_recorded(
    gasket: Apollonian3DGenerator,
) -> None:
    """Generator name, category, and parameters are recorded."""
    obj = gasket.generate(params={"max_depth": _TEST_MAX_DEPTH})
    assert obj.generator_name == "apollonian_3d"
    assert obj.category == "fractals"
    assert obj.parameters["max_depth"] == _TEST_MAX_DEPTH
    assert obj.parameters["min_radius"] == 0.01


def test_seed_recorded(
    gasket: Apollonian3DGenerator,
) -> None:
    """Seed is recorded in the MathObject."""
    obj = gasket.generate(seed=999, params={"max_depth": 1})
    assert obj.seed == 999


# ---------------------------------------------------------------------------
# Determinism
# ---------------------------------------------------------------------------


def test_determinism(
    gasket: Apollonian3DGenerator,
) -> None:
    """Same seed + params produces identical output."""
    obj1 = gasket.generate(seed=42, params={"max_depth": _TEST_MAX_DEPTH})
    obj2 = gasket.generate(seed=42, params={"max_depth": _TEST_MAX_DEPTH})

    assert obj1.mesh is not None and obj2.mesh is not None
    np.testing.assert_array_equal(obj1.mesh.vertices, obj2.mesh.vertices)
    np.testing.assert_array_equal(obj1.mesh.faces, obj2.mesh.faces)


def test_different_seeds_differ(
    gasket: Apollonian3DGenerator,
) -> None:
    """Different seeds produce different gaskets."""
    obj1 = gasket.generate(seed=1, params={"max_depth": _TEST_MAX_DEPTH})
    obj2 = gasket.generate(seed=2, params={"max_depth": _TEST_MAX_DEPTH})

    assert obj1.mesh is not None and obj2.mesh is not None
    assert not np.allclose(obj1.mesh.vertices, obj2.mesh.vertices)


# ---------------------------------------------------------------------------
# Bounding box
# ---------------------------------------------------------------------------


def test_bounding_box_set(
    gasket: Apollonian3DGenerator,
) -> None:
    """Bounding box is populated after generation."""
    obj = gasket.generate(params={"max_depth": _TEST_MAX_DEPTH})
    assert obj.bounding_box is not None


# ---------------------------------------------------------------------------
# Input validation
# ---------------------------------------------------------------------------


def test_negative_max_depth_raises(
    gasket: Apollonian3DGenerator,
) -> None:
    """Negative max_depth raises ValueError."""
    with pytest.raises(ValueError, match="max_depth must be >= 0"):
        gasket.generate(params={"max_depth": -1})


def test_excessive_max_depth_raises(
    gasket: Apollonian3DGenerator,
) -> None:
    """max_depth above limit raises ValueError."""
    with pytest.raises(ValueError, match="max_depth must be <= 8"):
        gasket.generate(params={"max_depth": 9})


def test_tiny_min_radius_raises(
    gasket: Apollonian3DGenerator,
) -> None:
    """min_radius below floor raises ValueError."""
    with pytest.raises(ValueError, match="min_radius must be >="):
        gasket.generate(params={"min_radius": 1e-12})


def test_excessive_icosphere_subdivisions_raises(
    gasket: Apollonian3DGenerator,
) -> None:
    """icosphere_subdivisions above limit raises ValueError."""
    with pytest.raises(ValueError, match="icosphere_subdivisions must be <="):
        gasket.generate(
            params={"max_depth": 0}, icosphere_subdivisions=10,
        )


def test_negative_icosphere_subdivisions_raises(
    gasket: Apollonian3DGenerator,
) -> None:
    """Negative icosphere_subdivisions raises ValueError."""
    with pytest.raises(ValueError, match="icosphere_subdivisions must be >= 0"):
        gasket.generate(
            params={"max_depth": 0}, icosphere_subdivisions=-1,
        )


# ---------------------------------------------------------------------------
# Zero depth produces only initial 4 spheres
# ---------------------------------------------------------------------------


def test_zero_depth_produces_initial_spheres(
    gasket: Apollonian3DGenerator,
) -> None:
    """max_depth=0 produces exactly the 4 initial inner spheres."""
    obj = gasket.generate(params={"max_depth": 0})
    obj.validate_or_raise()
    radii = obj.parameters["_sphere_radii"]
    assert len(radii) == 4


# ---------------------------------------------------------------------------
# min_radius controls sphere pruning
# ---------------------------------------------------------------------------


def test_larger_min_radius_fewer_spheres(
    gasket: Apollonian3DGenerator,
) -> None:
    """Larger min_radius prunes more spheres, producing fewer total."""
    obj_fine = gasket.generate(
        params={"max_depth": 3, "min_radius": 0.01},
    )
    obj_coarse = gasket.generate(
        params={"max_depth": 3, "min_radius": 0.1},
    )

    count_fine = len(obj_fine.parameters["_sphere_radii"])
    count_coarse = len(obj_coarse.parameters["_sphere_radii"])

    assert count_fine > count_coarse
