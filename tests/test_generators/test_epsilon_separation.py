"""Tests for epsilon separation of self-intersecting parametric surfaces."""

import numpy as np
import pytest
from scipy.spatial import cKDTree

from mathviz.core.generator import clear_registry, register
from mathviz.generators.parametric._mesh_utils import COINCIDENCE_THRESHOLD
from mathviz.generators.parametric.boy_surface import BoySurfaceGenerator
from mathviz.generators.parametric.cross_cap import CrossCapGenerator
from mathviz.generators.parametric.klein_bottle import KleinBottleGenerator
from mathviz.generators.parametric.roman_surface import RomanSurfaceGenerator

_GENERATORS = [
    KleinBottleGenerator,
    CrossCapGenerator,
    RomanSurfaceGenerator,
    BoySurfaceGenerator,
]

_GRID = 64


@pytest.fixture(autouse=True)
def _clean_registry():
    """Reset registry for each test."""
    clear_registry(suppress_discovery=True)
    for gen_cls in _GENERATORS:
        register(gen_cls)
    yield
    clear_registry(suppress_discovery=True)


def _count_coincident_pairs(
    vertices: np.ndarray, threshold: float,
) -> int:
    """Count vertex pairs closer than threshold."""
    tree = cKDTree(vertices)
    pairs = tree.query_pairs(threshold, output_type="ndarray")
    return len(pairs)


# ------------------------------------------------------------------
# Default epsilon eliminates coincident pairs
# ------------------------------------------------------------------


@pytest.mark.parametrize("gen_cls", _GENERATORS, ids=lambda c: c.name)
def test_default_epsilon_no_coincident_pairs(gen_cls: type) -> None:
    """Default epsilon produces zero coincident vertex pairs."""
    gen = gen_cls()
    obj = gen.generate(grid_resolution=_GRID)
    assert obj.mesh is not None

    count = _count_coincident_pairs(
        obj.mesh.vertices, COINCIDENCE_THRESHOLD,
    )
    assert count == 0, (
        f"{gen_cls.name}: {count} coincident pairs remain after separation"
    )


# ------------------------------------------------------------------
# Min nearest-neighbor distance > 0
# ------------------------------------------------------------------


@pytest.mark.parametrize("gen_cls", _GENERATORS, ids=lambda c: c.name)
def test_min_nn_distance_positive(gen_cls: type) -> None:
    """Default epsilon produces positive minimum nearest-neighbor distance."""
    gen = gen_cls()
    obj = gen.generate(grid_resolution=_GRID)
    assert obj.mesh is not None

    tree = cKDTree(obj.mesh.vertices)
    dists, _ = tree.query(obj.mesh.vertices, k=2)
    min_nn = dists[:, 1].min()
    assert min_nn > 0, f"{gen_cls.name}: min NN distance is {min_nn}"


# ------------------------------------------------------------------
# epsilon=0 disables separation, coincident pairs return
# ------------------------------------------------------------------


@pytest.mark.parametrize("gen_cls", _GENERATORS, ids=lambda c: c.name)
def test_epsilon_zero_has_coincident_pairs(gen_cls: type) -> None:
    """Setting separation_epsilon=0 disables separation."""
    gen = gen_cls()
    obj = gen.generate(
        params={"separation_epsilon": 0.0},
        grid_resolution=_GRID,
    )
    assert obj.mesh is not None

    count = _count_coincident_pairs(
        obj.mesh.vertices, COINCIDENCE_THRESHOLD,
    )
    assert count > 0, (
        f"{gen_cls.name}: expected coincident pairs with epsilon=0"
    )


# ------------------------------------------------------------------
# Vertex/face counts unchanged by separation
# ------------------------------------------------------------------


@pytest.mark.parametrize("gen_cls", _GENERATORS, ids=lambda c: c.name)
def test_vertex_face_counts_unchanged(gen_cls: type) -> None:
    """Separation does not add or remove vertices or faces."""
    gen = gen_cls()
    obj_on = gen.generate(grid_resolution=_GRID)
    obj_off = gen.generate(
        params={"separation_epsilon": 0.0},
        grid_resolution=_GRID,
    )
    assert obj_on.mesh is not None and obj_off.mesh is not None

    assert len(obj_on.mesh.vertices) == len(obj_off.mesh.vertices)
    assert len(obj_on.mesh.faces) == len(obj_off.mesh.faces)


# ------------------------------------------------------------------
# Bounding box within 1% of original
# ------------------------------------------------------------------


@pytest.mark.parametrize("gen_cls", _GENERATORS, ids=lambda c: c.name)
def test_bounding_box_within_tolerance(gen_cls: type) -> None:
    """Bounding box is not significantly changed (within 1%)."""
    gen = gen_cls()
    obj_on = gen.generate(grid_resolution=_GRID)
    obj_off = gen.generate(
        params={"separation_epsilon": 0.0},
        grid_resolution=_GRID,
    )
    assert obj_on.mesh is not None and obj_off.mesh is not None

    verts_on = obj_on.mesh.vertices
    verts_off = obj_off.mesh.vertices

    extent_off = verts_off.max(axis=0) - verts_off.min(axis=0)
    max_extent = extent_off.max()
    tolerance = max_extent * 0.01

    diff_min = np.abs(verts_on.min(axis=0) - verts_off.min(axis=0))
    diff_max = np.abs(verts_on.max(axis=0) - verts_off.max(axis=0))

    assert np.all(diff_min < tolerance), (
        f"{gen_cls.name}: min corner shifted by {diff_min}"
    )
    assert np.all(diff_max < tolerance), (
        f"{gen_cls.name}: max corner shifted by {diff_max}"
    )


# ------------------------------------------------------------------
# Existing test suites still pass (smoke test at low resolution)
# ------------------------------------------------------------------


@pytest.mark.parametrize("gen_cls", _GENERATORS, ids=lambda c: c.name)
def test_validate_still_passes(gen_cls: type) -> None:
    """Meshes with epsilon separation still pass validation."""
    gen = gen_cls()
    obj = gen.generate(grid_resolution=32)
    obj.validate_or_raise()


# ------------------------------------------------------------------
# Validation rejects negative and excessively large epsilon
# ------------------------------------------------------------------


@pytest.mark.parametrize("gen_cls", _GENERATORS, ids=lambda c: c.name)
def test_negative_epsilon_raises(gen_cls: type) -> None:
    """Negative separation_epsilon raises ValueError."""
    gen = gen_cls()
    with pytest.raises(ValueError, match="separation_epsilon must be >= 0"):
        gen.generate(params={"separation_epsilon": -0.01}, grid_resolution=16)


@pytest.mark.parametrize("gen_cls", _GENERATORS, ids=lambda c: c.name)
def test_excessive_epsilon_raises(gen_cls: type) -> None:
    """Excessively large separation_epsilon raises ValueError."""
    gen = gen_cls()
    with pytest.raises(ValueError, match="separation_epsilon must be <="):
        gen.generate(params={"separation_epsilon": 1000.0}, grid_resolution=16)
