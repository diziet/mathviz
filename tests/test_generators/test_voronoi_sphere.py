"""Tests for the spherical Voronoi tessellation generator."""

import numpy as np
import pytest

from mathviz.core.generator import clear_registry, get_generator, register
from mathviz.core.representation import RepresentationType
from mathviz.generators.geometry.voronoi_sphere import VoronoiSphereGenerator


@pytest.fixture(autouse=True)
def _clean_registry():
    """Reset registry for each test."""
    clear_registry(suppress_discovery=True)
    register(VoronoiSphereGenerator)
    yield
    clear_registry(suppress_discovery=True)


# ---------------------------------------------------------------------------
# Output structure
# ---------------------------------------------------------------------------


def test_produces_mesh_with_expected_structure() -> None:
    """Generator produces a mesh with expected cell structure."""
    gen = VoronoiSphereGenerator()
    obj = gen.generate(params={"num_cells": 20}, seed=42)
    obj.validate_or_raise()

    assert obj.mesh is not None
    assert obj.mesh.vertices.shape[1] == 3
    assert obj.mesh.faces.shape[1] == 3
    assert len(obj.mesh.vertices) > 0
    assert len(obj.mesh.faces) > 0
    assert obj.generator_name == "voronoi_sphere"
    assert obj.category == "geometry"
    assert obj.bounding_box is not None


# ---------------------------------------------------------------------------
# num_cells=6 produces icosahedron-like structure
# ---------------------------------------------------------------------------


def test_num_cells_6_icosahedron_like() -> None:
    """num_cells=6 produces a structure with roughly 6 cells."""
    gen = VoronoiSphereGenerator()
    obj = gen.generate(
        params={"num_cells": 6, "cell_style": "cells_only"}, seed=42,
    )
    obj.validate_or_raise()

    assert obj.mesh is not None
    # 6 cells, each a polygon with ~5 sides => ~6 * (sides-2) triangles
    # Should have a small number of faces consistent with 6 cells
    assert len(obj.mesh.faces) >= 6
    assert len(obj.mesh.faces) < 100


# ---------------------------------------------------------------------------
# num_cells=64 produces denser tessellation
# ---------------------------------------------------------------------------


def test_num_cells_64_denser() -> None:
    """num_cells=64 produces more geometry than num_cells=6."""
    gen = VoronoiSphereGenerator()
    obj_sparse = gen.generate(params={"num_cells": 6}, seed=42)
    obj_dense = gen.generate(params={"num_cells": 64}, seed=42)

    assert obj_sparse.mesh is not None
    assert obj_dense.mesh is not None
    assert len(obj_dense.mesh.vertices) > len(obj_sparse.mesh.vertices)
    assert len(obj_dense.mesh.faces) > len(obj_sparse.mesh.faces)


# ---------------------------------------------------------------------------
# Seed determinism
# ---------------------------------------------------------------------------


def test_seed_determinism() -> None:
    """Same seed produces identical output."""
    gen = VoronoiSphereGenerator()
    obj1 = gen.generate(seed=42)
    obj2 = gen.generate(seed=42)

    assert obj1.mesh is not None
    assert obj2.mesh is not None
    np.testing.assert_array_equal(obj1.mesh.vertices, obj2.mesh.vertices)
    np.testing.assert_array_equal(obj1.mesh.faces, obj2.mesh.faces)


def test_different_seeds_differ() -> None:
    """Different seeds produce different geometry."""
    gen = VoronoiSphereGenerator()
    obj1 = gen.generate(seed=42)
    obj2 = gen.generate(seed=99)

    assert obj1.mesh is not None
    assert obj2.mesh is not None
    assert not np.array_equal(obj1.mesh.vertices, obj2.mesh.vertices)


# ---------------------------------------------------------------------------
# cell_style changes output
# ---------------------------------------------------------------------------


def test_cell_style_ridges_only() -> None:
    """ridges_only style produces tube mesh geometry."""
    gen = VoronoiSphereGenerator()
    obj = gen.generate(
        params={"num_cells": 12, "cell_style": "ridges_only"}, seed=42,
    )
    obj.validate_or_raise()
    assert obj.mesh is not None
    assert len(obj.mesh.vertices) > 0


def test_cell_style_cells_only() -> None:
    """cells_only style produces cell face mesh."""
    gen = VoronoiSphereGenerator()
    obj = gen.generate(
        params={"num_cells": 12, "cell_style": "cells_only"}, seed=42,
    )
    obj.validate_or_raise()
    assert obj.mesh is not None
    assert len(obj.mesh.vertices) > 0


def test_cell_style_both() -> None:
    """both style produces more geometry than either alone."""
    gen = VoronoiSphereGenerator()
    obj_ridges = gen.generate(
        params={"num_cells": 12, "cell_style": "ridges_only"}, seed=42,
    )
    obj_cells = gen.generate(
        params={"num_cells": 12, "cell_style": "cells_only"}, seed=42,
    )
    obj_both = gen.generate(
        params={"num_cells": 12, "cell_style": "both"}, seed=42,
    )

    assert obj_both.mesh is not None
    assert obj_ridges.mesh is not None
    assert obj_cells.mesh is not None
    assert len(obj_both.mesh.vertices) > len(obj_ridges.mesh.vertices)
    assert len(obj_both.mesh.vertices) > len(obj_cells.mesh.vertices)


def test_cell_style_changes_geometry() -> None:
    """Different cell_style values produce different vertex counts."""
    gen = VoronoiSphereGenerator()
    obj_r = gen.generate(
        params={"num_cells": 20, "cell_style": "ridges_only"}, seed=42,
    )
    obj_c = gen.generate(
        params={"num_cells": 20, "cell_style": "cells_only"}, seed=42,
    )
    assert obj_r.mesh is not None
    assert obj_c.mesh is not None
    assert len(obj_r.mesh.vertices) != len(obj_c.mesh.vertices)


# ---------------------------------------------------------------------------
# edge_height=0 produces flat sphere
# ---------------------------------------------------------------------------


def test_edge_height_zero_flat_sphere() -> None:
    """edge_height=0 produces ridges at sphere surface level."""
    gen = VoronoiSphereGenerator()
    obj = gen.generate(
        params={"num_cells": 12, "edge_height": 0.0, "cell_style": "ridges_only"},
        seed=42,
    )
    obj.validate_or_raise()
    assert obj.mesh is not None

    # All ridge vertices should be near the sphere radius (1.0 default)
    distances = np.linalg.norm(obj.mesh.vertices, axis=1)
    # Tube thickening adds width, so check the median is near radius
    median_dist = np.median(distances)
    assert abs(median_dist - 1.0) < 0.2


# ---------------------------------------------------------------------------
# Registration and CLI
# ---------------------------------------------------------------------------


def test_registration() -> None:
    """Generator registers and can be looked up by name."""
    gen_cls = get_generator("voronoi_sphere")
    assert gen_cls is VoronoiSphereGenerator


def test_default_representation() -> None:
    """Default representation is TUBE."""
    gen = VoronoiSphereGenerator()
    rep = gen.get_default_representation()
    assert rep.type == RepresentationType.TUBE


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------


def test_min_cells_validation() -> None:
    """Rejects fewer than 4 cells."""
    gen = VoronoiSphereGenerator()
    with pytest.raises(ValueError, match="num_cells must be"):
        gen.generate(params={"num_cells": 2})


def test_invalid_cell_style() -> None:
    """Rejects invalid cell_style value."""
    gen = VoronoiSphereGenerator()
    with pytest.raises(ValueError, match="cell_style must be"):
        gen.generate(params={"cell_style": "invalid"})


def test_negative_radius() -> None:
    """Rejects negative radius."""
    gen = VoronoiSphereGenerator()
    with pytest.raises(ValueError, match="radius must be positive"):
        gen.generate(params={"radius": -1.0})
