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
    """cells_only produces a mesh with expected cell structure."""
    gen = VoronoiSphereGenerator()
    obj = gen.generate(params={"num_cells": 20, "cell_style": "cells_only"}, seed=42)
    obj.validate_or_raise()

    assert obj.mesh is not None
    assert obj.mesh.vertices.shape[1] == 3
    assert obj.mesh.faces.shape[1] == 3
    assert len(obj.mesh.vertices) > 0
    assert len(obj.mesh.faces) > 0
    assert obj.generator_name == "voronoi_sphere"
    assert obj.category == "geometry"
    assert obj.bounding_box is not None


def test_ridges_only_returns_curves_not_mesh() -> None:
    """ridges_only returns curves for the representation layer to thicken."""
    gen = VoronoiSphereGenerator()
    obj = gen.generate(params={"num_cells": 20, "cell_style": "ridges_only"}, seed=42)
    obj.validate_or_raise()

    assert obj.curves is not None
    assert len(obj.curves) > 0
    assert obj.mesh is None
    assert obj.generator_name == "voronoi_sphere"
    assert obj.bounding_box is not None

    for curve in obj.curves:
        assert curve.points.shape[1] == 3
        assert len(curve.points) >= 2


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
    assert len(obj.mesh.faces) >= 6
    assert len(obj.mesh.faces) < 100


# ---------------------------------------------------------------------------
# num_cells=64 produces denser tessellation
# ---------------------------------------------------------------------------


def test_num_cells_64_denser() -> None:
    """num_cells=64 produces more geometry than num_cells=6."""
    gen = VoronoiSphereGenerator()
    obj_sparse = gen.generate(
        params={"num_cells": 6, "cell_style": "cells_only"}, seed=42,
    )
    obj_dense = gen.generate(
        params={"num_cells": 64, "cell_style": "cells_only"}, seed=42,
    )

    assert obj_sparse.mesh is not None
    assert obj_dense.mesh is not None
    assert len(obj_dense.mesh.vertices) > len(obj_sparse.mesh.vertices)
    assert len(obj_dense.mesh.faces) > len(obj_sparse.mesh.faces)


def test_num_cells_64_more_ridge_curves() -> None:
    """num_cells=64 produces more ridge curves than num_cells=6."""
    gen = VoronoiSphereGenerator()
    obj_sparse = gen.generate(params={"num_cells": 6}, seed=42)
    obj_dense = gen.generate(params={"num_cells": 64}, seed=42)

    assert obj_sparse.curves is not None
    assert obj_dense.curves is not None
    assert len(obj_dense.curves) > len(obj_sparse.curves)


# ---------------------------------------------------------------------------
# Seed determinism
# ---------------------------------------------------------------------------


def test_seed_determinism() -> None:
    """Same seed produces identical output."""
    gen = VoronoiSphereGenerator()
    obj1 = gen.generate(seed=42)
    obj2 = gen.generate(seed=42)

    assert obj1.curves is not None
    assert obj2.curves is not None
    assert len(obj1.curves) == len(obj2.curves)
    for c1, c2 in zip(obj1.curves, obj2.curves):
        np.testing.assert_array_equal(c1.points, c2.points)


def test_different_seeds_differ() -> None:
    """Different seeds produce different geometry."""
    gen = VoronoiSphereGenerator()
    obj1 = gen.generate(seed=42)
    obj2 = gen.generate(seed=99)

    assert obj1.curves is not None
    assert obj2.curves is not None
    all_pts1 = np.concatenate([c.points for c in obj1.curves])
    all_pts2 = np.concatenate([c.points for c in obj2.curves])
    assert not np.array_equal(all_pts1, all_pts2)


# ---------------------------------------------------------------------------
# cell_style changes output
# ---------------------------------------------------------------------------


def test_cell_style_ridges_only() -> None:
    """ridges_only style produces curves for representation layer."""
    gen = VoronoiSphereGenerator()
    obj = gen.generate(
        params={"num_cells": 12, "cell_style": "ridges_only"}, seed=42,
    )
    obj.validate_or_raise()
    assert obj.curves is not None
    assert obj.mesh is None
    assert len(obj.curves) > 0


def test_cell_style_cells_only() -> None:
    """cells_only style produces cell face mesh only."""
    gen = VoronoiSphereGenerator()
    obj = gen.generate(
        params={"num_cells": 12, "cell_style": "cells_only"}, seed=42,
    )
    obj.validate_or_raise()
    assert obj.mesh is not None
    assert obj.curves is None
    assert len(obj.mesh.vertices) > 0


def test_cell_style_both() -> None:
    """both style produces mesh and curves together."""
    gen = VoronoiSphereGenerator()
    obj = gen.generate(
        params={"num_cells": 12, "cell_style": "both"}, seed=42,
    )
    obj.validate_or_raise()
    assert obj.mesh is not None
    assert obj.curves is not None
    assert len(obj.mesh.vertices) > 0
    assert len(obj.curves) > 0


def test_cell_style_changes_geometry() -> None:
    """Different cell_style values produce different geometry types."""
    gen = VoronoiSphereGenerator()
    obj_r = gen.generate(
        params={"num_cells": 20, "cell_style": "ridges_only"}, seed=42,
    )
    obj_c = gen.generate(
        params={"num_cells": 20, "cell_style": "cells_only"}, seed=42,
    )
    # ridges_only -> curves only; cells_only -> mesh only
    assert obj_r.curves is not None and obj_r.mesh is None
    assert obj_c.mesh is not None and obj_c.curves is None


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
    assert obj.curves is not None

    # All curve points should be at exactly the sphere radius (1.0 default)
    all_pts = np.concatenate([c.points for c in obj.curves])
    distances = np.linalg.norm(all_pts, axis=1)
    np.testing.assert_allclose(distances, 1.0, atol=1e-10)


# ---------------------------------------------------------------------------
# arc_resolution is configurable
# ---------------------------------------------------------------------------


def test_arc_resolution_changes_point_count() -> None:
    """Custom arc_resolution changes the number of points per curve."""
    gen = VoronoiSphereGenerator()
    obj_low = gen.generate(params={"num_cells": 12}, seed=42, arc_resolution=4)
    obj_high = gen.generate(params={"num_cells": 12}, seed=42, arc_resolution=32)

    assert obj_low.curves is not None
    assert obj_high.curves is not None
    low_pts = sum(len(c.points) for c in obj_low.curves)
    high_pts = sum(len(c.points) for c in obj_high.curves)
    assert high_pts > low_pts


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


def test_edge_width_zero_with_ridges_rejected() -> None:
    """edge_width=0 with ridge styles is rejected."""
    gen = VoronoiSphereGenerator()
    with pytest.raises(ValueError, match="edge_width must be > 0"):
        gen.generate(params={"edge_width": 0.0, "cell_style": "ridges_only"})
    with pytest.raises(ValueError, match="edge_width must be > 0"):
        gen.generate(params={"edge_width": 0.0, "cell_style": "both"})


def test_edge_width_zero_cells_only_allowed() -> None:
    """edge_width=0 is fine for cells_only since ridges aren't used."""
    gen = VoronoiSphereGenerator()
    obj = gen.generate(
        params={"edge_width": 0.0, "cell_style": "cells_only"}, seed=42,
    )
    obj.validate_or_raise()
    assert obj.mesh is not None
