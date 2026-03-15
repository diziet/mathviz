"""Tests for the Weaire-Phelan foam structure generator."""

import numpy as np
import pytest

from mathviz.core.generator import clear_registry, get_generator, register
from mathviz.core.representation import RepresentationType
from mathviz.generators.geometry.weaire_phelan import (
    CELLS_PER_UNIT,
    WeairePhelanGenerator,
)


@pytest.fixture(autouse=True)
def _clean_registry():
    """Reset registry for each test."""
    clear_registry(suppress_discovery=True)
    register(WeairePhelanGenerator)
    yield
    clear_registry(suppress_discovery=True)


# ---------------------------------------------------------------------------
# Cell count
# ---------------------------------------------------------------------------


def test_cells_per_axis_1_produces_8_cells() -> None:
    """cells_per_axis=1 produces 8 cells (2 dodecahedra + 6 tetrakaidecahedra)."""
    gen = WeairePhelanGenerator()
    obj = gen.generate(params={"cells_per_axis": 1, "edge_only": True})
    obj.validate_or_raise()

    assert obj.parameters["cell_count"] == CELLS_PER_UNIT
    assert obj.parameters["cell_count"] == 8


def test_cells_per_axis_2_produces_64_cells() -> None:
    """cells_per_axis=2 produces 8 * 2^3 = 64 cells."""
    gen = WeairePhelanGenerator()
    obj = gen.generate(params={"cells_per_axis": 2, "edge_only": True})
    obj.validate_or_raise()

    expected = CELLS_PER_UNIT * (2 ** 3)
    assert obj.parameters["cell_count"] == expected


def test_expected_cell_count_scales_cubically() -> None:
    """Cell count scales as 8 * cells_per_axis^3."""
    gen = WeairePhelanGenerator()
    for n in (1, 2, 3):
        obj = gen.generate(params={"cells_per_axis": n, "edge_only": True})
        assert obj.parameters["cell_count"] == CELLS_PER_UNIT * n ** 3


# ---------------------------------------------------------------------------
# Edge mode vs face mode
# ---------------------------------------------------------------------------


def test_edge_mode_produces_curves() -> None:
    """edge_only=True produces curves, not mesh."""
    gen = WeairePhelanGenerator()
    obj = gen.generate(params={"cells_per_axis": 1, "edge_only": True})
    obj.validate_or_raise()

    assert obj.curves is not None
    assert len(obj.curves) > 0
    assert obj.mesh is None
    assert obj.bounding_box is not None
    for curve in obj.curves:
        assert curve.points.shape == (2, 3)
        assert not curve.closed


def test_face_mode_produces_mesh() -> None:
    """edge_only=False produces mesh, not curves."""
    gen = WeairePhelanGenerator()
    obj = gen.generate(params={"cells_per_axis": 1, "edge_only": False})
    obj.validate_or_raise()

    assert obj.mesh is not None
    assert obj.mesh.vertices.shape[1] == 3
    assert obj.mesh.faces.shape[1] == 3
    assert len(obj.mesh.vertices) > 0
    assert len(obj.mesh.faces) > 0
    assert obj.curves is None
    assert obj.bounding_box is not None


def test_more_cells_produces_more_geometry() -> None:
    """cells_per_axis=2 produces more edges than cells_per_axis=1."""
    gen = WeairePhelanGenerator()
    obj_1 = gen.generate(params={"cells_per_axis": 1, "edge_only": True})
    obj_2 = gen.generate(params={"cells_per_axis": 2, "edge_only": True})

    assert obj_1.curves is not None
    assert obj_2.curves is not None
    assert len(obj_2.curves) > len(obj_1.curves)


# ---------------------------------------------------------------------------
# Output structure
# ---------------------------------------------------------------------------


def test_generator_name_and_category() -> None:
    """Output has correct generator name and category."""
    gen = WeairePhelanGenerator()
    obj = gen.generate(params={"cells_per_axis": 1})
    assert obj.generator_name == "weaire_phelan"
    assert obj.category == "geometry"


def test_bounding_box_present() -> None:
    """Output has a bounding box."""
    gen = WeairePhelanGenerator()
    obj = gen.generate(params={"cells_per_axis": 1})
    assert obj.bounding_box is not None
    size = obj.bounding_box.size
    assert all(s > 0 for s in size)


def test_deterministic_output() -> None:
    """Same parameters produce identical output."""
    gen = WeairePhelanGenerator()
    obj1 = gen.generate(params={"cells_per_axis": 1, "edge_only": True})
    obj2 = gen.generate(params={"cells_per_axis": 1, "edge_only": True})

    assert obj1.curves is not None
    assert obj2.curves is not None
    assert len(obj1.curves) == len(obj2.curves)
    for c1, c2 in zip(obj1.curves, obj2.curves):
        np.testing.assert_array_equal(c1.points, c2.points)


def test_different_seeds_produce_identical_output() -> None:
    """Structure is purely deterministic — seed has no effect."""
    gen = WeairePhelanGenerator()
    obj1 = gen.generate(params={"cells_per_axis": 1, "edge_only": True}, seed=1)
    obj2 = gen.generate(params={"cells_per_axis": 1, "edge_only": True}, seed=99)

    assert obj1.curves is not None
    assert obj2.curves is not None
    assert len(obj1.curves) == len(obj2.curves)
    for c1, c2 in zip(obj1.curves, obj2.curves):
        np.testing.assert_array_equal(c1.points, c2.points)


# ---------------------------------------------------------------------------
# Registration
# ---------------------------------------------------------------------------


def test_registration() -> None:
    """Generator registers and can be looked up by name."""
    gen_cls = get_generator("weaire_phelan")
    assert gen_cls is WeairePhelanGenerator


def test_default_representation_is_tube() -> None:
    """Default representation is TUBE for wireframe edges."""
    gen = WeairePhelanGenerator()
    rep = gen.get_default_representation()
    assert rep.type == RepresentationType.TUBE


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------


def test_cells_per_axis_zero_rejected() -> None:
    """cells_per_axis=0 raises ValueError."""
    gen = WeairePhelanGenerator()
    with pytest.raises(ValueError, match="cells_per_axis must be >= 1"):
        gen.generate(params={"cells_per_axis": 0})


def test_cells_per_axis_negative_rejected() -> None:
    """Negative cells_per_axis raises ValueError."""
    gen = WeairePhelanGenerator()
    with pytest.raises(ValueError, match="cells_per_axis must be >= 1"):
        gen.generate(params={"cells_per_axis": -1})


def test_cells_per_axis_too_large_rejected() -> None:
    """cells_per_axis above upper bound raises ValueError."""
    gen = WeairePhelanGenerator()
    with pytest.raises(ValueError, match="cells_per_axis must be <= 20"):
        gen.generate(params={"cells_per_axis": 21})


# ---------------------------------------------------------------------------
# Default params
# ---------------------------------------------------------------------------


def test_default_params() -> None:
    """Default params are cells_per_axis=2, edge_only=True."""
    gen = WeairePhelanGenerator()
    defaults = gen.get_default_params()
    assert defaults["cells_per_axis"] == 2
    assert defaults["edge_only"] is True


def test_edge_only_string_false_produces_mesh() -> None:
    """String 'false' from CLI is correctly parsed as False."""
    gen = WeairePhelanGenerator()
    obj = gen.generate(params={"cells_per_axis": 1, "edge_only": "false"})
    obj.validate_or_raise()
    assert obj.mesh is not None
    assert obj.curves is None
