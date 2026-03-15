"""Tests for the 3D Hilbert curve generator."""

import numpy as np
import pytest

from mathviz.core.container import Container, PlacementPolicy
from mathviz.core.generator import clear_registry, register
from mathviz.core.representation import RepresentationConfig, RepresentationType
from mathviz.generators.curves.hilbert_3d import Hilbert3DGenerator
from mathviz.pipeline.runner import run


@pytest.fixture(autouse=True)
def _clean_registry():
    """Reset registry and register the Hilbert 3D generator."""
    clear_registry(suppress_discovery=True)
    register(Hilbert3DGenerator)
    yield
    clear_registry(suppress_discovery=True)


# ===========================================================================
# Point count
# ===========================================================================


def test_order_1_produces_8_points() -> None:
    """Order 1 produces exactly 8 points."""
    gen = Hilbert3DGenerator()
    obj = gen.generate(params={"order": 1})
    assert obj.curves is not None
    assert len(obj.curves[0].points) == 8


def test_order_n_produces_8_to_the_n_points() -> None:
    """Order N produces exactly 8^N points."""
    gen = Hilbert3DGenerator()
    for order in range(1, 4):
        obj = gen.generate(params={"order": order})
        expected = 8 ** order
        actual = len(obj.curves[0].points)
        assert actual == expected, (
            f"Order {order}: expected {expected} points, got {actual}"
        )


# ===========================================================================
# Helpers
# ===========================================================================


def _to_grid(points: np.ndarray, order: int) -> np.ndarray:
    """Reverse-normalize curve points back to integer grid coordinates."""
    grid_max = (1 << order) - 1
    return np.round(points / points.max() * grid_max).astype(int)


# ===========================================================================
# Uniqueness — visits each grid cell exactly once
# ===========================================================================


def test_visits_each_grid_cell_exactly_once() -> None:
    """Curve visits each grid cell exactly once (no duplicates)."""
    gen = Hilbert3DGenerator()
    for order in range(1, 4):
        obj = gen.generate(params={"order": order})
        grid_points = _to_grid(obj.curves[0].points, order)
        unique = np.unique(grid_points, axis=0)
        assert len(unique) == 8 ** order, (
            f"Order {order}: expected {8 ** order} unique cells, "
            f"got {len(unique)}"
        )


# ===========================================================================
# Connectivity — consecutive points are adjacent
# ===========================================================================


def test_consecutive_points_are_adjacent() -> None:
    """Consecutive points on the curve are grid-adjacent (distance 1)."""
    gen = Hilbert3DGenerator()
    for order in range(1, 4):
        obj = gen.generate(params={"order": order})
        grid_points = _to_grid(obj.curves[0].points, order)
        diffs = np.diff(grid_points, axis=0)
        distances = np.sum(np.abs(diffs), axis=1)
        assert np.all(distances == 1), (
            f"Order {order}: non-adjacent steps found, "
            f"max distance {np.max(distances)}"
        )


# ===========================================================================
# Basic geometry checks
# ===========================================================================


def test_no_nan_values() -> None:
    """Hilbert 3D curve has no NaN values."""
    gen = Hilbert3DGenerator()
    obj = gen.generate(params={"order": 2})
    assert not np.any(np.isnan(obj.curves[0].points))


def test_bounding_box_finite() -> None:
    """Bounding box is finite."""
    gen = Hilbert3DGenerator()
    obj = gen.generate(params={"order": 2})
    assert obj.bounding_box is not None
    min_c = np.array(obj.bounding_box.min_corner)
    max_c = np.array(obj.bounding_box.max_corner)
    assert np.all(np.isfinite(min_c))
    assert np.all(np.isfinite(max_c))


def test_curve_is_open() -> None:
    """Hilbert curve is an open curve (not closed)."""
    gen = Hilbert3DGenerator()
    obj = gen.generate(params={"order": 2})
    assert not obj.curves[0].closed


def test_validate_or_raise_passes() -> None:
    """Generated MathObject passes validation."""
    gen = Hilbert3DGenerator()
    obj = gen.generate(params={"order": 2})
    obj.validate_or_raise()


# ===========================================================================
# Representation
# ===========================================================================


def test_default_representation_is_tube() -> None:
    """Default representation is TUBE."""
    gen = Hilbert3DGenerator()
    rep = gen.get_default_representation()
    assert rep.type == RepresentationType.TUBE


# ===========================================================================
# Parameter validation
# ===========================================================================


def test_order_below_minimum_raises() -> None:
    """Order below 1 raises ValueError."""
    gen = Hilbert3DGenerator()
    with pytest.raises(ValueError, match="order must be >= 1"):
        gen.generate(params={"order": 0})


def test_order_above_maximum_raises() -> None:
    """Order above 6 raises ValueError."""
    gen = Hilbert3DGenerator()
    with pytest.raises(ValueError, match="order must be <= 6"):
        gen.generate(params={"order": 7})


def test_negative_size_raises() -> None:
    """Negative size raises ValueError."""
    gen = Hilbert3DGenerator()
    with pytest.raises(ValueError, match="size must be positive"):
        gen.generate(params={"size": -1.0})


# ===========================================================================
# Registration and rendering
# ===========================================================================


def test_registers_and_renders() -> None:
    """Generator registers and renders through pipeline successfully."""
    result = run(
        "hilbert_3d",
        container=Container.with_uniform_margin(),
        placement=PlacementPolicy(),
        representation_config=RepresentationConfig(
            type=RepresentationType.TUBE,
            tube_radius=0.05,
            tube_sides=8,
        ),
        params={"order": 1},
    )
    assert result.math_object.mesh is not None
    assert len(result.math_object.mesh.vertices) > 0
