"""Tests for the double pendulum dynamical system generator."""

import numpy as np
import pytest

from mathviz.core.generator import clear_registry, register
from mathviz.core.representation import RepresentationType
from mathviz.generators.attractors.double_pendulum import DoublePendulumGenerator

_TEST_STEPS = 1500
_TEST_TRANSIENT = 200


@pytest.fixture(autouse=True)
def _clean_registry():
    """Reset registry and register double_pendulum for each test."""
    clear_registry(suppress_discovery=True)
    register(DoublePendulumGenerator)
    yield
    clear_registry(suppress_discovery=True)


@pytest.fixture
def gen() -> DoublePendulumGenerator:
    """Return a DoublePendulumGenerator instance."""
    return DoublePendulumGenerator()


def test_finite_nondegenerate(gen: DoublePendulumGenerator) -> None:
    """Double pendulum produces finite, non-degenerate geometry."""
    obj = gen.generate(integration_steps=_TEST_STEPS)
    assert obj.bounding_box is not None
    min_c = np.array(obj.bounding_box.min_corner)
    max_c = np.array(obj.bounding_box.max_corner)
    assert np.all(np.isfinite(min_c))
    assert np.all(np.isfinite(max_c))
    extents = max_c - min_c
    assert np.all(extents > 0), f"Degenerate bounding box: extents={extents}"


def test_no_nan(gen: DoublePendulumGenerator) -> None:
    """Trajectory contains no NaN values."""
    obj = gen.generate(integration_steps=_TEST_STEPS)
    assert obj.curves is not None
    assert not np.any(np.isnan(obj.curves[0].points))


def test_produces_3d_points(gen: DoublePendulumGenerator) -> None:
    """Projection produces 3D points (not collapsed to a plane)."""
    obj = gen.generate(integration_steps=_TEST_STEPS)
    assert obj.curves is not None
    points = obj.curves[0].points
    # Should be (N, 3)
    assert points.shape[1] == 3

    # All three dimensions should have non-trivial extent
    extents = points.max(axis=0) - points.min(axis=0)
    assert np.all(extents > 0.01), (
        f"Points collapsed to plane: extents={extents}"
    )


def test_deterministic(gen: DoublePendulumGenerator) -> None:
    """Same seed produces identical output."""
    obj1 = gen.generate(seed=42, integration_steps=_TEST_STEPS)
    obj2 = gen.generate(seed=42, integration_steps=_TEST_STEPS)
    assert obj1.curves is not None and obj2.curves is not None
    np.testing.assert_array_equal(
        obj1.curves[0].points, obj2.curves[0].points
    )


def test_different_seeds_diverge(gen: DoublePendulumGenerator) -> None:
    """Different seeds produce different trajectories."""
    obj1 = gen.generate(seed=1, integration_steps=_TEST_STEPS)
    obj2 = gen.generate(seed=2, integration_steps=_TEST_STEPS)
    assert obj1.curves is not None and obj2.curves is not None
    assert not np.allclose(obj1.curves[0].points, obj2.curves[0].points)


def test_default_representation_is_raw_point_cloud(
    gen: DoublePendulumGenerator,
) -> None:
    """Default representation is RAW_POINT_CLOUD."""
    rep = gen.get_default_representation()
    assert rep.type == RepresentationType.RAW_POINT_CLOUD


def test_point_count(gen: DoublePendulumGenerator) -> None:
    """Point count matches integration_steps - transient_steps."""
    obj = gen.generate(
        integration_steps=_TEST_STEPS,
        params={"transient_steps": _TEST_TRANSIENT},
    )
    assert obj.curves is not None
    expected = _TEST_STEPS - _TEST_TRANSIENT
    assert len(obj.curves[0].points) == expected


def test_metadata_recorded(gen: DoublePendulumGenerator) -> None:
    """Generator name, category, and parameters are recorded."""
    obj = gen.generate(integration_steps=_TEST_STEPS)
    assert obj.generator_name == "double_pendulum"
    assert obj.category == "attractors"
    assert obj.parameters["mass"] == 1.0
    assert obj.parameters["length"] == 1.0
    assert obj.parameters["gravity"] == pytest.approx(9.81)
