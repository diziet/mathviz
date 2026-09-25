"""Tests for the double pendulum dynamical system generator."""

import numpy as np
import pytest

from mathviz.core.generator import clear_registry, register
from mathviz.core.representation import RepresentationType
from mathviz.generators.attractors.double_pendulum import DoublePendulumGenerator
from tests.test_generators.conftest import TEST_STEPS_FAST as _TEST_STEPS
from tests.test_generators.conftest import TEST_TRANSIENT_FAST as _TEST_TRANSIENT


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
    assert points.shape[1] == 3

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


def _trajectory(params: dict[str, float]) -> np.ndarray:
    """Return the seed-42 trajectory with no transient, so row 0 is the start state."""
    obj = DoublePendulumGenerator().generate(
        seed=42,
        integration_steps=_TEST_STEPS,
        params={"transient_steps": 0, **params},
    )
    assert obj.curves is not None
    return obj.curves[0].points


@pytest.fixture(scope="module")
def default_trajectory() -> np.ndarray:
    """Return the trajectory from the default initial state."""
    return _trajectory({})


def test_default_params_start_at_previous_initial_state(
    default_trajectory: np.ndarray,
) -> None:
    """Default params start at (2.5, 2.0, 0.0) plus the seed-42 perturbation."""
    perturbation = np.random.default_rng(42).normal(scale=1e-2, size=4)
    expected = np.array([2.5, 2.0, 0.0]) + perturbation[:3]
    np.testing.assert_allclose(default_trajectory[0], expected, rtol=0, atol=1e-12)


@pytest.mark.parametrize(
    ("param", "column"), [("theta1", 0), ("theta2", 1), ("omega1", 2)],
)
def test_projected_initial_state_param_shifts_start_point(
    default_trajectory: np.ndarray, param: str, column: int,
) -> None:
    """Adding 0.75 to theta1, theta2 or omega1 adds 0.75 to that start coordinate."""
    shift = 0.75
    value = DoublePendulumGenerator().get_default_params()[param] + shift
    start = _trajectory({param: value})[0]

    expected = default_trajectory[0].copy()
    expected[column] += shift
    np.testing.assert_allclose(start, expected, rtol=0, atol=1e-12)


def test_omega2_changes_trajectory_after_start_point(
    default_trajectory: np.ndarray,
) -> None:
    """omega2 is not projected, so it keeps the start point and changes later points."""
    points = _trajectory({"omega2": 1.5})
    np.testing.assert_array_equal(points[0], default_trajectory[0])
    assert not np.allclose(points[1:], default_trajectory[1:]), (
        "omega2=1.5 should change the trajectory after the start point"
    )


@pytest.mark.parametrize("param", ["theta1", "theta2", "omega1", "omega2"])
@pytest.mark.parametrize("value", [float("nan"), float("inf")])
def test_non_finite_initial_state_param_raises(
    gen: DoublePendulumGenerator, param: str, value: float,
) -> None:
    """A NaN or infinite initial-state param raises ValueError naming the param."""
    with pytest.raises(ValueError, match=f"{param} must be finite"):
        gen.generate(params={param: value}, integration_steps=_TEST_STEPS)
