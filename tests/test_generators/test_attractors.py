"""Tests for dynamical system attractor generators, starting with the Lorenz attractor."""

import numpy as np
import pytest

from mathviz.core.container import Container, PlacementPolicy
from mathviz.core.generator import clear_registry, get_generator, register
from mathviz.core.representation import RepresentationConfig, RepresentationType
from mathviz.generators.attractors.lorenz import LorenzGenerator
from mathviz.pipeline.runner import run


@pytest.fixture(autouse=True)
def _clean_registry():
    """Reset registry and re-register lorenz for each test."""
    clear_registry(suppress_discovery=True)
    register(LorenzGenerator)
    yield
    clear_registry(suppress_discovery=True)


@pytest.fixture
def lorenz() -> LorenzGenerator:
    """Return a LorenzGenerator instance."""
    return LorenzGenerator()


# Use smaller step counts for test speed
from tests.test_generators.conftest import TEST_STEPS_FAST as _TEST_STEPS
from tests.test_generators.conftest import TEST_TRANSIENT_FAST as _TEST_TRANSIENT


# ---------------------------------------------------------------------------
# Default Lorenz produces a curve with integration_steps - transient_steps points
# ---------------------------------------------------------------------------


def test_default_lorenz_curve_point_count(lorenz: LorenzGenerator) -> None:
    """Default Lorenz produces a curve with integration_steps - transient_steps points."""
    obj = lorenz.generate(
        integration_steps=_TEST_STEPS,
        params={"transient_steps": _TEST_TRANSIENT},
    )
    obj.validate_or_raise()

    assert obj.curves is not None
    assert len(obj.curves) == 1

    expected_points = _TEST_STEPS - _TEST_TRANSIENT
    assert len(obj.curves[0].points) == expected_points


# ---------------------------------------------------------------------------
# Bounding box is finite and non-degenerate
# ---------------------------------------------------------------------------


def test_bounding_box_finite_and_nondegenerate(
    lorenz: LorenzGenerator,
) -> None:
    """Bounding box is finite and non-degenerate (no NaN, no collapse to a line)."""
    obj = lorenz.generate(integration_steps=_TEST_STEPS)

    assert obj.bounding_box is not None
    min_c = np.array(obj.bounding_box.min_corner)
    max_c = np.array(obj.bounding_box.max_corner)

    # No NaN or inf
    assert np.all(np.isfinite(min_c))
    assert np.all(np.isfinite(max_c))

    # Non-degenerate: extent is positive on all three axes
    extents = max_c - min_c
    assert np.all(extents > 0), f"Degenerate bounding box: extents={extents}"

    # Lorenz attractor should span reasonable range (not collapsed)
    assert np.all(extents > 1.0), (
        f"Bounding box suspiciously small: extents={extents}"
    )


# ---------------------------------------------------------------------------
# Different seeds produce different trajectories
# ---------------------------------------------------------------------------


def test_different_seeds_produce_different_trajectories(
    lorenz: LorenzGenerator,
) -> None:
    """Different seeds produce different trajectories (initial condition perturbation)."""
    obj1 = lorenz.generate(seed=1, integration_steps=_TEST_STEPS)
    obj2 = lorenz.generate(seed=2, integration_steps=_TEST_STEPS)

    assert obj1.curves is not None and obj2.curves is not None
    # Trajectories should diverge due to chaotic sensitivity
    assert not np.allclose(obj1.curves[0].points, obj2.curves[0].points)


# ---------------------------------------------------------------------------
# Same seed produces identical output (determinism)
# ---------------------------------------------------------------------------


def test_same_seed_deterministic(lorenz: LorenzGenerator) -> None:
    """Same seed produces identical output (determinism)."""
    obj1 = lorenz.generate(seed=42, integration_steps=_TEST_STEPS)
    obj2 = lorenz.generate(seed=42, integration_steps=_TEST_STEPS)

    assert obj1.curves is not None and obj2.curves is not None
    np.testing.assert_array_equal(
        obj1.curves[0].points, obj2.curves[0].points
    )
    assert obj1.seed == obj2.seed == 42


# ---------------------------------------------------------------------------
# Output has curves populated but NOT point_cloud
# ---------------------------------------------------------------------------


def test_output_has_curves_but_not_point_cloud(
    lorenz: LorenzGenerator,
) -> None:
    """Output has curves populated but NOT point_cloud (representation handles that)."""
    obj = lorenz.generate(integration_steps=_TEST_STEPS)

    assert obj.curves is not None and len(obj.curves) == 1
    assert obj.point_cloud is None
    assert obj.mesh is None


# ---------------------------------------------------------------------------
# Full pipeline with RAW_POINT_CLOUD produces a valid PointCloud
# ---------------------------------------------------------------------------


def test_full_pipeline_raw_point_cloud() -> None:
    """Full pipeline with RAW_POINT_CLOUD representation produces a valid PointCloud."""
    result = run(
        "lorenz",
        resolution_kwargs={"integration_steps": _TEST_STEPS},
        params={"transient_steps": _TEST_TRANSIENT},
        container=Container.with_uniform_margin(),
        placement=PlacementPolicy(),
        representation_config=RepresentationConfig(
            type=RepresentationType.RAW_POINT_CLOUD,
        ),
    )

    obj = result.math_object
    assert obj.point_cloud is not None
    assert len(obj.point_cloud.points) == _TEST_STEPS - _TEST_TRANSIENT
    assert obj.point_cloud.points.dtype == np.float64

    # Points should be finite
    assert np.all(np.isfinite(obj.point_cloud.points))


# ---------------------------------------------------------------------------
# Metadata is recorded correctly
# ---------------------------------------------------------------------------


def test_metadata_recorded(lorenz: LorenzGenerator) -> None:
    """Generator name, category, and parameters are recorded."""
    obj = lorenz.generate(integration_steps=_TEST_STEPS)

    assert obj.generator_name == "lorenz"
    assert obj.category == "attractors"
    assert obj.parameters["sigma"] == 10.0
    assert obj.parameters["rho"] == 28.0
    assert obj.parameters["beta"] == pytest.approx(8.0 / 3.0)
    assert obj.parameters["integration_steps"] == _TEST_STEPS
    assert obj.seed == 42


# ---------------------------------------------------------------------------
# Registry: lorenz is discoverable
# ---------------------------------------------------------------------------


def test_lorenz_registered_and_discoverable() -> None:
    """Lorenz generator is discoverable via the registry."""
    gen_cls = get_generator("lorenz")
    assert gen_cls is LorenzGenerator


def test_lorenz_alias_discoverable() -> None:
    """Lorenz generator is discoverable via its alias."""
    gen_cls = get_generator("lorenz_attractor")
    assert gen_cls is LorenzGenerator


def test_default_representation_is_tube(lorenz: LorenzGenerator) -> None:
    """Default representation for Lorenz is TUBE."""
    rep = lorenz.get_default_representation()
    assert rep.type == RepresentationType.TUBE
    assert rep.tube_radius == 0.05


# ---------------------------------------------------------------------------
# Curve is not closed
# ---------------------------------------------------------------------------


def test_curve_is_not_closed(lorenz: LorenzGenerator) -> None:
    """Lorenz trajectory curve is open (not closed)."""
    obj = lorenz.generate(integration_steps=_TEST_STEPS)
    assert obj.curves is not None
    assert not obj.curves[0].closed


# ---------------------------------------------------------------------------
# No NaN in trajectory points
# ---------------------------------------------------------------------------


def test_no_nan_in_trajectory(lorenz: LorenzGenerator) -> None:
    """Trajectory contains no NaN values."""
    obj = lorenz.generate(integration_steps=_TEST_STEPS)
    assert obj.curves is not None
    assert not np.any(np.isnan(obj.curves[0].points))


# ---------------------------------------------------------------------------
# Input validation
# ---------------------------------------------------------------------------


def test_negative_sigma_raises(lorenz: LorenzGenerator) -> None:
    """Negative sigma raises ValueError."""
    with pytest.raises(ValueError, match="sigma must be positive"):
        lorenz.generate(params={"sigma": -1.0})


def test_zero_sigma_raises(lorenz: LorenzGenerator) -> None:
    """Zero sigma raises ValueError."""
    with pytest.raises(ValueError, match="sigma must be positive"):
        lorenz.generate(params={"sigma": 0.0})


def test_transient_steps_exceeds_integration_raises(
    lorenz: LorenzGenerator,
) -> None:
    """transient_steps >= integration_steps raises ValueError."""
    with pytest.raises(ValueError, match="integration_steps - transient_steps must be"):
        lorenz.generate(
            params={"transient_steps": 5000},
            integration_steps=5000,
        )


def test_too_few_trajectory_points_raises(
    lorenz: LorenzGenerator,
) -> None:
    """Fewer than 10 trajectory points after transient raises ValueError."""
    with pytest.raises(ValueError, match="integration_steps - transient_steps must be"):
        lorenz.generate(
            params={"transient_steps": 995},
            integration_steps=1000,
        )


def test_integration_steps_below_minimum_raises(
    lorenz: LorenzGenerator,
) -> None:
    """integration_steps below minimum raises ValueError."""
    with pytest.raises(ValueError, match="integration_steps must be >= 100"):
        lorenz.generate(integration_steps=50)


def test_integration_steps_in_params_warns(
    lorenz: LorenzGenerator,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """integration_steps passed in params is ignored with a warning."""
    obj = lorenz.generate(
        params={"integration_steps": 999},
        integration_steps=_TEST_STEPS,
    )
    assert "should be passed as a resolution kwarg" in caplog.text
    # The resolution kwarg value is used, not the params value
    assert obj.parameters["integration_steps"] == _TEST_STEPS
