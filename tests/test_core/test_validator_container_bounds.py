"""Tests for the container_bounds check on PHYSICAL coordinates.

The transformer places the container center at (width/2, height/2, depth/2), so the usable
volume spans [margin, dimension - margin] on each axis. The expected bounds below are literals,
not values computed from the Container, so a wrong formula in the validator cannot match them.
"""

import itertools

import numpy as np
import pytest

from mathviz.core.container import Container, PlacementPolicy
from mathviz.core.engraving import EngravingProfile
from mathviz.core.math_object import CoordSpace, MathObject, Mesh, PointCloud
from mathviz.core.validator import CheckResult, Severity, validate_engraving, validate_mesh
from mathviz.pipeline.runner import run
from mathviz.pipeline.sampler import SamplerConfig
from mathviz.pipeline.transformer import fit

# 80 x 120 x 40 mm block with a different margin on each axis.
_ASYMMETRIC = Container(
    width_mm=80.0, height_mm=120.0, depth_mm=40.0,
    margin_x_mm=3.0, margin_y_mm=7.5, margin_z_mm=2.0,
)
_ASYMMETRIC_LOW = np.array([3.0, 7.5, 2.0])
_ASYMMETRIC_HIGH = np.array([77.0, 112.5, 38.0])

_KINDS = ("mesh", "point_cloud")
_ANCHORS = ("center", "front", "back", "top", "bottom", "left", "right")
_BOX_FACES = np.array(
    [
        [0, 1, 3], [0, 3, 2], [4, 6, 7], [4, 7, 5],
        [0, 4, 5], [0, 5, 1], [2, 3, 7], [2, 7, 6],
        [0, 2, 6], [0, 6, 4], [1, 5, 7], [1, 7, 3],
    ],
    dtype=np.int64,
)


def _box_corners(low: np.ndarray, high: np.ndarray) -> np.ndarray:
    """Return the 8 corners of the axis-aligned box [low, high], x varying slowest."""
    return np.array(list(itertools.product(*zip(low, high))), dtype=np.float64)


def _bounds_checks(result_checks: list[CheckResult]) -> list[CheckResult]:
    """Return the container_bounds checks from a list of checks."""
    return [c for c in result_checks if c.name == "container_bounds"]


def _check_bounds(kind: str, points: np.ndarray, container: Container) -> CheckResult:
    """Validate `points` as a mesh or a point cloud and return the container_bounds check."""
    if kind == "mesh":
        faces = np.array([[0, 1, 2]], dtype=np.int64)
        result = validate_mesh(Mesh(vertices=points, faces=faces), container=container)
    else:
        result = validate_engraving(PointCloud(points=points), EngravingProfile(), container)
    checks = _bounds_checks(result.checks)
    assert len(checks) == 1, f"expected one container_bounds check, got {checks}"
    return checks[0]


def _fit_box(policy: PlacementPolicy) -> MathObject:
    """Fit a 2 x 1 x 3 abstract box (mesh and point cloud) into the asymmetric container."""
    corners = _box_corners(np.array([-1.0, -0.5, -1.5]), np.array([1.0, 0.5, 1.5]))
    interior = np.random.default_rng(7).uniform(corners.min(axis=0), corners.max(axis=0), (200, 3))
    obj = MathObject(
        mesh=Mesh(vertices=corners, faces=_BOX_FACES.copy()),
        point_cloud=PointCloud(points=np.vstack([corners, interior])),
        coord_space=CoordSpace.ABSTRACT,
    )
    return fit(obj, _ASYMMETRIC, policy)


def _fitted_checks(fitted: MathObject) -> list[CheckResult]:
    """Return the mesh and point-cloud container_bounds checks for a fitted object."""
    mesh_checks = validate_mesh(fitted.mesh, container=_ASYMMETRIC).checks
    cloud_checks = validate_engraving(fitted.point_cloud, EngravingProfile(), _ASYMMETRIC).checks
    return _bounds_checks(mesh_checks) + _bounds_checks(cloud_checks)


class TestPhysicalBounds:
    """The check compares geometry with [margin, dimension - margin] on each axis."""

    @pytest.mark.parametrize("kind", _KINDS)
    def test_geometry_filling_default_usable_volume_passes(self, kind: str) -> None:
        """Corners at 5 mm and 95 mm fill a 100 mm block with 5 mm margins and pass."""
        corners = _box_corners(np.full(3, 5.0), np.full(3, 95.0))
        assert _check_bounds(kind, corners, Container()).passed is True

    @pytest.mark.parametrize("kind", _KINDS)
    def test_geometry_on_asymmetric_per_axis_bounds_passes(self, kind: str) -> None:
        """Corners exactly on each per-axis margin plane pass."""
        corners = _box_corners(_ASYMMETRIC_LOW, _ASYMMETRIC_HIGH)
        assert _check_bounds(kind, corners, _ASYMMETRIC).passed is True

    @pytest.mark.parametrize("kind", _KINDS)
    def test_geometry_centered_on_origin_fails(self, kind: str) -> None:
        """A box centered on (0, 0, 0) lies in the margin corner of the block and fails."""
        corners = _box_corners(np.full(3, -1.0), np.full(3, 1.0))
        check = _check_bounds(kind, corners, Container())
        assert check.passed is False
        assert check.severity == Severity.ERROR

    @pytest.mark.parametrize("kind", _KINDS)
    @pytest.mark.parametrize("axis", [0, 1, 2])
    @pytest.mark.parametrize("side", ["low", "high"])
    def test_point_one_micron_beyond_a_margin_fails(self, kind: str, axis: int, side: str) -> None:
        """One point 0.001 mm past a single margin plane fails with ERROR severity."""
        stray = (_ASYMMETRIC_LOW + _ASYMMETRIC_HIGH) / 2.0
        if side == "low":
            stray[axis] = _ASYMMETRIC_LOW[axis] - 1e-3
        else:
            stray[axis] = _ASYMMETRIC_HIGH[axis] + 1e-3
        points = np.vstack([_box_corners(_ASYMMETRIC_LOW, _ASYMMETRIC_HIGH), stray])
        check = _check_bounds(kind, points, _ASYMMETRIC)
        assert check.passed is False
        assert check.severity == Severity.ERROR
        assert "usable: [3.00..77.00], [7.50..112.50], [2.00..38.00]" in check.message

    @pytest.mark.parametrize("kind", _KINDS)
    def test_rounding_overshoot_within_tolerance_passes(self, kind: str) -> None:
        """Corners 1e-12 mm past every margin plane pass, as float64 rounding error."""
        corners = _box_corners(_ASYMMETRIC_LOW - 1e-12, _ASYMMETRIC_HIGH + 1e-12)
        assert _check_bounds(kind, corners, _ASYMMETRIC).passed is True

    @pytest.mark.parametrize("kind", _KINDS)
    def test_overshoot_beyond_tolerance_fails(self, kind: str) -> None:
        """Corners 1e-7 mm past the high margin planes fail."""
        corners = _box_corners(_ASYMMETRIC_LOW, _ASYMMETRIC_HIGH + 1e-7)
        assert _check_bounds(kind, corners, _ASYMMETRIC).passed is False


class TestFittedGeometry:
    """Geometry from transformer.fit passes unless the policy places it outside."""

    @pytest.mark.parametrize("anchor", _ANCHORS)
    def test_rotated_fit_passes_for_every_anchor(self, anchor: str) -> None:
        """Rotated geometry fitted with each anchor passes for both mesh and point cloud."""
        policy = PlacementPolicy(anchor=anchor, rotation_degrees=(30.0, 45.0, 60.0))
        checks = _fitted_checks(_fit_box(policy))
        assert len(checks) == 2
        assert all(c.passed for c in checks), [c.message for c in checks]

    def test_fit_stretched_to_all_six_margin_planes_passes(self) -> None:
        """Geometry stretched to touch every margin plane passes."""
        fitted = _fit_box(PlacementPolicy(preserve_aspect_ratio=False))
        np.testing.assert_allclose(fitted.bounding_box.min_corner, _ASYMMETRIC_LOW)
        np.testing.assert_allclose(fitted.bounding_box.max_corner, _ASYMMETRIC_HIGH)
        assert all(c.passed for c in _fitted_checks(fitted))

    def test_offset_inside_usable_volume_passes(self) -> None:
        """A 20 mm x offset keeps the 24 mm wide fit inside the 74 mm usable width."""
        fitted = _fit_box(PlacementPolicy(offset_mm=(20.0, 0.0, 0.0)))
        assert all(c.passed for c in _fitted_checks(fitted))

    @pytest.mark.parametrize("axis", [0, 1, 2])
    def test_offset_past_a_margin_fails(self, axis: int) -> None:
        """A 0.5 mm offset on stretched geometry moves it past a margin plane and fails."""
        offset = [0.0, 0.0, 0.0]
        offset[axis] = 0.5
        policy = PlacementPolicy(preserve_aspect_ratio=False, offset_mm=tuple(offset))
        checks = _fitted_checks(_fit_box(policy))
        assert len(checks) == 2
        assert not any(c.passed for c in checks)

    def test_depth_bias_past_usable_depth_fails(self) -> None:
        """depth_bias=1.5 makes the depth-limited fit 54 mm deep in a 36 mm usable depth."""
        checks = _fitted_checks(_fit_box(PlacementPolicy(depth_bias=1.5)))
        assert not any(c.passed for c in checks)

    def test_depth_bias_below_one_passes(self) -> None:
        """depth_bias=0.5 compresses the fit in z and passes."""
        checks = _fitted_checks(_fit_box(PlacementPolicy(depth_bias=0.5)))
        assert all(c.passed for c in checks)


class TestPipelineContainerBounds:
    """runner.run reports container_bounds against the fitted PHYSICAL geometry."""

    @pytest.mark.parametrize("generator", ["torus", "torus_knot", "heightmap", "sacks_spiral"])
    def test_default_pipeline_reports_no_container_bounds_error(self, generator: str) -> None:
        """Surface, TUBE, heightmap and point-cloud generators pass container_bounds."""
        result = run(
            generator,
            container=Container(),
            placement=PlacementPolicy(),
            sampler_config=SamplerConfig(num_points=2000, seed=0),
            engraving_profile=EngravingProfile(),
        )
        checks = _bounds_checks(result.validation.checks)
        assert checks, f"{generator}: no container_bounds check ran"
        assert all(c.passed for c in checks), [c.message for c in checks]

    def test_pipeline_offset_outside_container_reports_error(self) -> None:
        """A 1 mm x offset moves the torus, which fills x, past the 95 mm margin plane."""
        result = run(
            "torus",
            container=Container(),
            placement=PlacementPolicy(offset_mm=(1.0, 0.0, 0.0)),
        )
        assert result.validation.passed is False
        assert "container_bounds" in {c.name for c in result.validation.errors}
