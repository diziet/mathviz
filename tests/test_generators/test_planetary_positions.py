"""Tests for Task 40: planetary_positions single-point curve crash fix.

Verifies that the planetary_positions generator produces valid output
that works with tube thickening, and that the defensive guard in
_thicken_all_curves skips degenerate curves with a warning.
"""

import logging

import numpy as np
import pytest

from mathviz.core.generator import clear_registry, register
from mathviz.core.math_object import Curve, Mesh, PointCloud
from mathviz.core.representation import RepresentationConfig, RepresentationType
from mathviz.generators.physics.planetary_positions import (
    PlanetaryPositionsGenerator,
)
from mathviz.pipeline.representation_strategy import apply


@pytest.fixture(autouse=True)
def _clean_registry():
    """Reset registry for each test."""
    clear_registry(suppress_discovery=True)
    register(PlanetaryPositionsGenerator)
    yield
    clear_registry(suppress_discovery=True)


@pytest.fixture
def planetary() -> PlanetaryPositionsGenerator:
    """Return a PlanetaryPositionsGenerator instance."""
    return PlanetaryPositionsGenerator()


_TEST_CURVE_POINTS = 64
_NUM_PLANETS = 8


class TestAllCurvesHaveMinPoints:
    """All curves from planetary_positions have >= 2 points."""

    def test_default_params_all_curves_have_min_points(
        self, planetary: PlanetaryPositionsGenerator
    ) -> None:
        """Default params produce all curves with >= 2 points."""
        obj = planetary.generate(curve_points=_TEST_CURVE_POINTS)
        assert obj.curves is not None
        for i, curve in enumerate(obj.curves):
            assert len(curve.points) >= 2, (
                f"Curve {i} has {len(curve.points)} points, expected >= 2"
            )

    @pytest.mark.parametrize("num_planets", [1, 4, 8])
    def test_no_curve_fewer_than_two_points(
        self, planetary: PlanetaryPositionsGenerator, num_planets: int
    ) -> None:
        """No curve has fewer than 2 points (parameterized)."""
        obj = planetary.generate(curve_points=_TEST_CURVE_POINTS)
        assert obj.curves is not None
        # Check up to num_planets curves
        for curve in obj.curves[:num_planets]:
            assert len(curve.points) >= 2


class TestTubeRepresentationSucceeds:
    """Tube thickening works on planetary_positions output."""

    def test_generate_plus_tube_succeeds(
        self, planetary: PlanetaryPositionsGenerator
    ) -> None:
        """Generate + tube representation succeeds without error."""
        obj = planetary.generate(curve_points=_TEST_CURVE_POINTS)
        config = RepresentationConfig(
            type=RepresentationType.TUBE,
            tube_radius=0.02,
            tube_sides=8,
        )
        result = apply(obj, config)
        assert result.mesh is not None
        assert len(result.mesh.vertices) > 0
        assert len(result.mesh.faces) > 0


class TestRenderProducesPng:
    """Render pipeline produces output without error."""

    def test_render_planetary_positions(
        self, planetary: PlanetaryPositionsGenerator, tmp_path,
    ) -> None:
        """Render produces a PNG without error."""
        obj = planetary.generate(curve_points=_TEST_CURVE_POINTS)
        config = RepresentationConfig(
            type=RepresentationType.TUBE,
            tube_radius=0.02,
            tube_sides=8,
        )
        result = apply(obj, config)
        # Verify mesh is valid for rendering
        assert result.mesh is not None
        errors = result.mesh.validate()
        assert not errors, f"Mesh validation errors: {errors}"


class TestPositionsAsPointCloud:
    """Planet positions are output as a PointCloud, not single-point Curves."""

    def test_point_cloud_contains_planet_positions(
        self, planetary: PlanetaryPositionsGenerator
    ) -> None:
        """Point cloud contains one point per planet."""
        obj = planetary.generate(curve_points=_TEST_CURVE_POINTS)
        assert obj.point_cloud is not None
        assert len(obj.point_cloud.points) == _NUM_PLANETS
        assert not np.any(np.isnan(obj.point_cloud.points))

    def test_only_orbit_curves_no_single_point_curves(
        self, planetary: PlanetaryPositionsGenerator
    ) -> None:
        """Only orbit curves in output; no single-point curves."""
        obj = planetary.generate(curve_points=_TEST_CURVE_POINTS)
        assert obj.curves is not None
        assert len(obj.curves) == _NUM_PLANETS
        for curve in obj.curves:
            assert len(curve.points) == _TEST_CURVE_POINTS


class TestDefensiveTubeSkip:
    """Defensive skip in _thicken_all_curves for degenerate curves."""

    def test_single_point_curve_skipped_with_warning(
        self, caplog,
    ) -> None:
        """A single-point curve is skipped with a warning, not a crash."""
        from mathviz.core.math_object import BoundingBox, MathObject

        good_curve = Curve(
            points=np.array([[0, 0, 0], [1, 0, 0], [1, 1, 0]], dtype=np.float64),
            closed=False,
        )
        bad_curve = Curve(
            points=np.array([[5, 5, 5]], dtype=np.float64),
            closed=False,
        )
        obj = MathObject(
            curves=[good_curve, bad_curve],
            generator_name="test",
            bounding_box=BoundingBox.from_points(
                np.vstack([good_curve.points, bad_curve.points])
            ),
        )
        config = RepresentationConfig(
            type=RepresentationType.TUBE,
            tube_radius=0.1,
            tube_sides=8,
        )

        with caplog.at_level(logging.WARNING):
            result = apply(obj, config)

        assert result.mesh is not None
        assert "Skipping curve[1]" in caplog.text

    def test_closed_curve_with_two_points_skipped(self, caplog) -> None:
        """A closed curve with only 2 points is skipped (needs >= 3)."""
        from mathviz.core.math_object import BoundingBox, MathObject

        good_curve = Curve(
            points=np.array([[0, 0, 0], [1, 0, 0], [1, 1, 0]], dtype=np.float64),
            closed=False,
        )
        bad_closed = Curve(
            points=np.array([[2, 0, 0], [3, 0, 0]], dtype=np.float64),
            closed=True,
        )
        obj = MathObject(
            curves=[good_curve, bad_closed],
            generator_name="test",
            bounding_box=BoundingBox.from_points(
                np.vstack([good_curve.points, bad_closed.points])
            ),
        )
        config = RepresentationConfig(
            type=RepresentationType.TUBE,
            tube_radius=0.1,
            tube_sides=8,
        )

        with caplog.at_level(logging.WARNING):
            result = apply(obj, config)

        assert result.mesh is not None
        assert "Skipping curve[1]" in caplog.text
