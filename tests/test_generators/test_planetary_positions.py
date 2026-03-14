"""Tests for Task 40: planetary_positions single-point curve crash fix.

Verifies that the planetary_positions generator produces valid output
that works with tube thickening, and that the defensive guard in
_thicken_all_curves skips degenerate curves with an error log.
"""

import logging

import numpy as np
import pytest

from mathviz.core.generator import clear_registry, register
from mathviz.core.math_object import BoundingBox, Curve, MathObject
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


@pytest.fixture
def tube_config() -> RepresentationConfig:
    """Return a tube representation config for tests."""
    return RepresentationConfig(
        type=RepresentationType.TUBE,
        tube_radius=0.02,
        tube_sides=8,
    )


_TEST_CURVE_POINTS = 64


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


class TestTubeRepresentationSucceeds:
    """Tube thickening works on planetary_positions output."""

    def test_generate_plus_tube_succeeds(
        self,
        planetary: PlanetaryPositionsGenerator,
        tube_config: RepresentationConfig,
    ) -> None:
        """Generate + tube representation succeeds without error."""
        obj = planetary.generate(curve_points=_TEST_CURVE_POINTS)
        result = apply(obj, tube_config)
        assert result.mesh is not None
        assert len(result.mesh.vertices) > 0
        assert len(result.mesh.faces) > 0


class TestMeshValidAfterTube:
    """Mesh produced by tube thickening passes validation."""

    def test_tube_mesh_passes_validation(
        self,
        planetary: PlanetaryPositionsGenerator,
        tube_config: RepresentationConfig,
    ) -> None:
        """Tube mesh has no validation errors."""
        obj = planetary.generate(curve_points=_TEST_CURVE_POINTS)
        result = apply(obj, tube_config)
        assert result.mesh is not None
        errors = result.mesh.validate()
        assert not errors, f"Mesh validation errors: {errors}"


def _make_obj_with_bad_curve(bad_curve: Curve) -> MathObject:
    """Build a MathObject with one good curve and one degenerate curve."""
    good_curve = Curve(
        points=np.array([[0, 0, 0], [1, 0, 0], [1, 1, 0]], dtype=np.float64),
        closed=False,
    )
    return MathObject(
        curves=[good_curve, bad_curve],
        generator_name="test",
        bounding_box=BoundingBox.from_points(
            np.vstack([good_curve.points, bad_curve.points])
        ),
    )


_DEFENSIVE_TUBE_CONFIG = RepresentationConfig(
    type=RepresentationType.TUBE,
    tube_radius=0.1,
    tube_sides=8,
)


class TestDefensiveTubeSkip:
    """Defensive skip in _thicken_all_curves for degenerate curves."""

    def test_single_point_curve_skipped_with_error_log(
        self, caplog,
    ) -> None:
        """A single-point curve is skipped with an error log, not a crash."""
        bad_curve = Curve(
            points=np.array([[5, 5, 5]], dtype=np.float64),
            closed=False,
        )
        obj = _make_obj_with_bad_curve(bad_curve)

        with caplog.at_level(logging.ERROR):
            result = apply(obj, _DEFENSIVE_TUBE_CONFIG)

        assert result.mesh is not None
        assert "Skipping curve[1]" in caplog.text

    def test_closed_curve_with_two_points_skipped(self, caplog) -> None:
        """A closed curve with only 2 points is skipped (needs >= 3)."""
        bad_closed = Curve(
            points=np.array([[2, 0, 0], [3, 0, 0]], dtype=np.float64),
            closed=True,
        )
        obj = _make_obj_with_bad_curve(bad_closed)

        with caplog.at_level(logging.ERROR):
            result = apply(obj, _DEFENSIVE_TUBE_CONFIG)

        assert result.mesh is not None
        assert "Skipping curve[1]" in caplog.text
