"""Tests for the Hopf fibration generator."""

import numpy as np
import pytest

from mathviz.core.generator import clear_registry, get_generator, register
from mathviz.core.representation import RepresentationType
from mathviz.generators.parametric.hopf_fibration import HopfFibrationGenerator


@pytest.fixture(autouse=True)
def _clean_registry():
    """Reset registry and register Hopf fibration generator for each test."""
    clear_registry(suppress_discovery=True)
    register(HopfFibrationGenerator)
    yield
    clear_registry(suppress_discovery=True)


@pytest.fixture
def gen() -> HopfFibrationGenerator:
    """Return a HopfFibrationGenerator instance."""
    return HopfFibrationGenerator()


_TEST_FIBER_POINTS = 64


class TestHopfFibrationStructure:
    """Tests for fiber count and organization."""

    def test_produces_correct_number_of_curves(
        self, gen: HopfFibrationGenerator,
    ) -> None:
        """Output has num_fibers * num_circles curves."""
        num_fibers = 8
        num_circles = 3
        obj = gen.generate(
            params={"num_fibers": num_fibers, "num_circles": num_circles},
            fiber_points=_TEST_FIBER_POINTS,
        )
        obj.validate_or_raise()
        assert obj.curves is not None
        assert len(obj.curves) == num_fibers * num_circles

    def test_each_fiber_is_closed(
        self, gen: HopfFibrationGenerator,
    ) -> None:
        """Every fiber curve is a closed loop."""
        obj = gen.generate(
            params={"num_fibers": 4, "num_circles": 2},
            fiber_points=_TEST_FIBER_POINTS,
        )
        assert obj.curves is not None
        for curve in obj.curves:
            assert curve.closed is True

    def test_varying_num_circles(
        self, gen: HopfFibrationGenerator,
    ) -> None:
        """Different num_circles values produce different total curves."""
        results = {}
        for nc in [2, 4, 6]:
            obj = gen.generate(
                params={"num_fibers": 4, "num_circles": nc},
                fiber_points=_TEST_FIBER_POINTS,
            )
            assert obj.curves is not None
            results[nc] = len(obj.curves)

        assert results[2] < results[4] < results[6]

    def test_different_circles_produce_different_torus_configs(
        self, gen: HopfFibrationGenerator,
    ) -> None:
        """Different num_circles produce geometrically distinct outputs."""
        obj_a = gen.generate(
            params={"num_fibers": 8, "num_circles": 2},
            fiber_points=_TEST_FIBER_POINTS,
        )
        obj_b = gen.generate(
            params={"num_fibers": 8, "num_circles": 5},
            fiber_points=_TEST_FIBER_POINTS,
        )
        assert obj_a.curves is not None and obj_b.curves is not None
        # Different number of curves
        assert len(obj_a.curves) != len(obj_b.curves)

        # Bounding boxes differ — fibers at different latitudes produce
        # tori with different spatial extents
        assert obj_a.bounding_box is not None and obj_b.bounding_box is not None
        size_a = np.array(obj_a.bounding_box.size)
        size_b = np.array(obj_b.bounding_box.size)
        assert not np.allclose(size_a, size_b, atol=1e-6)


class TestHopfFibrationRegistration:
    """Tests for registration and rendering."""

    def test_registers_and_renders(
        self, gen: HopfFibrationGenerator,
    ) -> None:
        """Generator registers and produces valid renderable output."""
        gen_cls = get_generator("hopf_fibration")
        assert gen_cls is HopfFibrationGenerator

        obj = gen.generate(fiber_points=_TEST_FIBER_POINTS)
        obj.validate_or_raise()
        assert obj.curves is not None
        assert len(obj.curves) > 0

    def test_alias_discoverable(self) -> None:
        """Alias 'hopf' is discoverable."""
        assert get_generator("hopf") is HopfFibrationGenerator

    def test_default_representation_is_tube(
        self, gen: HopfFibrationGenerator,
    ) -> None:
        """Default representation is TUBE with thin radius."""
        rep = gen.get_default_representation()
        assert rep.type == RepresentationType.TUBE
        assert rep.tube_radius is not None
        assert rep.tube_radius < 0.1

    def test_metadata_recorded(
        self, gen: HopfFibrationGenerator,
    ) -> None:
        """Generator metadata is recorded correctly."""
        obj = gen.generate(fiber_points=_TEST_FIBER_POINTS)
        assert obj.generator_name == "hopf_fibration"
        assert obj.category == "parametric"


class TestHopfFibrationDeterminism:
    """Tests for deterministic output."""

    def test_deterministic_across_seeds(
        self, gen: HopfFibrationGenerator,
    ) -> None:
        """Output is identical regardless of seed (no randomness)."""
        obj1 = gen.generate(seed=42, fiber_points=_TEST_FIBER_POINTS)
        obj2 = gen.generate(seed=99, fiber_points=_TEST_FIBER_POINTS)
        assert obj1.curves is not None and obj2.curves is not None
        assert len(obj1.curves) == len(obj2.curves)
        for c1, c2 in zip(obj1.curves, obj2.curves):
            np.testing.assert_array_equal(c1.points, c2.points)


class TestHopfFibrationValidation:
    """Tests for parameter validation and data quality."""

    def test_no_nan_values(self, gen: HopfFibrationGenerator) -> None:
        """No NaN values in output curves."""
        obj = gen.generate(fiber_points=_TEST_FIBER_POINTS)
        obj.validate_or_raise()
        assert obj.curves is not None
        for curve in obj.curves:
            assert not np.any(np.isnan(curve.points))

    def test_all_points_finite(self, gen: HopfFibrationGenerator) -> None:
        """All output points are finite (no infinity)."""
        obj = gen.generate(fiber_points=_TEST_FIBER_POINTS)
        assert obj.curves is not None
        for curve in obj.curves:
            assert np.all(np.isfinite(curve.points))

    def test_bounding_box_finite(
        self, gen: HopfFibrationGenerator,
    ) -> None:
        """Bounding box has finite corners."""
        obj = gen.generate(fiber_points=_TEST_FIBER_POINTS)
        assert obj.bounding_box is not None
        min_c = np.array(obj.bounding_box.min_corner)
        max_c = np.array(obj.bounding_box.max_corner)
        assert np.all(np.isfinite(min_c))
        assert np.all(np.isfinite(max_c))

    def test_invalid_num_fibers_raises(
        self, gen: HopfFibrationGenerator,
    ) -> None:
        """num_fibers < 1 raises ValueError."""
        with pytest.raises(ValueError, match="num_fibers must be"):
            gen.generate(params={"num_fibers": 0})

    def test_invalid_num_circles_raises(
        self, gen: HopfFibrationGenerator,
    ) -> None:
        """num_circles < 1 raises ValueError."""
        with pytest.raises(ValueError, match="num_circles must be"):
            gen.generate(params={"num_circles": 0})

    def test_invalid_fiber_points_raises(
        self, gen: HopfFibrationGenerator,
    ) -> None:
        """fiber_points < 4 raises ValueError."""
        with pytest.raises(ValueError, match="fiber_points must be"):
            gen.generate(fiber_points=2)

    def test_invalid_projection_point_length_raises(
        self, gen: HopfFibrationGenerator,
    ) -> None:
        """projection_point with wrong length raises ValueError."""
        with pytest.raises(ValueError, match="projection_point must have exactly 4"):
            gen.generate(params={"projection_point": [0, 0, 0]})

    def test_projection_near_pole_clamps_extreme_points(
        self, gen: HopfFibrationGenerator,
    ) -> None:
        """Points near the projection pole are clamped, not infinite."""
        # Use projection_point with pz=0, which puts the pole inside S³
        obj = gen.generate(
            params={
                "num_fibers": 4,
                "num_circles": 2,
                "projection_point": [0.0, 0.0, 0.0, 0.0],
            },
            fiber_points=_TEST_FIBER_POINTS,
        )
        assert obj.curves is not None
        for curve in obj.curves:
            assert np.all(np.isfinite(curve.points))
            assert np.all(np.abs(curve.points) <= 1e6)
