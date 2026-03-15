"""Tests for the gravitational lensing grid generator."""

import numpy as np
import pytest

from mathviz.core.generator import clear_registry, get_generator, register
from mathviz.core.representation import RepresentationType
from mathviz.generators.physics.gravitational_lensing import (
    GravitationalLensingGenerator,
)


@pytest.fixture(autouse=True)
def _clean_registry():
    """Reset registry and register gravitational lensing generator."""
    clear_registry(suppress_discovery=True)
    register(GravitationalLensingGenerator)
    yield
    clear_registry(suppress_discovery=True)


@pytest.fixture
def gen() -> GravitationalLensingGenerator:
    """Return a GravitationalLensingGenerator instance."""
    return GravitationalLensingGenerator()


_TEST_GRID_POINTS = 50
_TEST_GRID_LINES = 10


class TestDeflection:
    """Tests for deflection behavior."""

    def test_near_center_more_deflected(
        self, gen: GravitationalLensingGenerator
    ) -> None:
        """Grid lines near center are more deflected than far ones."""
        obj = gen.generate(
            params={
                "mass": 1.0,
                "grid_lines": _TEST_GRID_LINES,
                "grid_extent": 5.0,
            },
            grid_points=_TEST_GRID_POINTS,
        )
        obj.validate_or_raise()
        assert obj.curves is not None

        # Compare a line near center vs a line far from center
        # Lines are: first grid_lines are horizontal (constant y),
        # next grid_lines are vertical (constant x)
        # line_positions = linspace(-5, 5, 10)
        # Index 4 or 5 is near center, index 0 is at edge
        near_center_line = obj.curves[_TEST_GRID_LINES // 2]
        far_line = obj.curves[0]

        # Z values encode deflection magnitude; near-center should
        # have higher average z (more deflection)
        near_z_mean = np.mean(near_center_line.points[:, 2])
        far_z_mean = np.mean(far_line.points[:, 2])
        assert near_z_mean > far_z_mean, (
            f"Near-center deflection ({near_z_mean:.4f}) should exceed "
            f"far deflection ({far_z_mean:.4f})"
        )

    def test_mass_zero_flat_grid(
        self, gen: GravitationalLensingGenerator
    ) -> None:
        """mass=0 produces a flat grid (no deflection)."""
        obj = gen.generate(
            params={"mass": 0.0, "grid_lines": _TEST_GRID_LINES},
            grid_points=_TEST_GRID_POINTS,
        )
        obj.validate_or_raise()
        assert obj.curves is not None

        for curve in obj.curves:
            # With mass=0, z should be 0 everywhere
            np.testing.assert_allclose(
                curve.points[:, 2], 0.0, atol=1e-12,
                err_msg="mass=0 should produce flat grid with z=0",
            )

    def test_higher_mass_more_deflection(
        self, gen: GravitationalLensingGenerator
    ) -> None:
        """Higher mass produces more deflection."""
        obj_low = gen.generate(
            params={"mass": 0.5, "grid_lines": _TEST_GRID_LINES},
            grid_points=_TEST_GRID_POINTS,
        )
        obj_high = gen.generate(
            params={"mass": 2.0, "grid_lines": _TEST_GRID_LINES},
            grid_points=_TEST_GRID_POINTS,
        )
        assert obj_low.curves is not None and obj_high.curves is not None

        low_max_z = max(
            np.max(c.points[:, 2]) for c in obj_low.curves
        )
        high_max_z = max(
            np.max(c.points[:, 2]) for c in obj_high.curves
        )
        assert high_max_z > low_max_z


class TestNoNaN:
    """Tests for NaN-free output."""

    def test_no_nan_in_curves(
        self, gen: GravitationalLensingGenerator
    ) -> None:
        """No NaN values in output curves."""
        obj = gen.generate(
            params={"grid_lines": _TEST_GRID_LINES},
            grid_points=_TEST_GRID_POINTS,
        )
        obj.validate_or_raise()
        assert obj.curves is not None
        for curve in obj.curves:
            assert not np.any(np.isnan(curve.points))


class TestRegistration:
    """Tests for registration and rendering."""

    def test_registers_and_renders(
        self, gen: GravitationalLensingGenerator
    ) -> None:
        """Generator registers and produces valid renderable output."""
        gen_cls = get_generator("gravitational_lensing")
        assert gen_cls is GravitationalLensingGenerator

        obj = gen.generate(grid_points=_TEST_GRID_POINTS)
        obj.validate_or_raise()
        assert obj.curves is not None
        assert len(obj.curves) > 0

    def test_alias_grav_lens(self) -> None:
        """Alias grav_lens is discoverable."""
        gen_cls = get_generator("grav_lens")
        assert gen_cls is GravitationalLensingGenerator

    def test_alias_spacetime_grid(self) -> None:
        """Alias spacetime_grid is discoverable."""
        gen_cls = get_generator("spacetime_grid")
        assert gen_cls is GravitationalLensingGenerator

    def test_default_representation_is_tube(
        self, gen: GravitationalLensingGenerator
    ) -> None:
        """Default representation is TUBE."""
        rep = gen.get_default_representation()
        assert rep.type == RepresentationType.TUBE


class TestMetadata:
    """Tests for metadata and determinism."""

    def test_metadata_recorded(
        self, gen: GravitationalLensingGenerator
    ) -> None:
        """Generator metadata is recorded correctly."""
        obj = gen.generate(grid_points=_TEST_GRID_POINTS)
        assert obj.generator_name == "gravitational_lensing"
        assert obj.category == "physics"

    def test_bounding_box_finite(
        self, gen: GravitationalLensingGenerator
    ) -> None:
        """Bounding box is finite."""
        obj = gen.generate(
            params={"grid_lines": _TEST_GRID_LINES},
            grid_points=_TEST_GRID_POINTS,
        )
        assert obj.bounding_box is not None
        min_c = np.array(obj.bounding_box.min_corner)
        max_c = np.array(obj.bounding_box.max_corner)
        assert np.all(np.isfinite(min_c))
        assert np.all(np.isfinite(max_c))

    def test_deterministic_same_seed(
        self, gen: GravitationalLensingGenerator
    ) -> None:
        """Same seed produces identical output."""
        obj1 = gen.generate(
            seed=42, params={"grid_lines": _TEST_GRID_LINES},
            grid_points=_TEST_GRID_POINTS,
        )
        obj2 = gen.generate(
            seed=42, params={"grid_lines": _TEST_GRID_LINES},
            grid_points=_TEST_GRID_POINTS,
        )
        assert obj1.curves is not None and obj2.curves is not None
        assert len(obj1.curves) == len(obj2.curves)
        for c1, c2 in zip(obj1.curves, obj2.curves):
            np.testing.assert_array_equal(c1.points, c2.points)

    def test_grid_lines_controls_curve_count(
        self, gen: GravitationalLensingGenerator
    ) -> None:
        """grid_lines controls the number of output curves."""
        for grid_lines in (4, 8, 12):
            obj = gen.generate(
                params={"grid_lines": grid_lines},
                grid_points=_TEST_GRID_POINTS,
            )
            # Total curves = grid_lines (horizontal) + grid_lines (vertical)
            assert obj.curves is not None
            assert len(obj.curves) == 2 * grid_lines


class TestValidation:
    """Tests for parameter validation."""

    def test_negative_mass_raises(
        self, gen: GravitationalLensingGenerator
    ) -> None:
        """Negative mass raises ValueError."""
        with pytest.raises(ValueError, match="mass must be >= 0"):
            gen.generate(params={"mass": -1.0})

    def test_grid_lines_too_low_raises(
        self, gen: GravitationalLensingGenerator
    ) -> None:
        """grid_lines below minimum raises ValueError."""
        with pytest.raises(ValueError, match="grid_lines must be >="):
            gen.generate(params={"grid_lines": 1})

    def test_grid_extent_too_small_raises(
        self, gen: GravitationalLensingGenerator
    ) -> None:
        """grid_extent below minimum raises ValueError."""
        with pytest.raises(ValueError, match="grid_extent must be >="):
            gen.generate(params={"grid_extent": 0.0})

    def test_grid_points_too_high_raises(
        self, gen: GravitationalLensingGenerator
    ) -> None:
        """grid_points above maximum raises ValueError."""
        with pytest.raises(ValueError, match="grid_points must be <="):
            gen.generate(grid_points=10_000_001)
