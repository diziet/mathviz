"""Tests for the magnetic field line generator."""

import numpy as np
import pytest

from mathviz.core.generator import clear_registry, get_generator, register
from mathviz.core.representation import RepresentationType
from mathviz.generators.physics.magnetic_field import MagneticFieldGenerator


@pytest.fixture(autouse=True)
def _clean_registry():
    """Reset registry and register magnetic field generator for each test."""
    clear_registry(suppress_discovery=True)
    register(MagneticFieldGenerator)
    yield
    clear_registry(suppress_discovery=True)


@pytest.fixture
def gen() -> MagneticFieldGenerator:
    """Return a MagneticFieldGenerator instance."""
    return MagneticFieldGenerator()


_TEST_LINE_POINTS = 200
_TEST_NUM_LINES = 8


def _assert_bbox_finite(obj) -> None:
    """Assert bounding box exists and has finite corners."""
    assert obj.bounding_box is not None
    min_c = np.array(obj.bounding_box.min_corner)
    max_c = np.array(obj.bounding_box.max_corner)
    assert np.all(np.isfinite(min_c))
    assert np.all(np.isfinite(max_c))


class TestMagneticFieldDipole:
    """Tests for dipole field configuration."""

    def test_dipole_lines_loop_between_poles(
        self, gen: MagneticFieldGenerator
    ) -> None:
        """Dipole field lines loop from one pole to the other."""
        obj = gen.generate(
            params={"field_type": "dipole", "num_lines": _TEST_NUM_LINES},
            line_points=_TEST_LINE_POINTS,
        )
        obj.validate_or_raise()

        assert obj.curves is not None
        assert len(obj.curves) > 0

        # For a dipole, field lines should have points above and below
        # the equatorial plane (z > 0 and z < 0), showing they loop
        for curve in obj.curves:
            z_vals = curve.points[:, 2]
            has_positive_z = np.any(z_vals > 0)
            has_negative_z = np.any(z_vals < 0)
            assert has_positive_z and has_negative_z, (
                "Dipole field line should span both hemispheres"
            )


class TestMagneticFieldQuadrupole:
    """Tests for quadrupole field configuration."""

    def test_quadrupole_more_complex_pattern(
        self, gen: MagneticFieldGenerator
    ) -> None:
        """Quadrupole produces a more complex line pattern than dipole."""
        dipole_obj = gen.generate(
            params={"field_type": "dipole", "num_lines": _TEST_NUM_LINES},
            seed=42,
            line_points=_TEST_LINE_POINTS,
        )
        quad_obj = gen.generate(
            params={"field_type": "quadrupole", "num_lines": _TEST_NUM_LINES},
            seed=42,
            line_points=_TEST_LINE_POINTS,
        )

        assert dipole_obj.curves is not None
        assert quad_obj.curves is not None

        # Quadrupole lines should differ from dipole lines
        # Compare the spatial extent (bounding boxes should differ)
        dipole_bbox = dipole_obj.bounding_box
        quad_bbox = quad_obj.bounding_box
        assert dipole_bbox is not None and quad_bbox is not None

        dipole_extent = np.array(dipole_bbox.max_corner) - np.array(
            dipole_bbox.min_corner
        )
        quad_extent = np.array(quad_bbox.max_corner) - np.array(
            quad_bbox.min_corner
        )

        # Patterns should be meaningfully different
        assert not np.allclose(dipole_extent, quad_extent, atol=0.1), (
            "Quadrupole and dipole should produce different spatial extents"
        )


class TestMagneticFieldNoNaN:
    """Tests for NaN-free output across field types."""

    @pytest.mark.parametrize("field_type", ["dipole", "quadrupole"])
    def test_no_nan(
        self, gen: MagneticFieldGenerator, field_type: str
    ) -> None:
        """No NaN values in output for either field type."""
        obj = gen.generate(
            params={"field_type": field_type, "num_lines": _TEST_NUM_LINES},
            line_points=_TEST_LINE_POINTS,
        )
        obj.validate_or_raise()
        assert obj.curves is not None
        for curve in obj.curves:
            assert not np.any(np.isnan(curve.points))


class TestMagneticFieldNumLines:
    """Tests for num_lines parameter."""

    def test_num_lines_controls_curve_count(
        self, gen: MagneticFieldGenerator
    ) -> None:
        """num_lines controls the number of curves output."""
        for num_lines in (4, 8, 16):
            obj = gen.generate(
                params={"num_lines": num_lines},
                line_points=_TEST_LINE_POINTS,
            )
            obj.validate_or_raise()
            assert obj.curves is not None
            assert len(obj.curves) == num_lines


class TestMagneticFieldGeneral:
    """General tests for the magnetic field generator."""

    def test_registers_and_renders(self, gen: MagneticFieldGenerator) -> None:
        """Generator registers and produces valid renderable output."""
        gen_cls = get_generator("magnetic_field")
        assert gen_cls is MagneticFieldGenerator

        obj = gen.generate(line_points=_TEST_LINE_POINTS)
        obj.validate_or_raise()
        assert obj.curves is not None
        assert len(obj.curves) > 0

    def test_default_representation_is_tube(
        self, gen: MagneticFieldGenerator
    ) -> None:
        """Default representation is TUBE."""
        rep = gen.get_default_representation()
        assert rep.type == RepresentationType.TUBE

    def test_metadata_recorded(self, gen: MagneticFieldGenerator) -> None:
        """Generator metadata is recorded correctly."""
        obj = gen.generate(line_points=_TEST_LINE_POINTS)
        assert obj.generator_name == "magnetic_field"
        assert obj.category == "physics"

    def test_bounding_box_finite(self, gen: MagneticFieldGenerator) -> None:
        """Bounding box is finite."""
        obj = gen.generate(
            params={"num_lines": _TEST_NUM_LINES},
            line_points=_TEST_LINE_POINTS,
        )
        _assert_bbox_finite(obj)

    def test_same_seed_deterministic(
        self, gen: MagneticFieldGenerator
    ) -> None:
        """Same seed produces identical output."""
        obj1 = gen.generate(
            seed=42, params={"num_lines": _TEST_NUM_LINES},
            line_points=_TEST_LINE_POINTS,
        )
        obj2 = gen.generate(
            seed=42, params={"num_lines": _TEST_NUM_LINES},
            line_points=_TEST_LINE_POINTS,
        )
        assert obj1.curves is not None and obj2.curves is not None
        assert len(obj1.curves) == len(obj2.curves)
        for c1, c2 in zip(obj1.curves, obj2.curves):
            np.testing.assert_array_equal(c1.points, c2.points)

    def test_alias_discoverable(self) -> None:
        """Alias mag_field is discoverable."""
        gen_cls = get_generator("mag_field")
        assert gen_cls is MagneticFieldGenerator

    def test_invalid_field_type_raises(
        self, gen: MagneticFieldGenerator
    ) -> None:
        """Invalid field_type raises ValueError."""
        with pytest.raises(ValueError, match="field_type must be one of"):
            gen.generate(params={"field_type": "invalid"})

    def test_negative_spread_raises(
        self, gen: MagneticFieldGenerator
    ) -> None:
        """Negative spread raises ValueError."""
        with pytest.raises(ValueError, match="spread must be positive"):
            gen.generate(params={"spread": -1.0})
