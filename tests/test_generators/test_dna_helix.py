"""Tests for the DNA double helix generator."""

import numpy as np
import pytest

from mathviz.core.generator import clear_registry, get_generator, register
from mathviz.core.representation import RepresentationType
from mathviz.generators.parametric.dna_helix import DNAHelixGenerator


@pytest.fixture(autouse=True)
def _clean_registry():
    """Reset registry and register DNA helix generator for each test."""
    clear_registry(suppress_discovery=True)
    register(DNAHelixGenerator)
    yield
    clear_registry(suppress_discovery=True)


@pytest.fixture
def gen() -> DNAHelixGenerator:
    """Return a DNAHelixGenerator instance."""
    return DNAHelixGenerator()


_TEST_CURVE_POINTS = 128


class TestDNAHelixStructure:
    """Tests for helix structure — two backbones plus rungs."""

    def test_produces_two_helix_curves_plus_rungs(
        self, gen: DNAHelixGenerator,
    ) -> None:
        """Output contains two helix backbone curves and rung curves."""
        obj = gen.generate(
            params={"turns": 2, "base_pairs_per_turn": 5},
            curve_points=_TEST_CURVE_POINTS,
        )
        obj.validate_or_raise()
        assert obj.curves is not None

        expected_rungs = 2 * 5  # turns * base_pairs_per_turn
        expected_total = 2 + expected_rungs  # 2 helices + rungs
        assert len(obj.curves) == expected_total

    def test_number_of_rungs_equals_turns_times_bpt(
        self, gen: DNAHelixGenerator,
    ) -> None:
        """Number of rung curves = turns * base_pairs_per_turn."""
        for turns, bpt in [(1, 10), (2, 5), (3, 10), (4, 8)]:
            obj = gen.generate(
                params={"turns": turns, "base_pairs_per_turn": bpt},
                curve_points=_TEST_CURVE_POINTS,
            )
            obj.validate_or_raise()
            assert obj.curves is not None
            expected_rungs = turns * bpt
            # Total curves = 2 backbones + rungs
            assert len(obj.curves) == 2 + expected_rungs


class TestDNAHelixLength:
    """Tests for helix length scaling with turns."""

    def test_more_turns_produce_longer_structure(
        self, gen: DNAHelixGenerator,
    ) -> None:
        """More turns produce a taller structure along z-axis."""
        obj_short = gen.generate(
            params={"turns": 2}, curve_points=_TEST_CURVE_POINTS,
        )
        obj_long = gen.generate(
            params={"turns": 5}, curve_points=_TEST_CURVE_POINTS,
        )
        assert obj_short.bounding_box is not None
        assert obj_long.bounding_box is not None

        z_extent_short = (
            obj_short.bounding_box.max_corner[2]
            - obj_short.bounding_box.min_corner[2]
        )
        z_extent_long = (
            obj_long.bounding_box.max_corner[2]
            - obj_long.bounding_box.min_corner[2]
        )
        assert z_extent_long > z_extent_short


class TestDNAHelixRegistration:
    """Tests for registration and rendering."""

    def test_registers_and_renders(self, gen: DNAHelixGenerator) -> None:
        """Generator registers and produces valid renderable output."""
        gen_cls = get_generator("dna_helix")
        assert gen_cls is DNAHelixGenerator

        obj = gen.generate(curve_points=_TEST_CURVE_POINTS)
        obj.validate_or_raise()
        assert obj.curves is not None
        assert len(obj.curves) > 0

    def test_alias_discoverable(self) -> None:
        """Aliases dna and double_helix are discoverable."""
        assert get_generator("dna") is DNAHelixGenerator
        assert get_generator("double_helix") is DNAHelixGenerator

    def test_default_representation_is_tube(
        self, gen: DNAHelixGenerator,
    ) -> None:
        """Default representation is TUBE."""
        rep = gen.get_default_representation()
        assert rep.type == RepresentationType.TUBE

    def test_metadata_recorded(self, gen: DNAHelixGenerator) -> None:
        """Generator metadata is recorded correctly."""
        obj = gen.generate(curve_points=_TEST_CURVE_POINTS)
        assert obj.generator_name == "dna_helix"
        assert obj.category == "parametric"


class TestDNAHelixDeterminism:
    """Tests for deterministic output."""

    def test_deterministic_across_seeds(self, gen: DNAHelixGenerator) -> None:
        """Output is identical regardless of seed (no randomness involved)."""
        obj1 = gen.generate(seed=42, curve_points=_TEST_CURVE_POINTS)
        obj2 = gen.generate(seed=99, curve_points=_TEST_CURVE_POINTS)
        assert obj1.curves is not None and obj2.curves is not None
        assert len(obj1.curves) == len(obj2.curves)
        for c1, c2 in zip(obj1.curves, obj2.curves):
            np.testing.assert_array_equal(c1.points, c2.points)


class TestDNAHelixValidation:
    """Tests for parameter validation and data quality."""

    def test_no_nan_values(self, gen: DNAHelixGenerator) -> None:
        """No NaN values in output curves."""
        obj = gen.generate(curve_points=_TEST_CURVE_POINTS)
        obj.validate_or_raise()
        assert obj.curves is not None
        for curve in obj.curves:
            assert not np.any(np.isnan(curve.points))

    def test_bounding_box_finite(self, gen: DNAHelixGenerator) -> None:
        """Bounding box has finite corners."""
        obj = gen.generate(curve_points=_TEST_CURVE_POINTS)
        assert obj.bounding_box is not None
        min_c = np.array(obj.bounding_box.min_corner)
        max_c = np.array(obj.bounding_box.max_corner)
        assert np.all(np.isfinite(min_c))
        assert np.all(np.isfinite(max_c))

    def test_invalid_turns_raises(self, gen: DNAHelixGenerator) -> None:
        """Too-small turns raises ValueError."""
        with pytest.raises(ValueError, match="turns must be"):
            gen.generate(params={"turns": 0.1})

    def test_invalid_radius_raises(self, gen: DNAHelixGenerator) -> None:
        """Too-small radius raises ValueError."""
        with pytest.raises(ValueError, match="radius must be"):
            gen.generate(params={"radius": 0.0})
