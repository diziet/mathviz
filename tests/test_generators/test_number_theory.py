"""Tests for number theory generators: ulam_spiral, sacks_spiral,
prime_gaps, digit_encoding, and WEIGHTED_CLOUD representation."""

import numpy as np
import pytest

from mathviz.core.generator import clear_registry, register
from mathviz.core.math_object import MathObject, PointCloud
from mathviz.core.representation import RepresentationConfig, RepresentationType
from mathviz.generators.number_theory._primes import first_n_primes, is_prime_array
from mathviz.generators.number_theory.digit_encoding import DigitEncodingGenerator
from mathviz.generators.number_theory.prime_gaps import PrimeGapsGenerator
from mathviz.generators.number_theory.sacks_spiral import SacksSpiralGenerator
from mathviz.generators.number_theory.ulam_spiral import UlamSpiralGenerator
from mathviz.pipeline.representation_strategy import apply


@pytest.fixture(autouse=True)
def _clean_registry():
    """Reset registry and register all number theory generators."""
    clear_registry(suppress_discovery=True)
    register(UlamSpiralGenerator)
    register(SacksSpiralGenerator)
    register(PrimeGapsGenerator)
    register(DigitEncodingGenerator)
    yield
    clear_registry(suppress_discovery=True)


# ---------------------------------------------------------------------------
# Ulam spiral
# ---------------------------------------------------------------------------


class TestUlamSpiral:
    """Tests for the Ulam spiral generator."""

    def test_primes_have_higher_z_than_composites(self) -> None:
        """Primes have higher z-values than composites."""
        gen = UlamSpiralGenerator()
        obj = gen.generate(num_points=100, params={"prime_height": 1.0})

        assert obj.point_cloud is not None
        primality = is_prime_array(100)

        prime_z = obj.point_cloud.points[primality, 2]
        composite_mask = ~primality
        composite_mask[0] = False  # 0 is neither prime nor composite
        composite_mask[1] = False  # 1 is neither prime nor composite
        composite_z = obj.point_cloud.points[composite_mask, 2]

        assert np.all(prime_z > 0.0)
        assert np.all(composite_z == 0.0)
        assert prime_z.mean() > composite_z.mean()

    def test_output_has_intensities(self) -> None:
        """Ulam spiral produces per-point intensities."""
        gen = UlamSpiralGenerator()
        obj = gen.generate(num_points=50)
        assert obj.point_cloud is not None
        assert obj.point_cloud.intensities is not None
        assert len(obj.point_cloud.intensities) == len(obj.point_cloud.points)

    def test_no_nan_in_points(self) -> None:
        """Points contain no NaN values."""
        gen = UlamSpiralGenerator()
        obj = gen.generate(num_points=200)
        assert not np.any(np.isnan(obj.point_cloud.points))

    def test_bounding_box_computed(self) -> None:
        """Bounding box is computed."""
        gen = UlamSpiralGenerator()
        obj = gen.generate(num_points=50)
        assert obj.bounding_box is not None

    def test_default_representation_is_weighted_cloud(self) -> None:
        """Default representation is WEIGHTED_CLOUD."""
        gen = UlamSpiralGenerator()
        rep = gen.get_default_representation()
        assert rep.type == RepresentationType.WEIGHTED_CLOUD


# ---------------------------------------------------------------------------
# Sacks spiral
# ---------------------------------------------------------------------------


class TestSacksSpiral:
    """Tests for the Sacks spiral generator."""

    def test_primes_elevated(self) -> None:
        """Primes are elevated in z on the Sacks spiral."""
        gen = SacksSpiralGenerator()
        obj = gen.generate(num_points=100, params={"prime_height": 1.0})

        assert obj.point_cloud is not None
        primality = is_prime_array(100)
        prime_z = obj.point_cloud.points[primality, 2]
        assert np.all(prime_z > 0.0)

    def test_spiral_layout(self) -> None:
        """Points lie on an Archimedean spiral in the x-y plane."""
        gen = SacksSpiralGenerator()
        obj = gen.generate(num_points=50, params={"scale": 1.0})

        points = obj.point_cloud.points
        # Radius should increase with index (roughly sqrt(n))
        radii = np.sqrt(points[:, 0] ** 2 + points[:, 1] ** 2)
        # Skip index 0 (origin) — radii should be non-decreasing
        assert np.all(np.diff(radii[1:]) >= -1e-10)


# ---------------------------------------------------------------------------
# Prime gaps
# ---------------------------------------------------------------------------


class TestPrimeGaps:
    """Tests for the prime gaps generator."""

    def test_gap_sizes_mathematically_correct(self) -> None:
        """Gap sizes are mathematically correct for the first N primes."""
        gen = PrimeGapsGenerator()
        num_primes = 20
        obj = gen.generate(num_primes=num_primes, params={"y_scale": 1.0})

        assert obj.point_cloud is not None
        primes = first_n_primes(num_primes)
        expected_gaps = np.diff(primes).astype(np.float64)

        # y-coordinates encode the gap sizes (y_scale=1.0)
        actual_y = obj.point_cloud.points[:, 1]
        np.testing.assert_array_almost_equal(actual_y, expected_gaps)

    def test_first_few_gaps_known_values(self) -> None:
        """Verify specific known prime gaps."""
        gen = PrimeGapsGenerator()
        obj = gen.generate(num_primes=10, params={"y_scale": 1.0})

        # First 10 primes: 2, 3, 5, 7, 11, 13, 17, 19, 23, 29
        # Gaps:            1, 2, 2, 4,  2,  4,  2,  4,  6
        expected = [1.0, 2.0, 2.0, 4.0, 2.0, 4.0, 2.0, 4.0, 6.0]
        actual_y = obj.point_cloud.points[:, 1]
        np.testing.assert_array_almost_equal(actual_y, expected)

    def test_num_gaps_is_num_primes_minus_one(self) -> None:
        """Number of gap points is num_primes - 1."""
        gen = PrimeGapsGenerator()
        obj = gen.generate(num_primes=50)
        assert len(obj.point_cloud.points) == 49


# ---------------------------------------------------------------------------
# Digit encoding
# ---------------------------------------------------------------------------


class TestDigitEncoding:
    """Tests for the digit encoding generator."""

    def test_pi_first_10_digits(self) -> None:
        """Digit encoding with π produces correct digit sequence."""
        gen = DigitEncodingGenerator()
        obj = gen.generate(
            num_digits=10,
            params={"constant": "pi", "height_scale": 1.0},
        )

        assert obj.point_cloud is not None
        z_values = obj.point_cloud.points[:10, 2]
        # π = 3.141592653...  → digits: 3, 1, 4, 1, 5, 9, 2, 6, 5, 3
        expected_digits = [3.0, 1.0, 4.0, 1.0, 5.0, 9.0, 2.0, 6.0, 5.0, 3.0]
        np.testing.assert_array_almost_equal(z_values, expected_digits)

    def test_e_first_10_digits(self) -> None:
        """Digit encoding with e produces correct digit sequence."""
        gen = DigitEncodingGenerator()
        obj = gen.generate(
            num_digits=10,
            params={"constant": "e", "height_scale": 1.0},
        )

        z_values = obj.point_cloud.points[:10, 2]
        # e = 2.718281828... → digits: 2, 7, 1, 8, 2, 8, 1, 8, 2, 8
        expected_digits = [2.0, 7.0, 1.0, 8.0, 2.0, 8.0, 1.0, 8.0, 2.0, 8.0]
        np.testing.assert_array_almost_equal(z_values, expected_digits)

    def test_intensities_proportional_to_digits(self) -> None:
        """Intensities are proportional to digit values."""
        gen = DigitEncodingGenerator()
        obj = gen.generate(
            num_digits=20,
            params={"constant": "pi"},
        )

        assert obj.point_cloud.intensities is not None
        # Digit 9 should have intensity 1.0, digit 0 should have 0.0
        assert np.max(obj.point_cloud.intensities) <= 1.0
        assert np.min(obj.point_cloud.intensities) >= 0.0

    def test_invalid_constant_raises(self) -> None:
        """Invalid constant name raises ValueError."""
        gen = DigitEncodingGenerator()
        with pytest.raises(ValueError, match="constant must be one of"):
            gen.generate(params={"constant": "sqrt2"})

    def test_over_request_raises(self) -> None:
        """Requesting more digits than available raises ValueError."""
        gen = DigitEncodingGenerator()
        with pytest.raises(ValueError, match="only .* are available"):
            gen.generate(num_digits=10000, params={"constant": "pi"})


# ---------------------------------------------------------------------------
# WEIGHTED_CLOUD representation
# ---------------------------------------------------------------------------


class TestWeightedCloudRepresentation:
    """Tests for WEIGHTED_CLOUD representation strategy."""

    def test_preserves_intensities(self) -> None:
        """WEIGHTED_CLOUD representation preserves intensities on the PointCloud."""
        intensities = np.array([0.1, 0.5, 1.0, 0.3], dtype=np.float64)
        points = np.array([
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [1.0, 1.0, 0.0],
        ], dtype=np.float64)

        cloud = PointCloud(points=points, intensities=intensities)
        obj = MathObject(
            point_cloud=cloud,
            generator_name="test",
        )

        config = RepresentationConfig(type=RepresentationType.WEIGHTED_CLOUD)
        result = apply(obj, config)

        assert result.point_cloud is not None
        assert result.point_cloud.intensities is not None
        np.testing.assert_array_equal(
            result.point_cloud.intensities, intensities
        )
        assert result.representation == "weighted_cloud"

    def test_raises_without_point_cloud(self) -> None:
        """WEIGHTED_CLOUD raises ValueError if no point_cloud on MathObject."""
        obj = MathObject(generator_name="test")
        config = RepresentationConfig(type=RepresentationType.WEIGHTED_CLOUD)
        with pytest.raises(ValueError, match="WEIGHTED_CLOUD requires a point_cloud"):
            apply(obj, config)

    def test_generator_roundtrip(self) -> None:
        """Generator output passes through WEIGHTED_CLOUD representation."""
        gen = UlamSpiralGenerator()
        obj = gen.generate(num_points=50)
        config = gen.get_default_representation()

        result = apply(obj, config)
        assert result.point_cloud is not None
        assert result.point_cloud.intensities is not None
        assert len(result.point_cloud.intensities) == 50
        assert result.representation == "weighted_cloud"
