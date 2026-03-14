"""Tests for physics generators: kepler_orbit, nbody, planetary_positions."""

import numpy as np
import pytest

from mathviz.core.generator import clear_registry, get_generator, register
from mathviz.core.representation import RepresentationType
from mathviz.generators.physics.kepler_orbit import KeplerOrbitGenerator
from mathviz.generators.physics.nbody import NBodyGenerator
from mathviz.generators.physics.planetary_positions import (
    PlanetaryPositionsGenerator,
)


@pytest.fixture(autouse=True)
def _clean_registry():
    """Reset registry and register physics generators for each test."""
    clear_registry(suppress_discovery=True)
    register(KeplerOrbitGenerator)
    register(NBodyGenerator)
    register(PlanetaryPositionsGenerator)
    yield
    clear_registry(suppress_discovery=True)


@pytest.fixture
def kepler() -> KeplerOrbitGenerator:
    """Return a KeplerOrbitGenerator instance."""
    return KeplerOrbitGenerator()


@pytest.fixture
def nbody() -> NBodyGenerator:
    """Return an NBodyGenerator instance."""
    return NBodyGenerator()


@pytest.fixture
def planetary() -> PlanetaryPositionsGenerator:
    """Return a PlanetaryPositionsGenerator instance."""
    return PlanetaryPositionsGenerator()


_TEST_CURVE_POINTS = 128
_TEST_STEPS = 1000


# ---------------------------------------------------------------------------
# Kepler orbit tests
# ---------------------------------------------------------------------------


class TestKeplerOrbit:
    """Tests for the Kepler orbit generator."""

    def test_circular_orbit_eccentricity_zero(
        self, kepler: KeplerOrbitGenerator
    ) -> None:
        """Eccentricity=0 produces a circular curve (constant radius)."""
        obj = kepler.generate(
            params={"eccentricity": 0.0, "semi_major_axis": 2.0},
            curve_points=_TEST_CURVE_POINTS,
        )
        obj.validate_or_raise()

        assert obj.curves is not None
        assert len(obj.curves) == 1

        points = obj.curves[0].points
        radii = np.linalg.norm(points, axis=1)
        assert np.allclose(radii, 2.0, atol=1e-10), (
            f"Expected constant radius 2.0, got min={radii.min():.6f} "
            f"max={radii.max():.6f}"
        )

    def test_orbit_is_closed(self, kepler: KeplerOrbitGenerator) -> None:
        """Kepler orbit curve is marked as closed."""
        obj = kepler.generate(curve_points=_TEST_CURVE_POINTS)
        assert obj.curves is not None
        assert obj.curves[0].closed is True

    def test_default_produces_valid_output(
        self, kepler: KeplerOrbitGenerator
    ) -> None:
        """Default params produce valid MathObject with no NaN."""
        obj = kepler.generate(curve_points=_TEST_CURVE_POINTS)
        obj.validate_or_raise()
        assert not np.any(np.isnan(obj.curves[0].points))

    def test_bounding_box_finite(
        self, kepler: KeplerOrbitGenerator
    ) -> None:
        """Bounding box is finite and non-degenerate."""
        obj = kepler.generate(curve_points=_TEST_CURVE_POINTS)
        assert obj.bounding_box is not None
        min_c = np.array(obj.bounding_box.min_corner)
        max_c = np.array(obj.bounding_box.max_corner)
        assert np.all(np.isfinite(min_c))
        assert np.all(np.isfinite(max_c))

    def test_negative_semi_major_axis_raises(
        self, kepler: KeplerOrbitGenerator
    ) -> None:
        """Negative semi_major_axis raises ValueError."""
        with pytest.raises(ValueError, match="semi_major_axis must be positive"):
            kepler.generate(params={"semi_major_axis": -1.0})

    def test_eccentricity_out_of_range_raises(
        self, kepler: KeplerOrbitGenerator
    ) -> None:
        """Eccentricity >= 1 raises ValueError."""
        with pytest.raises(ValueError, match="eccentricity must be in"):
            kepler.generate(params={"eccentricity": 1.0})

    def test_metadata_recorded(
        self, kepler: KeplerOrbitGenerator
    ) -> None:
        """Generator metadata is recorded correctly."""
        obj = kepler.generate(curve_points=_TEST_CURVE_POINTS)
        assert obj.generator_name == "kepler_orbit"
        assert obj.category == "physics"

    def test_default_representation_is_tube(
        self, kepler: KeplerOrbitGenerator
    ) -> None:
        """Default representation is TUBE."""
        rep = kepler.get_default_representation()
        assert rep.type == RepresentationType.TUBE

    def test_registry_discoverable(self) -> None:
        """Kepler orbit is discoverable via the registry."""
        gen_cls = get_generator("kepler_orbit")
        assert gen_cls is KeplerOrbitGenerator


# ---------------------------------------------------------------------------
# N-body tests
# ---------------------------------------------------------------------------


class TestNBody:
    """Tests for the N-body generator."""

    def test_two_bodies_produces_orbital_curves(
        self, nbody: NBodyGenerator
    ) -> None:
        """Two-body simulation produces 2 recognizable curves."""
        obj = nbody.generate(
            params={"num_bodies": 2},
            integration_steps=_TEST_STEPS,
        )
        obj.validate_or_raise()

        assert obj.curves is not None
        assert len(obj.curves) == 2
        for curve in obj.curves:
            assert len(curve.points) == _TEST_STEPS
            assert not np.any(np.isnan(curve.points))

    def test_same_seed_deterministic(self, nbody: NBodyGenerator) -> None:
        """Same seed produces identical trajectories."""
        obj1 = nbody.generate(
            seed=42, params={"num_bodies": 2}, integration_steps=_TEST_STEPS
        )
        obj2 = nbody.generate(
            seed=42, params={"num_bodies": 2}, integration_steps=_TEST_STEPS
        )

        assert obj1.curves is not None and obj2.curves is not None
        for c1, c2 in zip(obj1.curves, obj2.curves):
            np.testing.assert_array_equal(c1.points, c2.points)

    def test_different_seeds_differ(self, nbody: NBodyGenerator) -> None:
        """Different seeds produce different trajectories."""
        obj1 = nbody.generate(
            seed=1, params={"num_bodies": 2}, integration_steps=_TEST_STEPS
        )
        obj2 = nbody.generate(
            seed=2, params={"num_bodies": 2}, integration_steps=_TEST_STEPS
        )

        assert obj1.curves is not None and obj2.curves is not None
        assert not np.allclose(obj1.curves[0].points, obj2.curves[0].points)

    def test_default_produces_three_curves(
        self, nbody: NBodyGenerator
    ) -> None:
        """Default params produce 3 curves (3 bodies)."""
        obj = nbody.generate(integration_steps=_TEST_STEPS)
        obj.validate_or_raise()
        assert obj.curves is not None
        assert len(obj.curves) == 3

    def test_bounding_box_finite(self, nbody: NBodyGenerator) -> None:
        """Bounding box is finite."""
        obj = nbody.generate(
            params={"num_bodies": 2}, integration_steps=_TEST_STEPS
        )
        assert obj.bounding_box is not None
        min_c = np.array(obj.bounding_box.min_corner)
        max_c = np.array(obj.bounding_box.max_corner)
        assert np.all(np.isfinite(min_c))
        assert np.all(np.isfinite(max_c))

    def test_too_few_bodies_raises(self, nbody: NBodyGenerator) -> None:
        """num_bodies < 2 raises ValueError."""
        with pytest.raises(ValueError, match="num_bodies must be >= 2"):
            nbody.generate(params={"num_bodies": 1})

    def test_default_representation_is_raw_point_cloud(
        self, nbody: NBodyGenerator
    ) -> None:
        """Default representation is RAW_POINT_CLOUD."""
        rep = nbody.get_default_representation()
        assert rep.type == RepresentationType.RAW_POINT_CLOUD

    def test_metadata_recorded(self, nbody: NBodyGenerator) -> None:
        """Generator metadata is recorded correctly."""
        obj = nbody.generate(integration_steps=_TEST_STEPS)
        assert obj.generator_name == "nbody"
        assert obj.category == "physics"

    def test_registry_discoverable(self) -> None:
        """N-body is discoverable via the registry."""
        gen_cls = get_generator("nbody")
        assert gen_cls is NBodyGenerator

    def test_alias_discoverable(self) -> None:
        """N-body alias is discoverable."""
        gen_cls = get_generator("n_body")
        assert gen_cls is NBodyGenerator


# ---------------------------------------------------------------------------
# Planetary positions tests
# ---------------------------------------------------------------------------


class TestPlanetaryPositions:
    """Tests for the planetary positions generator."""

    def test_produces_multiple_curves(
        self, planetary: PlanetaryPositionsGenerator
    ) -> None:
        """Produces one curve per planet (8 planets)."""
        obj = planetary.generate(curve_points=_TEST_CURVE_POINTS)
        obj.validate_or_raise()

        assert obj.curves is not None
        assert len(obj.curves) == 8

    def test_orbits_are_closed(
        self, planetary: PlanetaryPositionsGenerator
    ) -> None:
        """All orbital curves are closed."""
        obj = planetary.generate(curve_points=_TEST_CURVE_POINTS)
        assert obj.curves is not None
        for curve in obj.curves:
            assert curve.closed is True

    def test_no_nan_in_output(
        self, planetary: PlanetaryPositionsGenerator
    ) -> None:
        """No NaN values in any curve."""
        obj = planetary.generate(curve_points=_TEST_CURVE_POINTS)
        assert obj.curves is not None
        for curve in obj.curves:
            assert not np.any(np.isnan(curve.points))

    def test_orbits_ordered_by_distance(
        self, planetary: PlanetaryPositionsGenerator
    ) -> None:
        """Inner planets have smaller orbits than outer planets."""
        obj = planetary.generate(curve_points=_TEST_CURVE_POINTS)
        assert obj.curves is not None

        mean_radii = [
            float(np.mean(np.linalg.norm(c.points, axis=1)))
            for c in obj.curves
        ]
        # Should be roughly increasing (Mercury < Venus < ... < Neptune)
        for i in range(len(mean_radii) - 1):
            assert mean_radii[i] < mean_radii[i + 1], (
                f"Planet {i} radius {mean_radii[i]:.2f} >= "
                f"planet {i+1} radius {mean_radii[i+1]:.2f}"
            )

    def test_bounding_box_finite(
        self, planetary: PlanetaryPositionsGenerator
    ) -> None:
        """Bounding box is finite and non-degenerate."""
        obj = planetary.generate(curve_points=_TEST_CURVE_POINTS)
        assert obj.bounding_box is not None
        min_c = np.array(obj.bounding_box.min_corner)
        max_c = np.array(obj.bounding_box.max_corner)
        assert np.all(np.isfinite(min_c))
        assert np.all(np.isfinite(max_c))
        extents = max_c - min_c
        assert np.all(extents > 0)

    def test_metadata_recorded(
        self, planetary: PlanetaryPositionsGenerator
    ) -> None:
        """Generator metadata is recorded correctly."""
        obj = planetary.generate(curve_points=_TEST_CURVE_POINTS)
        assert obj.generator_name == "planetary_positions"
        assert obj.category == "physics"

    def test_default_representation_is_tube(
        self, planetary: PlanetaryPositionsGenerator
    ) -> None:
        """Default representation is TUBE."""
        rep = planetary.get_default_representation()
        assert rep.type == RepresentationType.TUBE

    def test_registry_discoverable(self) -> None:
        """Planetary positions is discoverable via the registry."""
        gen_cls = get_generator("planetary_positions")
        assert gen_cls is PlanetaryPositionsGenerator

    def test_alias_discoverable(self) -> None:
        """Solar system alias is discoverable."""
        gen_cls = get_generator("solar_system")
        assert gen_cls is PlanetaryPositionsGenerator
