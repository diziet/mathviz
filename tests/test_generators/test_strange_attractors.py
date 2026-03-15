"""Tests for strange attractor generators: Clifford, Dequan Li, Sprott."""

import numpy as np
import pytest

from mathviz.core.generator import clear_registry, list_generators, register
from mathviz.core.representation import RepresentationType
from mathviz.generators.attractors.clifford import CliffordGenerator
from mathviz.generators.attractors.dequan_li import DequanLiGenerator
from mathviz.generators.attractors.sprott import SPROTT_VARIANTS, SprottGenerator

_TEST_STEPS = 1500
_TEST_POINTS = 1000


@pytest.fixture(autouse=True)
def _clean_registry():
    """Reset registry and register generators for each test."""
    clear_registry(suppress_discovery=True)
    register(CliffordGenerator)
    register(DequanLiGenerator)
    register(SprottGenerator)
    yield
    clear_registry(suppress_discovery=True)


# ---------------------------------------------------------------------------
# Clifford attractor
# ---------------------------------------------------------------------------


def test_clifford_produces_point_cloud_with_expected_count() -> None:
    """Clifford attractor produces a point cloud with expected number of points."""
    gen = CliffordGenerator()
    obj = gen.generate(num_points=_TEST_POINTS)
    assert obj.point_cloud is not None
    assert len(obj.point_cloud.points) == _TEST_POINTS


def test_clifford_output_varies_with_seed() -> None:
    """Clifford output varies with seed."""
    gen = CliffordGenerator()
    obj1 = gen.generate(seed=1, num_points=_TEST_POINTS)
    obj2 = gen.generate(seed=2, num_points=_TEST_POINTS)
    assert obj1.point_cloud is not None and obj2.point_cloud is not None
    assert not np.array_equal(obj1.point_cloud.points, obj2.point_cloud.points)


def test_clifford_deterministic_with_same_seed() -> None:
    """Same seed produces identical Clifford output."""
    gen = CliffordGenerator()
    obj1 = gen.generate(seed=42, num_points=_TEST_POINTS)
    obj2 = gen.generate(seed=42, num_points=_TEST_POINTS)
    np.testing.assert_array_equal(
        obj1.point_cloud.points, obj2.point_cloud.points
    )


def test_clifford_bounding_box_finite() -> None:
    """Clifford bounding box is finite and non-degenerate."""
    gen = CliffordGenerator()
    obj = gen.generate(num_points=_TEST_POINTS)
    assert obj.bounding_box is not None
    min_c = np.array(obj.bounding_box.min_corner)
    max_c = np.array(obj.bounding_box.max_corner)
    assert np.all(np.isfinite(min_c))
    assert np.all(np.isfinite(max_c))
    extents = max_c - min_c
    # x and y should have nonzero extent; z is scaled iteration index
    assert extents[0] > 0
    assert extents[1] > 0


def test_clifford_no_nan() -> None:
    """Clifford point cloud contains no NaN values."""
    gen = CliffordGenerator()
    obj = gen.generate(num_points=_TEST_POINTS)
    assert not np.any(np.isnan(obj.point_cloud.points))


def test_clifford_default_representation_sparse_shell() -> None:
    """Clifford defaults to SPARSE_SHELL representation."""
    gen = CliffordGenerator()
    rep = gen.get_default_representation()
    assert rep.type == RepresentationType.SPARSE_SHELL


def test_clifford_rejects_too_few_points() -> None:
    """Clifford raises ValueError for num_points below minimum."""
    gen = CliffordGenerator()
    with pytest.raises(ValueError, match="num_points must be"):
        gen.generate(num_points=10)


# ---------------------------------------------------------------------------
# Dequan Li attractor
# ---------------------------------------------------------------------------


def test_dequan_li_produces_curve_with_many_points() -> None:
    """Dequan Li attractor produces a curve with expected number of points."""
    gen = DequanLiGenerator()
    obj = gen.generate(integration_steps=_TEST_STEPS)
    assert obj.curves is not None
    assert len(obj.curves[0].points) > 100


def test_dequan_li_default_params_bounded() -> None:
    """Dequan Li default params produce a bounded (non-diverging) trajectory."""
    gen = DequanLiGenerator()
    obj = gen.generate(integration_steps=_TEST_STEPS)
    assert obj.bounding_box is not None
    min_c = np.array(obj.bounding_box.min_corner)
    max_c = np.array(obj.bounding_box.max_corner)
    assert np.all(np.isfinite(min_c))
    assert np.all(np.isfinite(max_c))
    extents = max_c - min_c
    assert np.all(extents > 0), f"Degenerate bounding box: extents={extents}"
    assert np.all(extents < 500), f"Trajectory seems divergent: extents={extents}"


def test_dequan_li_no_nan() -> None:
    """Dequan Li trajectory contains no NaN values."""
    gen = DequanLiGenerator()
    obj = gen.generate(integration_steps=_TEST_STEPS)
    assert not np.any(np.isnan(obj.curves[0].points))


def test_dequan_li_deterministic() -> None:
    """Dequan Li is deterministic given seed."""
    gen = DequanLiGenerator()
    obj1 = gen.generate(seed=42, integration_steps=_TEST_STEPS)
    obj2 = gen.generate(seed=42, integration_steps=_TEST_STEPS)
    np.testing.assert_array_equal(
        obj1.curves[0].points, obj2.curves[0].points
    )


def test_dequan_li_varies_with_seed() -> None:
    """Dequan Li output varies with seed."""
    gen = DequanLiGenerator()
    obj1 = gen.generate(seed=1, integration_steps=_TEST_STEPS)
    obj2 = gen.generate(seed=2, integration_steps=_TEST_STEPS)
    assert not np.array_equal(obj1.curves[0].points, obj2.curves[0].points)


def test_dequan_li_default_representation_tube() -> None:
    """Dequan Li defaults to TUBE representation."""
    gen = DequanLiGenerator()
    rep = gen.get_default_representation()
    assert rep.type == RepresentationType.TUBE


def test_dequan_li_validates_params() -> None:
    """Dequan Li rejects invalid ODE parameters."""
    gen = DequanLiGenerator()
    with pytest.raises(ValueError, match="a must be positive"):
        gen.generate(params={"a": -1.0}, integration_steps=_TEST_STEPS)


# ---------------------------------------------------------------------------
# Sprott systems
# ---------------------------------------------------------------------------


def test_sprott_accepts_system_parameter() -> None:
    """Sprott generator accepts a system parameter to select variant."""
    gen = SprottGenerator()
    for variant in SPROTT_VARIANTS:
        obj = gen.generate(
            params={"system": variant}, integration_steps=_TEST_STEPS,
        )
        assert obj.curves is not None
        assert len(obj.curves[0].points) > 0
        assert obj.parameters["system"] == variant


def test_sprott_each_variant_produces_distinct_trajectory() -> None:
    """Each Sprott variant produces a distinct trajectory."""
    gen = SprottGenerator()
    trajectories: dict[str, np.ndarray] = {}
    for variant in sorted(SPROTT_VARIANTS):
        obj = gen.generate(
            params={"system": variant},
            seed=42,
            integration_steps=_TEST_STEPS,
        )
        trajectories[variant] = obj.curves[0].points

    variants = sorted(trajectories.keys())
    for i, v1 in enumerate(variants):
        for v2 in variants[i + 1:]:
            assert not np.array_equal(
                trajectories[v1], trajectories[v2]
            ), f"{v1} and {v2} produced identical trajectories"


def test_sprott_default_is_sprott_a() -> None:
    """Sprott defaults to sprott_a system."""
    gen = SprottGenerator()
    defaults = gen.get_default_params()
    assert defaults["system"] == "sprott_a"


def test_sprott_rejects_unknown_system() -> None:
    """Sprott raises ValueError for unknown system name."""
    gen = SprottGenerator()
    with pytest.raises(ValueError, match="Unknown Sprott system"):
        gen.generate(
            params={"system": "sprott_z"}, integration_steps=_TEST_STEPS,
        )


def test_sprott_deterministic() -> None:
    """Sprott is deterministic given seed and system."""
    gen = SprottGenerator()
    obj1 = gen.generate(
        params={"system": "sprott_b"}, seed=42, integration_steps=_TEST_STEPS,
    )
    obj2 = gen.generate(
        params={"system": "sprott_b"}, seed=42, integration_steps=_TEST_STEPS,
    )
    np.testing.assert_array_equal(
        obj1.curves[0].points, obj2.curves[0].points
    )


def test_sprott_no_nan() -> None:
    """Sprott trajectories contain no NaN values."""
    gen = SprottGenerator()
    for variant in SPROTT_VARIANTS:
        obj = gen.generate(
            params={"system": variant}, integration_steps=_TEST_STEPS,
        )
        assert not np.any(np.isnan(obj.curves[0].points)), (
            f"NaN found in {variant}"
        )


def test_sprott_bounded_trajectories() -> None:
    """Sprott trajectories are bounded (not diverging)."""
    gen = SprottGenerator()
    for variant in SPROTT_VARIANTS:
        obj = gen.generate(
            params={"system": variant}, integration_steps=_TEST_STEPS,
        )
        assert obj.bounding_box is not None
        min_c = np.array(obj.bounding_box.min_corner)
        max_c = np.array(obj.bounding_box.max_corner)
        assert np.all(np.isfinite(min_c)), f"{variant} has infinite min"
        assert np.all(np.isfinite(max_c)), f"{variant} has infinite max"


def test_sprott_default_representation_tube() -> None:
    """Sprott defaults to TUBE representation."""
    gen = SprottGenerator()
    rep = gen.get_default_representation()
    assert rep.type == RepresentationType.TUBE


# ---------------------------------------------------------------------------
# Registry integration
# ---------------------------------------------------------------------------


def test_all_three_generators_register_and_appear_in_list() -> None:
    """All three generators register correctly and appear in list."""
    metas = list_generators()
    names = {m.name for m in metas}
    assert "clifford" in names
    assert "dequan_li" in names
    assert "sprott" in names
