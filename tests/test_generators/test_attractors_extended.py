"""Tests for remaining attractor generators: Rössler, Chen, Thomas, Halvorsen, Aizawa."""

import numpy as np
import pytest

from mathviz.core.generator import clear_registry, register
from mathviz.generators.attractors.aizawa import AizawaGenerator
from mathviz.generators.attractors.chen import ChenGenerator
from mathviz.generators.attractors.halvorsen import HalvorsenGenerator
from mathviz.generators.attractors.rossler import RosslerGenerator
from mathviz.generators.attractors.thomas import ThomasGenerator

_TEST_STEPS = 5000


@pytest.fixture(autouse=True)
def _clean_registry():
    """Reset registry and register generators for each test."""
    clear_registry(suppress_discovery=True)
    register(RosslerGenerator)
    register(ChenGenerator)
    register(ThomasGenerator)
    register(HalvorsenGenerator)
    register(AizawaGenerator)
    yield
    clear_registry(suppress_discovery=True)


def _assert_finite_nondegenerate(obj) -> None:
    """Assert bounding box is finite and non-degenerate on all 3 axes."""
    assert obj.bounding_box is not None
    min_c = np.array(obj.bounding_box.min_corner)
    max_c = np.array(obj.bounding_box.max_corner)
    assert np.all(np.isfinite(min_c))
    assert np.all(np.isfinite(max_c))
    extents = max_c - min_c
    assert np.all(extents > 0), f"Degenerate bounding box: extents={extents}"


def _assert_no_nan(obj) -> None:
    """Assert trajectory contains no NaN values."""
    assert obj.curves is not None
    assert not np.any(np.isnan(obj.curves[0].points))


def _assert_deterministic(generator, seed: int = 42) -> None:
    """Assert same seed produces identical output."""
    obj1 = generator.generate(seed=seed, integration_steps=_TEST_STEPS)
    obj2 = generator.generate(seed=seed, integration_steps=_TEST_STEPS)
    assert obj1.curves is not None and obj2.curves is not None
    np.testing.assert_array_equal(
        obj1.curves[0].points, obj2.curves[0].points
    )


# ---------------------------------------------------------------------------
# Rössler
# ---------------------------------------------------------------------------


def test_rossler_finite_nondegenerate() -> None:
    """Rössler produces finite, non-degenerate geometry."""
    gen = RosslerGenerator()
    obj = gen.generate(integration_steps=_TEST_STEPS)
    _assert_finite_nondegenerate(obj)
    _assert_no_nan(obj)


def test_rossler_wider_than_tall() -> None:
    """Rössler bounding box is wider than tall (characteristic folded-band)."""
    gen = RosslerGenerator()
    # Use more steps for reliable shape characterization
    obj = gen.generate(integration_steps=20_000)
    assert obj.bounding_box is not None
    min_c = np.array(obj.bounding_box.min_corner)
    max_c = np.array(obj.bounding_box.max_corner)
    extents = max_c - min_c
    # X-Y diagonal extent should exceed Z extent (folded-band is flat)
    xy_diagonal = np.sqrt(extents[0] ** 2 + extents[1] ** 2)
    z_extent = extents[2]
    assert xy_diagonal > z_extent, (
        f"Expected wider than tall: xy_diag={xy_diagonal}, z={z_extent}"
    )


def test_rossler_deterministic() -> None:
    """Rössler is deterministic given seed."""
    _assert_deterministic(RosslerGenerator())


# ---------------------------------------------------------------------------
# Chen
# ---------------------------------------------------------------------------


def test_chen_finite_nondegenerate() -> None:
    """Chen produces finite, non-degenerate geometry."""
    gen = ChenGenerator()
    obj = gen.generate(integration_steps=_TEST_STEPS)
    _assert_finite_nondegenerate(obj)
    _assert_no_nan(obj)


def test_chen_deterministic() -> None:
    """Chen is deterministic given seed."""
    _assert_deterministic(ChenGenerator())


# ---------------------------------------------------------------------------
# Thomas
# ---------------------------------------------------------------------------


def test_thomas_finite_nondegenerate() -> None:
    """Thomas produces finite, non-degenerate geometry."""
    gen = ThomasGenerator()
    obj = gen.generate(integration_steps=_TEST_STEPS)
    _assert_finite_nondegenerate(obj)
    _assert_no_nan(obj)


def test_thomas_bounded_with_default_b() -> None:
    """Thomas with default b≈0.208186 produces a bounded attractor."""
    gen = ThomasGenerator()
    obj = gen.generate(integration_steps=_TEST_STEPS)
    assert obj.bounding_box is not None
    min_c = np.array(obj.bounding_box.min_corner)
    max_c = np.array(obj.bounding_box.max_corner)
    extents = max_c - min_c
    # Bounded attractor: all extents should be < 20 (Thomas is compact)
    assert np.all(extents < 20.0), (
        f"Thomas attractor seems divergent: extents={extents}"
    )
    assert np.all(np.isfinite(min_c))


def test_thomas_deterministic() -> None:
    """Thomas is deterministic given seed."""
    _assert_deterministic(ThomasGenerator())


# ---------------------------------------------------------------------------
# Halvorsen
# ---------------------------------------------------------------------------


def test_halvorsen_finite_nondegenerate() -> None:
    """Halvorsen produces finite, non-degenerate geometry."""
    gen = HalvorsenGenerator()
    obj = gen.generate(integration_steps=_TEST_STEPS)
    _assert_finite_nondegenerate(obj)
    _assert_no_nan(obj)


def test_halvorsen_deterministic() -> None:
    """Halvorsen is deterministic given seed."""
    _assert_deterministic(HalvorsenGenerator())


# ---------------------------------------------------------------------------
# Aizawa
# ---------------------------------------------------------------------------


def test_aizawa_finite_nondegenerate() -> None:
    """Aizawa produces finite, non-degenerate geometry."""
    gen = AizawaGenerator()
    obj = gen.generate(integration_steps=_TEST_STEPS)
    _assert_finite_nondegenerate(obj)
    _assert_no_nan(obj)


def test_aizawa_deterministic() -> None:
    """Aizawa is deterministic given seed."""
    _assert_deterministic(AizawaGenerator())


# ---------------------------------------------------------------------------
# Default representation is RAW_POINT_CLOUD for all new attractors
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "gen_cls",
    [RosslerGenerator, ChenGenerator, ThomasGenerator,
     HalvorsenGenerator, AizawaGenerator],
    ids=["rossler", "chen", "thomas", "halvorsen", "aizawa"],
)
def test_default_representation_is_raw_point_cloud(gen_cls) -> None:
    """All new attractors default to RAW_POINT_CLOUD representation."""
    from mathviz.core.representation import RepresentationType
    gen = gen_cls()
    rep = gen.get_default_representation()
    assert rep.type == RepresentationType.RAW_POINT_CLOUD
