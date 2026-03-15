"""Tests for exotic knot generators: pretzel, borromean, chain, trefoil-on-torus, cinquefoil."""

import numpy as np
import pytest

from mathviz.core.generator import (
    clear_registry,
    get_generator,
    list_generators,
    register,
)
from mathviz.core.representation import RepresentationType
from mathviz.generators.knots.exotic_knots import (
    CinquefoilKnotGenerator,
    PretzelKnotGenerator,
)
from mathviz.generators.knots.linked_structures import (
    BorromeanRingsGenerator,
    ChainLinksGenerator,
)
from mathviz.generators.knots.trefoil_on_torus import TrefoilOnTorusGenerator


@pytest.fixture(autouse=True)
def _clean_registry():
    """Reset registry and register exotic knot generators for each test."""
    clear_registry(suppress_discovery=True)
    register(PretzelKnotGenerator)
    register(CinquefoilKnotGenerator)
    register(BorromeanRingsGenerator)
    register(ChainLinksGenerator)
    register(TrefoilOnTorusGenerator)
    yield
    clear_registry(suppress_discovery=True)


_TEST_CURVE_POINTS = 128


# ===========================================================================
# Pretzel knot
# ===========================================================================


def test_pretzel_knot_produces_closed_curve() -> None:
    """Pretzel knot produces a single closed curve."""
    gen = PretzelKnotGenerator()
    obj = gen.generate(curve_points=_TEST_CURVE_POINTS)
    assert obj.curves is not None
    assert len(obj.curves) == 1
    assert obj.curves[0].closed


def test_pretzel_knot_no_nan() -> None:
    """Pretzel knot has no NaN values."""
    gen = PretzelKnotGenerator()
    obj = gen.generate(curve_points=_TEST_CURVE_POINTS)
    assert not np.any(np.isnan(obj.curves[0].points))


def test_pretzel_knot_endpoints_close() -> None:
    """Pretzel knot endpoints are close for a periodic curve."""
    gen = PretzelKnotGenerator()
    obj = gen.generate(curve_points=_TEST_CURVE_POINTS)
    points = obj.curves[0].points
    distance = np.linalg.norm(points[-1] - points[0])
    assert distance < 0.5, f"Endpoint gap too large: {distance}"


def test_pretzel_knot_custom_params() -> None:
    """Pretzel knot accepts custom p, q values."""
    gen = PretzelKnotGenerator()
    obj = gen.generate(params={"p": 3, "q": 5}, curve_points=_TEST_CURVE_POINTS)
    assert obj.parameters["p"] == 3
    assert obj.parameters["q"] == 5


def test_pretzel_knot_invalid_p_raises() -> None:
    """Pretzel knot with p < 1 raises ValueError."""
    gen = PretzelKnotGenerator()
    with pytest.raises(ValueError, match="p must be >= 1"):
        gen.generate(params={"p": 0})


def test_pretzel_knot_default_representation() -> None:
    """Pretzel knot defaults to TUBE representation."""
    gen = PretzelKnotGenerator()
    rep = gen.get_default_representation()
    assert rep.type == RepresentationType.TUBE


# ===========================================================================
# Borromean rings
# ===========================================================================


def test_borromean_rings_produces_three_closed_curves() -> None:
    """Borromean rings produces exactly 3 closed curves."""
    gen = BorromeanRingsGenerator()
    obj = gen.generate(curve_points=_TEST_CURVE_POINTS)
    assert obj.curves is not None
    assert len(obj.curves) == 3
    for curve in obj.curves:
        assert curve.closed


def test_borromean_rings_no_nan() -> None:
    """Borromean rings have no NaN values."""
    gen = BorromeanRingsGenerator()
    obj = gen.generate(curve_points=_TEST_CURVE_POINTS)
    for curve in obj.curves:
        assert not np.any(np.isnan(curve.points))


def test_borromean_rings_default_representation() -> None:
    """Borromean rings default to TUBE representation."""
    gen = BorromeanRingsGenerator()
    rep = gen.get_default_representation()
    assert rep.type == RepresentationType.TUBE


# ===========================================================================
# Chain links
# ===========================================================================


def test_chain_links_five_produces_five_curves() -> None:
    """Chain links with num_links=5 produces 5 closed curves."""
    gen = ChainLinksGenerator()
    obj = gen.generate(
        params={"num_links": 5}, curve_points=_TEST_CURVE_POINTS,
    )
    assert obj.curves is not None
    assert len(obj.curves) == 5
    for curve in obj.curves:
        assert curve.closed


def test_chain_links_one_produces_single_ring() -> None:
    """Chain links with num_links=1 produces a single ring."""
    gen = ChainLinksGenerator()
    obj = gen.generate(
        params={"num_links": 1}, curve_points=_TEST_CURVE_POINTS,
    )
    assert obj.curves is not None
    assert len(obj.curves) == 1
    assert obj.curves[0].closed


def test_chain_links_no_nan() -> None:
    """Chain links have no NaN values."""
    gen = ChainLinksGenerator()
    obj = gen.generate(curve_points=_TEST_CURVE_POINTS)
    for curve in obj.curves:
        assert not np.any(np.isnan(curve.points))


def test_chain_links_invalid_num_links_raises() -> None:
    """Chain links with num_links < 1 raises ValueError."""
    gen = ChainLinksGenerator()
    with pytest.raises(ValueError, match="num_links must be >= 1"):
        gen.generate(params={"num_links": 0})


def test_chain_links_default_representation() -> None:
    """Chain links default to TUBE representation."""
    gen = ChainLinksGenerator()
    rep = gen.get_default_representation()
    assert rep.type == RepresentationType.TUBE


# ===========================================================================
# Trefoil on torus
# ===========================================================================


def test_trefoil_on_torus_produces_curves_and_mesh() -> None:
    """Trefoil on torus produces both curves and a mesh."""
    gen = TrefoilOnTorusGenerator()
    obj = gen.generate(curve_points=_TEST_CURVE_POINTS)
    assert obj.curves is not None
    assert len(obj.curves) >= 1
    assert obj.curves[0].closed
    assert obj.mesh is not None
    assert len(obj.mesh.vertices) > 0
    assert len(obj.mesh.faces) > 0


def test_trefoil_on_torus_no_nan() -> None:
    """Trefoil on torus has no NaN values in curves or mesh."""
    gen = TrefoilOnTorusGenerator()
    obj = gen.generate(curve_points=_TEST_CURVE_POINTS)
    assert not np.any(np.isnan(obj.curves[0].points))
    assert not np.any(np.isnan(obj.mesh.vertices))


def test_trefoil_on_torus_default_representation() -> None:
    """Trefoil on torus defaults to TUBE representation."""
    gen = TrefoilOnTorusGenerator()
    rep = gen.get_default_representation()
    assert rep.type == RepresentationType.TUBE


# ===========================================================================
# Cinquefoil knot
# ===========================================================================


def test_cinquefoil_knot_produces_closed_curve() -> None:
    """Cinquefoil knot produces a single closed curve."""
    gen = CinquefoilKnotGenerator()
    obj = gen.generate(curve_points=_TEST_CURVE_POINTS)
    assert obj.curves is not None
    assert len(obj.curves) == 1
    assert obj.curves[0].closed


def test_cinquefoil_knot_distinct_from_trefoil() -> None:
    """Cinquefoil knot curve is distinct from a trefoil (2,3) knot."""
    cinquefoil = CinquefoilKnotGenerator()
    obj_cinq = cinquefoil.generate(curve_points=_TEST_CURVE_POINTS)

    # Generate a trefoil-like curve for comparison (2,3 torus knot)
    pretzel = PretzelKnotGenerator()
    obj_tref = pretzel.generate(
        params={"p": 2, "q": 3}, curve_points=_TEST_CURVE_POINTS,
    )

    max_diff = np.max(
        np.abs(obj_cinq.curves[0].points - obj_tref.curves[0].points)
    )
    assert max_diff > 0.01, f"Cinquefoil too similar to trefoil: {max_diff}"


def test_cinquefoil_knot_no_nan() -> None:
    """Cinquefoil knot has no NaN values."""
    gen = CinquefoilKnotGenerator()
    obj = gen.generate(curve_points=_TEST_CURVE_POINTS)
    assert not np.any(np.isnan(obj.curves[0].points))


def test_cinquefoil_knot_default_representation() -> None:
    """Cinquefoil knot defaults to TUBE representation."""
    gen = CinquefoilKnotGenerator()
    rep = gen.get_default_representation()
    assert rep.type == RepresentationType.TUBE


# ===========================================================================
# Registration — all generators appear in registry
# ===========================================================================


def test_all_generators_registered() -> None:
    """All exotic knot generators are registered and discoverable."""
    expected_names = {
        "pretzel_knot",
        "cinquefoil_knot",
        "borromean_rings",
        "chain_links",
        "trefoil_on_torus",
    }
    for name in expected_names:
        gen_cls = get_generator(name)
        assert gen_cls is not None


def test_all_generators_in_list() -> None:
    """All exotic knot generators appear in list_generators()."""
    all_meta = list_generators()
    registered_names = {m.name for m in all_meta}
    expected_names = {
        "pretzel_knot",
        "cinquefoil_knot",
        "borromean_rings",
        "chain_links",
        "trefoil_on_torus",
    }
    assert expected_names.issubset(registered_names)
