"""Tests for knot generators: torus, figure-eight, lissajous, seven-crossing."""

import numpy as np
import pytest

from mathviz.core.container import Container, PlacementPolicy
from mathviz.core.generator import (
    clear_registry,
    get_generator,
    get_generator_meta,
    register,
)
from mathviz.core.math_object import Mesh
from mathviz.core.representation import RepresentationConfig, RepresentationType
from mathviz.generators.knots.figure_eight_knot import FigureEightKnotGenerator
from mathviz.generators.knots.lissajous_knot import LissajousKnotGenerator
from mathviz.generators.knots.seven_crossing_knots import SevenCrossingKnotsGenerator
from mathviz.generators.knots.torus_knot import TorusKnotGenerator
from mathviz.pipeline.runner import run


def _assert_mesh_watertight(mesh: Mesh) -> None:
    """Assert every edge appears in exactly 2 faces (watertight manifold)."""
    assert len(mesh.vertices) > 0
    assert len(mesh.faces) > 0

    edge_count: dict[tuple[int, int], int] = {}
    for face in mesh.faces:
        for i in range(3):
            edge = tuple(sorted([int(face[i]), int(face[(i + 1) % 3])]))
            edge_count[edge] = edge_count.get(edge, 0) + 1

    non_manifold = {e: c for e, c in edge_count.items() if c != 2}
    assert len(non_manifold) == 0, (
        f"Mesh is not watertight: {len(non_manifold)} non-manifold edges"
    )


@pytest.fixture(autouse=True)
def _clean_registry():
    """Reset registry and re-register all knot generators for each test."""
    clear_registry(suppress_discovery=True)
    register(TorusKnotGenerator)
    register(FigureEightKnotGenerator)
    register(LissajousKnotGenerator)
    register(SevenCrossingKnotsGenerator)
    yield
    clear_registry(suppress_discovery=True)


@pytest.fixture
def knot() -> TorusKnotGenerator:
    """Return a TorusKnotGenerator instance."""
    return TorusKnotGenerator()


# Use fewer points for test speed
_TEST_CURVE_POINTS = 128


# ---------------------------------------------------------------------------
# "trefoil" alias resolves to torus_knot and produces p=2, q=3 defaults
# ---------------------------------------------------------------------------


def test_trefoil_alias_resolves_to_torus_knot() -> None:
    """'trefoil' alias resolves to the TorusKnotGenerator class."""
    gen_cls = get_generator("trefoil")
    assert gen_cls is TorusKnotGenerator


def test_trefoil_defaults_p2_q3() -> None:
    """'trefoil' alias produces p=2, q=3 defaults via pipeline resolution."""
    result = run(
        "trefoil",
        container=Container.with_uniform_margin(),
        placement=PlacementPolicy(),
        resolution_kwargs={"curve_points": _TEST_CURVE_POINTS},
    )
    assert result.math_object.parameters["p"] == 2
    assert result.math_object.parameters["q"] == 3


# ---------------------------------------------------------------------------
# "cinquefoil" alias produces p=2, q=5 defaults
# ---------------------------------------------------------------------------


def test_cinquefoil_alias_resolves_to_torus_knot() -> None:
    """'cinquefoil' alias resolves to the TorusKnotGenerator class."""
    gen_cls = get_generator("cinquefoil")
    assert gen_cls is TorusKnotGenerator


def test_cinquefoil_defaults_p2_q5() -> None:
    """'cinquefoil' alias produces p=2, q=5 defaults via pipeline resolution."""
    result = run(
        "cinquefoil",
        container=Container.with_uniform_margin(),
        placement=PlacementPolicy(),
        resolution_kwargs={"curve_points": _TEST_CURVE_POINTS},
    )
    assert result.math_object.parameters["p"] == 2
    assert result.math_object.parameters["q"] == 5


# ---------------------------------------------------------------------------
# Output curve is closed (first and last points coincident or close)
# ---------------------------------------------------------------------------


def test_curve_is_closed(knot: TorusKnotGenerator) -> None:
    """Output curve is marked as closed."""
    obj = knot.generate(curve_points=_TEST_CURVE_POINTS)
    assert obj.curves is not None
    assert len(obj.curves) == 1
    assert obj.curves[0].closed


def test_curve_endpoints_coincident(knot: TorusKnotGenerator) -> None:
    """First and last points of the closed curve are close in space."""
    obj = knot.generate(curve_points=_TEST_CURVE_POINTS)
    points = obj.curves[0].points
    distance = np.linalg.norm(points[-1] - points[0])
    # For a periodic curve sampled without endpoint, gap scales with 1/num_points
    assert distance < 0.2, f"Curve endpoint gap too large: {distance}"


# ---------------------------------------------------------------------------
# Full pipeline with TUBE representation produces a watertight mesh
# ---------------------------------------------------------------------------


def test_full_pipeline_tube_produces_watertight_mesh() -> None:
    """Full pipeline with TUBE produces a mesh with vertices and faces."""
    result = run(
        "torus_knot",
        container=Container.with_uniform_margin(),
        placement=PlacementPolicy(),
        resolution_kwargs={"curve_points": _TEST_CURVE_POINTS},
        representation_config=RepresentationConfig(
            type=RepresentationType.TUBE,
            tube_radius=0.1,
            tube_sides=8,
        ),
    )
    assert result.math_object.mesh is not None
    _assert_mesh_watertight(result.math_object.mesh)


# ---------------------------------------------------------------------------
# Explicit --param p=3 --param q=5 via alias still works (param override)
# ---------------------------------------------------------------------------


def test_param_override_via_alias() -> None:
    """Explicit p and q params override alias defaults."""
    result = run(
        "trefoil",
        params={"p": 3, "q": 5},
        container=Container.with_uniform_margin(),
        placement=PlacementPolicy(),
        resolution_kwargs={"curve_points": _TEST_CURVE_POINTS},
    )
    assert result.math_object.parameters["p"] == 3
    assert result.math_object.parameters["q"] == 5


def test_alias_provenance_recorded_in_generator_name() -> None:
    """When resolved via alias, generator_name records the alias used."""
    result = run(
        "trefoil",
        container=Container.with_uniform_margin(),
        placement=PlacementPolicy(),
        resolution_kwargs={"curve_points": _TEST_CURVE_POINTS},
    )
    assert result.math_object.generator_name == "trefoil"


# ---------------------------------------------------------------------------
# Canonical name works
# ---------------------------------------------------------------------------


def test_canonical_name_registered() -> None:
    """torus_knot is registered and discoverable."""
    gen_cls = get_generator("torus_knot")
    assert gen_cls is TorusKnotGenerator


def test_metadata_recorded(knot: TorusKnotGenerator) -> None:
    """Generator name, category, and parameters are recorded."""
    obj = knot.generate(curve_points=_TEST_CURVE_POINTS)
    assert obj.generator_name == "torus_knot"
    assert obj.category == "knots"
    assert obj.parameters["p"] == 2
    assert obj.parameters["q"] == 3
    assert obj.parameters["R"] == 1.0
    assert obj.parameters["r"] == 0.4
    assert obj.seed == 42


def test_default_representation_is_tube(knot: TorusKnotGenerator) -> None:
    """Default representation for torus knot is TUBE."""
    rep = knot.get_default_representation()
    assert rep.type == RepresentationType.TUBE
    assert rep.tube_radius == 0.1


# ---------------------------------------------------------------------------
# Determinism
# ---------------------------------------------------------------------------


def test_same_seed_deterministic(knot: TorusKnotGenerator) -> None:
    """Same seed produces identical output."""
    obj1 = knot.generate(seed=42, curve_points=_TEST_CURVE_POINTS)
    obj2 = knot.generate(seed=42, curve_points=_TEST_CURVE_POINTS)
    assert obj1.curves is not None and obj2.curves is not None
    np.testing.assert_array_equal(
        obj1.curves[0].points, obj2.curves[0].points
    )


# ---------------------------------------------------------------------------
# Bounding box
# ---------------------------------------------------------------------------


def test_bounding_box_finite(knot: TorusKnotGenerator) -> None:
    """Bounding box is finite and non-degenerate."""
    obj = knot.generate(curve_points=_TEST_CURVE_POINTS)
    assert obj.bounding_box is not None
    min_c = np.array(obj.bounding_box.min_corner)
    max_c = np.array(obj.bounding_box.max_corner)
    assert np.all(np.isfinite(min_c))
    assert np.all(np.isfinite(max_c))
    extents = max_c - min_c
    assert np.all(extents > 0)


# ---------------------------------------------------------------------------
# Input validation
# ---------------------------------------------------------------------------


def test_negative_p_raises(knot: TorusKnotGenerator) -> None:
    """Negative p raises ValueError."""
    with pytest.raises(ValueError, match="p must be >= 1"):
        knot.generate(params={"p": 0})


def test_negative_q_raises(knot: TorusKnotGenerator) -> None:
    """Negative q raises ValueError."""
    with pytest.raises(ValueError, match="q must be >= 1"):
        knot.generate(params={"q": 0})


def test_negative_major_radius_raises(knot: TorusKnotGenerator) -> None:
    """Negative major radius raises ValueError."""
    with pytest.raises(ValueError, match="R.*must be positive"):
        knot.generate(params={"R": -1.0})


def test_minor_radius_exceeds_major_raises(knot: TorusKnotGenerator) -> None:
    """Minor radius >= major radius raises ValueError."""
    with pytest.raises(ValueError, match="r must be less than R"):
        knot.generate(params={"R": 1.0, "r": 1.5})


def test_too_few_curve_points_raises(knot: TorusKnotGenerator) -> None:
    """Fewer than minimum curve points raises ValueError."""
    with pytest.raises(ValueError, match="curve_points must be >= 16"):
        knot.generate(curve_points=4)


def test_no_nan_in_curve(knot: TorusKnotGenerator) -> None:
    """Curve contains no NaN values."""
    obj = knot.generate(curve_points=_TEST_CURVE_POINTS)
    assert obj.curves is not None
    assert not np.any(np.isnan(obj.curves[0].points))


def test_curve_points_in_params_warns(
    knot: TorusKnotGenerator,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """curve_points passed in params is ignored with a warning."""
    obj = knot.generate(
        params={"curve_points": 999},
        curve_points=_TEST_CURVE_POINTS,
    )
    assert "should be passed as a resolution kwarg" in caplog.text
    assert obj.parameters["curve_points"] == _TEST_CURVE_POINTS


# ===========================================================================
# Figure-eight knot tests
# ===========================================================================


def test_figure_eight_produces_closed_curve() -> None:
    """Figure-eight knot produces a closed curve."""
    gen = FigureEightKnotGenerator()
    obj = gen.generate(curve_points=_TEST_CURVE_POINTS)
    assert obj.curves is not None
    assert len(obj.curves) == 1
    assert obj.curves[0].closed


def test_figure_eight_endpoints_close() -> None:
    """Figure-eight knot endpoints are close in space."""
    gen = FigureEightKnotGenerator()
    obj = gen.generate(curve_points=_TEST_CURVE_POINTS)
    points = obj.curves[0].points
    distance = np.linalg.norm(points[-1] - points[0])
    assert distance < 0.5, f"Endpoint gap too large: {distance}"


def test_figure_eight_no_nan() -> None:
    """Figure-eight knot has no NaN values."""
    gen = FigureEightKnotGenerator()
    obj = gen.generate(curve_points=_TEST_CURVE_POINTS)
    assert not np.any(np.isnan(obj.curves[0].points))


def test_figure_eight_default_representation() -> None:
    """Figure-eight knot defaults to TUBE representation."""
    gen = FigureEightKnotGenerator()
    rep = gen.get_default_representation()
    assert rep.type == RepresentationType.TUBE


# ===========================================================================
# Lissajous knot tests
# ===========================================================================


def test_lissajous_knot_coprime_produces_closed_curve() -> None:
    """Lissajous knot with coprime frequencies produces a closed curve."""
    gen = LissajousKnotGenerator()
    obj = gen.generate(
        params={"nx": 2, "ny": 3, "nz": 5},
        curve_points=_TEST_CURVE_POINTS,
    )
    assert obj.curves is not None
    assert obj.curves[0].closed


def test_lissajous_knot_endpoints_close() -> None:
    """Lissajous knot endpoints are close in space."""
    gen = LissajousKnotGenerator()
    obj = gen.generate(curve_points=_TEST_CURVE_POINTS)
    points = obj.curves[0].points
    distance = np.linalg.norm(points[-1] - points[0])
    assert distance < 0.5, f"Endpoint gap too large: {distance}"


def test_lissajous_knot_no_nan() -> None:
    """Lissajous knot has no NaN values."""
    gen = LissajousKnotGenerator()
    obj = gen.generate(curve_points=_TEST_CURVE_POINTS)
    assert not np.any(np.isnan(obj.curves[0].points))


def test_lissajous_knot_default_representation() -> None:
    """Lissajous knot defaults to TUBE representation."""
    gen = LissajousKnotGenerator()
    rep = gen.get_default_representation()
    assert rep.type == RepresentationType.TUBE


# ===========================================================================
# Seven-crossing knots tests
# ===========================================================================


def test_seven_crossing_different_indices_produce_distinct_curves() -> None:
    """Different knot_index values produce distinct curves."""
    gen = SevenCrossingKnotsGenerator()
    curves = {}
    for idx in range(1, 8):
        obj = gen.generate(
            params={"knot_index": idx},
            curve_points=_TEST_CURVE_POINTS,
        )
        curves[idx] = obj.curves[0].points

    # Every pair of knots should differ
    for i in range(1, 8):
        for j in range(i + 1, 8):
            max_diff = np.max(np.abs(curves[i] - curves[j]))
            assert max_diff > 0.01, (
                f"Knots 7_{i} and 7_{j} are too similar: max_diff={max_diff}"
            )


def test_seven_crossing_produces_closed_curve() -> None:
    """Seven-crossing knot produces a closed curve."""
    gen = SevenCrossingKnotsGenerator()
    obj = gen.generate(curve_points=_TEST_CURVE_POINTS)
    assert obj.curves is not None
    assert obj.curves[0].closed


def test_seven_crossing_invalid_index_raises() -> None:
    """Invalid knot_index raises ValueError."""
    gen = SevenCrossingKnotsGenerator()
    with pytest.raises(ValueError, match="knot_index must be between"):
        gen.generate(params={"knot_index": 0})
    with pytest.raises(ValueError, match="knot_index must be between"):
        gen.generate(params={"knot_index": 8})


def test_seven_crossing_no_nan() -> None:
    """Seven-crossing knot has no NaN values for all indices."""
    gen = SevenCrossingKnotsGenerator()
    for idx in range(1, 8):
        obj = gen.generate(
            params={"knot_index": idx},
            curve_points=_TEST_CURVE_POINTS,
        )
        assert not np.any(np.isnan(obj.curves[0].points)), (
            f"NaN in 7_{idx} knot"
        )


# ===========================================================================
# TUBE representation on closed knots produces watertight meshes
# ===========================================================================


def test_figure_eight_tube_watertight() -> None:
    """TUBE representation on figure-eight knot produces watertight mesh."""
    result = run(
        "figure_eight_knot",
        container=Container.with_uniform_margin(),
        placement=PlacementPolicy(),
        resolution_kwargs={"curve_points": _TEST_CURVE_POINTS},
        representation_config=RepresentationConfig(
            type=RepresentationType.TUBE, tube_radius=0.1, tube_sides=8,
        ),
    )
    assert result.math_object.mesh is not None
    _assert_mesh_watertight(result.math_object.mesh)
