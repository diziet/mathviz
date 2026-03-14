"""Tests for torus knot generator with trefoil and cinquefoil aliases."""

import numpy as np
import pytest

from mathviz.core.container import Container, PlacementPolicy
from mathviz.core.generator import (
    clear_registry,
    get_generator,
    get_generator_meta,
    register,
)
from mathviz.core.representation import RepresentationConfig, RepresentationType
from mathviz.generators.knots.torus_knot import TorusKnotGenerator
from mathviz.pipeline.runner import run


@pytest.fixture(autouse=True)
def _clean_registry():
    """Reset registry and re-register torus_knot for each test."""
    clear_registry(suppress_discovery=True)
    register(TorusKnotGenerator)
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
    obj = result.math_object
    assert obj.mesh is not None
    assert len(obj.mesh.vertices) > 0
    assert len(obj.mesh.faces) > 0

    # Watertight check: every edge appears in exactly 2 faces
    edge_count: dict[tuple[int, int], int] = {}
    for face in obj.mesh.faces:
        for i in range(3):
            edge = tuple(sorted([int(face[i]), int(face[(i + 1) % 3])]))
            edge_count[edge] = edge_count.get(edge, 0) + 1

    non_manifold = {e: c for e, c in edge_count.items() if c != 2}
    assert len(non_manifold) == 0, (
        f"Mesh is not watertight: {len(non_manifold)} non-manifold edges"
    )


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
