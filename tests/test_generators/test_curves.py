"""Tests for curve generators: lissajous, logarithmic spiral, cardioid,
fibonacci spiral, parabolic envelope."""

import numpy as np
import pytest

from mathviz.core.container import Container, PlacementPolicy
from mathviz.core.generator import clear_registry, register
from mathviz.core.representation import RepresentationConfig, RepresentationType
from mathviz.generators.curves.cardioid import CardioidGenerator
from mathviz.generators.curves.fibonacci_spiral import FibonacciSpiralGenerator
from mathviz.generators.curves.lissajous_curve import LissajousCurveGenerator
from mathviz.generators.curves.logarithmic_spiral import LogarithmicSpiralGenerator
from mathviz.generators.curves.parabolic_envelope import ParabolicEnvelopeGenerator
from mathviz.pipeline.runner import run

_TEST_CURVE_POINTS = 128


@pytest.fixture(autouse=True)
def _clean_registry():
    """Reset registry and register all curve generators."""
    clear_registry(suppress_discovery=True)
    register(LissajousCurveGenerator)
    register(LogarithmicSpiralGenerator)
    register(CardioidGenerator)
    register(FibonacciSpiralGenerator)
    register(ParabolicEnvelopeGenerator)
    yield
    clear_registry(suppress_discovery=True)


# ===========================================================================
# Lissajous curve
# ===========================================================================


def test_lissajous_curve_closed() -> None:
    """Lissajous curve with integer frequencies is closed."""
    gen = LissajousCurveGenerator()
    obj = gen.generate(curve_points=_TEST_CURVE_POINTS)
    assert obj.curves is not None
    assert obj.curves[0].closed


def test_lissajous_curve_no_nan() -> None:
    """Lissajous curve has no NaN values."""
    gen = LissajousCurveGenerator()
    obj = gen.generate(curve_points=_TEST_CURVE_POINTS)
    assert not np.any(np.isnan(obj.curves[0].points))


def test_lissajous_curve_default_tube() -> None:
    """Lissajous curve defaults to TUBE with thin radius."""
    gen = LissajousCurveGenerator()
    rep = gen.get_default_representation()
    assert rep.type == RepresentationType.TUBE
    assert rep.tube_radius < 0.1  # thinner than knots


# ===========================================================================
# Logarithmic spiral
# ===========================================================================


def test_logarithmic_spiral_extent_scales_with_turns() -> None:
    """Logarithmic spiral extent scales with the number of turns."""
    gen = LogarithmicSpiralGenerator()
    obj_few = gen.generate(
        params={"turns": 2.0}, curve_points=_TEST_CURVE_POINTS,
    )
    obj_many = gen.generate(
        params={"turns": 5.0}, curve_points=_TEST_CURVE_POINTS,
    )
    bbox_few = obj_few.bounding_box
    bbox_many = obj_many.bounding_box

    extent_few = max(
        b - a for a, b in zip(bbox_few.min_corner, bbox_few.max_corner)
    )
    extent_many = max(
        b - a for a, b in zip(bbox_many.min_corner, bbox_many.max_corner)
    )
    assert extent_many > extent_few, (
        f"More turns should produce larger extent: "
        f"{extent_many} vs {extent_few}"
    )


def test_logarithmic_spiral_open() -> None:
    """Logarithmic spiral is an open curve."""
    gen = LogarithmicSpiralGenerator()
    obj = gen.generate(curve_points=_TEST_CURVE_POINTS)
    assert obj.curves is not None
    assert not obj.curves[0].closed


def test_logarithmic_spiral_no_nan() -> None:
    """Logarithmic spiral has no NaN values."""
    gen = LogarithmicSpiralGenerator()
    obj = gen.generate(curve_points=_TEST_CURVE_POINTS)
    assert not np.any(np.isnan(obj.curves[0].points))


# ===========================================================================
# Cardioid
# ===========================================================================


def test_cardioid_closed() -> None:
    """Cardioid curve is closed."""
    gen = CardioidGenerator()
    obj = gen.generate(curve_points=_TEST_CURVE_POINTS)
    assert obj.curves is not None
    assert obj.curves[0].closed


def test_cardioid_no_nan() -> None:
    """Cardioid has no NaN values."""
    gen = CardioidGenerator()
    obj = gen.generate(curve_points=_TEST_CURVE_POINTS)
    assert not np.any(np.isnan(obj.curves[0].points))


def test_cardioid_bounding_box_finite() -> None:
    """Cardioid bounding box is finite."""
    gen = CardioidGenerator()
    obj = gen.generate(curve_points=_TEST_CURVE_POINTS)
    assert obj.bounding_box is not None
    min_c = np.array(obj.bounding_box.min_corner)
    max_c = np.array(obj.bounding_box.max_corner)
    assert np.all(np.isfinite(min_c))
    assert np.all(np.isfinite(max_c))


# ===========================================================================
# Fibonacci spiral
# ===========================================================================


def test_fibonacci_spiral_open() -> None:
    """Fibonacci spiral is an open curve."""
    gen = FibonacciSpiralGenerator()
    obj = gen.generate(curve_points=_TEST_CURVE_POINTS)
    assert obj.curves is not None
    assert not obj.curves[0].closed


def test_fibonacci_spiral_no_nan() -> None:
    """Fibonacci spiral has no NaN values."""
    gen = FibonacciSpiralGenerator()
    obj = gen.generate(curve_points=_TEST_CURVE_POINTS)
    assert not np.any(np.isnan(obj.curves[0].points))


def test_fibonacci_spiral_thin_tube() -> None:
    """Fibonacci spiral defaults to thin tube radius."""
    gen = FibonacciSpiralGenerator()
    rep = gen.get_default_representation()
    assert rep.type == RepresentationType.TUBE
    assert rep.tube_radius < 0.1


# ===========================================================================
# Parabolic envelope
# ===========================================================================


def test_parabolic_envelope_produces_surface_mesh() -> None:
    """Parabolic envelope produces a surface mesh, not just a curve."""
    gen = ParabolicEnvelopeGenerator()
    obj = gen.generate(curve_points=32)
    assert obj.mesh is not None
    assert obj.curves is None or len(obj.curves) == 0
    assert len(obj.mesh.vertices) > 0
    assert len(obj.mesh.faces) > 0


def test_parabolic_envelope_mesh_valid() -> None:
    """Parabolic envelope mesh has valid face indices."""
    gen = ParabolicEnvelopeGenerator()
    obj = gen.generate(curve_points=32)
    assert obj.mesh.faces.min() >= 0
    assert obj.mesh.faces.max() < len(obj.mesh.vertices)


def test_parabolic_envelope_no_nan() -> None:
    """Parabolic envelope has no NaN values."""
    gen = ParabolicEnvelopeGenerator()
    obj = gen.generate(curve_points=32)
    assert not np.any(np.isnan(obj.mesh.vertices))


def test_parabolic_envelope_bounding_box_finite() -> None:
    """Parabolic envelope bounding box is finite."""
    gen = ParabolicEnvelopeGenerator()
    obj = gen.generate(curve_points=32)
    assert obj.bounding_box is not None
    min_c = np.array(obj.bounding_box.min_corner)
    max_c = np.array(obj.bounding_box.max_corner)
    assert np.all(np.isfinite(min_c))
    assert np.all(np.isfinite(max_c))


# ===========================================================================
# TUBE on closed curves produces watertight meshes
# ===========================================================================


def test_cardioid_tube_watertight() -> None:
    """TUBE representation on cardioid produces watertight mesh."""
    result = run(
        "cardioid",
        container=Container.with_uniform_margin(),
        placement=PlacementPolicy(),
        resolution_kwargs={"curve_points": _TEST_CURVE_POINTS},
        representation_config=RepresentationConfig(
            type=RepresentationType.TUBE, tube_radius=0.05, tube_sides=8,
        ),
    )
    obj = result.math_object
    assert obj.mesh is not None
    assert len(obj.mesh.vertices) > 0
    assert len(obj.mesh.faces) > 0

    edge_count: dict[tuple[int, int], int] = {}
    for face in obj.mesh.faces:
        for i in range(3):
            edge = tuple(sorted([int(face[i]), int(face[(i + 1) % 3])]))
            edge_count[edge] = edge_count.get(edge, 0) + 1

    non_manifold = {e: c for e, c in edge_count.items() if c != 2}
    assert len(non_manifold) == 0, (
        f"Mesh is not watertight: {len(non_manifold)} non-manifold edges"
    )


def test_lissajous_curve_tube_watertight() -> None:
    """TUBE representation on Lissajous curve produces watertight mesh."""
    result = run(
        "lissajous_curve",
        container=Container.with_uniform_margin(),
        placement=PlacementPolicy(),
        resolution_kwargs={"curve_points": _TEST_CURVE_POINTS},
        representation_config=RepresentationConfig(
            type=RepresentationType.TUBE, tube_radius=0.05, tube_sides=8,
        ),
    )
    obj = result.math_object
    assert obj.mesh is not None
    assert len(obj.mesh.vertices) > 0
    assert len(obj.mesh.faces) > 0

    edge_count: dict[tuple[int, int], int] = {}
    for face in obj.mesh.faces:
        for i in range(3):
            edge = tuple(sorted([int(face[i]), int(face[(i + 1) % 3])]))
            edge_count[edge] = edge_count.get(edge, 0) + 1

    non_manifold = {e: c for e, c in edge_count.items() if c != 2}
    assert len(non_manifold) == 0, (
        f"Mesh is not watertight: {len(non_manifold)} non-manifold edges"
    )
