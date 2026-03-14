"""End-to-end fixture tests comparing regenerated outputs against reference files.

Each test regenerates a generator at the same seed and low resolution used
to create the reference fixtures, then compares vertex/point counts,
bounding boxes, and metadata fields against the stored reference.
"""

import json
from pathlib import Path
from typing import Any

import pytest

from mathviz.core.container import Container, PlacementPolicy
from mathviz.core.generator import clear_registry, register
from mathviz.core.representation import RepresentationConfig, RepresentationType
from mathviz.generators.attractors.lorenz import LorenzGenerator
from mathviz.generators.curves.lissajous_curve import LissajousCurveGenerator
from mathviz.generators.fractals.mandelbulb import MandelbulbGenerator
from mathviz.generators.implicit.gyroid import GyroidGenerator
from mathviz.generators.knots.torus_knot import TorusKnotGenerator
from mathviz.generators.number_theory.ulam_spiral import UlamSpiralGenerator
from mathviz.generators.parametric.torus import TorusGenerator
from mathviz.pipeline.runner import run

_GENERATOR_CLASSES = [
    TorusGenerator,
    GyroidGenerator,
    LorenzGenerator,
    MandelbulbGenerator,
    TorusKnotGenerator,
    UlamSpiralGenerator,
    LissajousCurveGenerator,
]


@pytest.fixture(autouse=True)
def _ensure_full_registry():
    """Re-register all needed generators before each test."""
    clear_registry(suppress_discovery=True)
    for gen_cls in _GENERATOR_CLASSES:
        register(gen_cls)
    yield
    clear_registry(suppress_discovery=True)

FIXTURES_DIR = Path(__file__).parent.parent / "fixtures"
REFERENCE_SUMMARY_PATH = FIXTURES_DIR / "reference_summary.json"
SEED = 42
VERTEX_COUNT_TOLERANCE = 0.01  # ±1%
BOUNDING_BOX_EPSILON = 1e-6


def _load_reference_summary() -> dict[str, Any]:
    """Load the reference summary JSON."""
    return json.loads(REFERENCE_SUMMARY_PATH.read_text(encoding="utf-8"))


REFERENCE = _load_reference_summary()

# Generator specs: maps name to low-res kwargs and optional representation
GENERATOR_SPECS: dict[str, dict[str, Any]] = {
    "torus": {
        "resolution_kwargs": {"grid_resolution": 16},
    },
    "gyroid": {
        "resolution_kwargs": {"voxel_resolution": 16},
    },
    "lorenz": {
        "resolution_kwargs": {"integration_steps": 2000},
    },
    "mandelbulb": {
        "resolution_kwargs": {"voxel_resolution": 16},
    },
    "torus_knot": {
        "resolution_kwargs": {"curve_points": 64},
    },
    "ulam_spiral": {
        "resolution_kwargs": {"num_points": 200},
        "representation": RepresentationConfig(
            type=RepresentationType.WEIGHTED_CLOUD,
        ),
    },
    "lissajous_curve": {
        "resolution_kwargs": {"curve_points": 64},
        "representation": RepresentationConfig(
            type=RepresentationType.TUBE, tube_radius=0.05,
        ),
    },
}


def _run_generator(name: str, seed: int = SEED) -> Any:
    """Run a generator through the pipeline and return the PipelineResult."""
    spec = GENERATOR_SPECS[name]
    return run(
        generator=name,
        seed=seed,
        resolution_kwargs=spec["resolution_kwargs"],
        container=Container.with_uniform_margin(),
        placement=PlacementPolicy(),
        representation_config=spec.get("representation"),
    )


def _get_geometry_count(result: Any) -> int:
    """Return vertex count (mesh) or point count (point cloud) from result."""
    obj = result.math_object
    if obj.mesh is not None:
        return len(obj.mesh.vertices)
    if obj.point_cloud is not None:
        return len(obj.point_cloud.points)
    raise ValueError(f"No geometry in result for {obj.generator_name}")


def _get_ref_count(ref: dict) -> int:
    """Return vertex_count or point_count from reference."""
    if "vertex_count" in ref:
        return ref["vertex_count"]
    return ref["point_count"]


@pytest.fixture(params=list(GENERATOR_SPECS.keys()))
def generator_name(request: pytest.FixtureRequest) -> str:
    """Parametrize over all fixture generators."""
    return request.param


class TestFixtureVertexCount:
    """Verify regenerated geometry matches reference vertex/point counts."""

    def test_count_within_tolerance(self, generator_name: str) -> None:
        """Regenerated vertex/point count is within ±1% of reference."""
        ref = REFERENCE[generator_name]
        result = _run_generator(generator_name)
        actual_count = _get_geometry_count(result)
        expected_count = _get_ref_count(ref)

        tolerance = max(1, int(expected_count * VERTEX_COUNT_TOLERANCE))
        assert abs(actual_count - expected_count) <= tolerance, (
            f"{generator_name}: count {actual_count} differs from "
            f"reference {expected_count} by more than ±{tolerance}"
        )


class TestFixtureBoundingBox:
    """Verify regenerated bounding boxes match reference within epsilon."""

    def test_bounding_box_within_epsilon(self, generator_name: str) -> None:
        """Regenerated bounding box matches reference within ε."""
        ref = REFERENCE[generator_name]
        result = _run_generator(generator_name)
        obj = result.math_object

        assert obj.bounding_box is not None, (
            f"{generator_name}: regenerated object has no bounding_box"
        )

        ref_bbox = ref["bounding_box"]
        ref_min = ref_bbox["min_corner"]
        ref_max = ref_bbox["max_corner"]
        actual_min = list(obj.bounding_box.min_corner)
        actual_max = list(obj.bounding_box.max_corner)

        for axis in range(3):
            assert abs(actual_min[axis] - ref_min[axis]) < BOUNDING_BOX_EPSILON, (
                f"{generator_name}: min_corner[{axis}] = {actual_min[axis]}, "
                f"reference = {ref_min[axis]}"
            )
            assert abs(actual_max[axis] - ref_max[axis]) < BOUNDING_BOX_EPSILON, (
                f"{generator_name}: max_corner[{axis}] = {actual_max[axis]}, "
                f"reference = {ref_max[axis]}"
            )


class TestFixtureMetadata:
    """Verify metadata fields match exactly across regenerations."""

    def test_generator_name_matches(self, generator_name: str) -> None:
        """Regenerated generator_name matches reference exactly."""
        ref = REFERENCE[generator_name]
        result = _run_generator(generator_name)
        assert result.math_object.generator_name == ref["generator_name"]

    def test_seed_matches(self, generator_name: str) -> None:
        """Regenerated seed matches reference exactly."""
        ref = REFERENCE[generator_name]
        result = _run_generator(generator_name)
        assert result.math_object.seed == ref["seed"]


class TestFixtureSanity:
    """Sanity checks that verify the tests actually detect mismatches."""

    def test_wrong_seed_produces_mismatch(self) -> None:
        """A deliberately wrong seed produces different output for lorenz."""
        ref = REFERENCE["lorenz"]
        result_wrong_seed = _run_generator("lorenz", seed=99)
        obj = result_wrong_seed.math_object

        ref_bbox = ref["bounding_box"]
        actual_min = list(obj.bounding_box.min_corner)
        ref_min = ref_bbox["min_corner"]

        has_mismatch = any(
            abs(actual_min[i] - ref_min[i]) > BOUNDING_BOX_EPSILON
            for i in range(3)
        )
        assert has_mismatch, (
            "Expected bounding box mismatch with wrong seed, but got match"
        )
