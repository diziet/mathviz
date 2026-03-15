"""Tests for the IFS fractal generator."""

from pathlib import Path

import numpy as np
import pytest

from mathviz.core.container import Container, PlacementPolicy
from mathviz.core.generator import clear_registry, get_generator, register
from mathviz.core.representation import RepresentationConfig, RepresentationType
from mathviz.generators.fractals.ifs_fractal import IFSFractalGenerator
from mathviz.pipeline.runner import ExportConfig, run

_TEST_NUM_POINTS = 5_000


@pytest.fixture(autouse=True)
def _clean_registry():
    """Reset registry and register IFS fractal for each test."""
    clear_registry(suppress_discovery=True)
    register(IFSFractalGenerator)
    yield
    clear_registry(suppress_discovery=True)


@pytest.fixture
def gen() -> IFSFractalGenerator:
    """Return an IFSFractalGenerator instance."""
    return IFSFractalGenerator()


# ---------------------------------------------------------------------------
# Barnsley fern produces expected shape bounds
# ---------------------------------------------------------------------------


def test_barnsley_fern_shape_bounds(gen: IFSFractalGenerator) -> None:
    """Barnsley fern point cloud has tall-and-narrow shape."""
    obj = gen.generate(num_points=_TEST_NUM_POINTS)
    obj.validate_or_raise()
    assert obj.point_cloud is not None

    points = obj.point_cloud.points
    assert points.shape == (_TEST_NUM_POINTS, 3)
    assert points.dtype == np.float64

    bbox = obj.bounding_box
    assert bbox is not None
    width = bbox.max_corner[0] - bbox.min_corner[0]
    height = bbox.max_corner[1] - bbox.min_corner[1]
    # Fern is taller than wide
    assert height > width


def test_barnsley_fern_2d_extruded(gen: IFSFractalGenerator) -> None:
    """2D extruded mode produces thin z-extent."""
    obj = gen.generate(
        params={"dimensions": "2d_extruded"},
        num_points=_TEST_NUM_POINTS,
    )
    obj.validate_or_raise()
    assert obj.point_cloud is not None

    bbox = obj.bounding_box
    assert bbox is not None
    z_extent = bbox.max_corner[2] - bbox.min_corner[2]
    y_extent = bbox.max_corner[1] - bbox.min_corner[1]
    # Z should be much thinner than Y (fern height)
    assert z_extent < y_extent * 0.5


# ---------------------------------------------------------------------------
# Different presets produce distinct point distributions
# ---------------------------------------------------------------------------


def test_different_presets_produce_distinct_distributions(
    gen: IFSFractalGenerator,
) -> None:
    """Different presets produce point clouds with different centroids."""
    presets = ["barnsley_fern", "maple_leaf", "spiral"]
    centroids = []
    for preset in presets:
        obj = gen.generate(
            params={"preset": preset},
            num_points=_TEST_NUM_POINTS,
        )
        centroids.append(obj.point_cloud.points.mean(axis=0))

    # Each pair of presets should have different centroids
    for i in range(len(centroids)):
        for j in range(i + 1, len(centroids)):
            assert not np.allclose(centroids[i], centroids[j], atol=0.1), (
                f"Presets {presets[i]} and {presets[j]} produced "
                f"similar centroids"
            )


# ---------------------------------------------------------------------------
# Output is seed-dependent
# ---------------------------------------------------------------------------


def test_seed_determinism(gen: IFSFractalGenerator) -> None:
    """Same seed produces identical output."""
    obj1 = gen.generate(seed=42, num_points=_TEST_NUM_POINTS)
    obj2 = gen.generate(seed=42, num_points=_TEST_NUM_POINTS)
    np.testing.assert_array_equal(
        obj1.point_cloud.points, obj2.point_cloud.points,
    )


def test_different_seeds_differ(gen: IFSFractalGenerator) -> None:
    """Different seeds produce different point clouds."""
    obj1 = gen.generate(seed=42, num_points=_TEST_NUM_POINTS)
    obj2 = gen.generate(seed=99, num_points=_TEST_NUM_POINTS)
    assert not np.array_equal(
        obj1.point_cloud.points, obj2.point_cloud.points,
    )


# ---------------------------------------------------------------------------
# Registers and renders successfully
# ---------------------------------------------------------------------------


def test_registered() -> None:
    """ifs_fractal is discoverable via the registry."""
    gen_cls = get_generator("ifs_fractal")
    assert gen_cls is IFSFractalGenerator


def test_alias_registered() -> None:
    """Aliases are discoverable via the registry."""
    assert get_generator("ifs") is IFSFractalGenerator
    assert get_generator("barnsley_fern") is IFSFractalGenerator


def test_full_pipeline_renders(tmp_path: Path) -> None:
    """Full pipeline with SPARSE_SHELL produces a valid PLY."""
    out_path = tmp_path / "ifs.ply"
    result = run(
        "ifs_fractal",
        resolution_kwargs={"num_points": _TEST_NUM_POINTS},
        container=Container.with_uniform_margin(),
        placement=PlacementPolicy(),
        representation_config=RepresentationConfig(
            type=RepresentationType.SPARSE_SHELL,
        ),
        export_config=ExportConfig(
            path=out_path, export_type="point_cloud",
        ),
    )
    assert result.export_path == out_path
    assert out_path.exists()
    assert out_path.stat().st_size > 0


# ---------------------------------------------------------------------------
# Default representation
# ---------------------------------------------------------------------------


def test_default_representation(gen: IFSFractalGenerator) -> None:
    """Default representation is SPARSE_SHELL."""
    rep = gen.get_default_representation()
    assert rep.type == RepresentationType.SPARSE_SHELL


# ---------------------------------------------------------------------------
# Metadata
# ---------------------------------------------------------------------------


def test_metadata_recorded(gen: IFSFractalGenerator) -> None:
    """Generator name, category, and parameters are recorded."""
    obj = gen.generate(num_points=_TEST_NUM_POINTS)
    assert obj.generator_name == "ifs_fractal"
    assert obj.category == "fractals"
    assert obj.parameters["preset"] == "barnsley_fern"
    assert obj.parameters["dimensions"] == "3d"
    assert obj.parameters["num_points"] == _TEST_NUM_POINTS


# ---------------------------------------------------------------------------
# Input validation
# ---------------------------------------------------------------------------


def test_invalid_preset_raises(gen: IFSFractalGenerator) -> None:
    """Invalid preset raises ValueError."""
    with pytest.raises(ValueError, match="preset must be one of"):
        gen.generate(params={"preset": "invalid"})


def test_invalid_dimensions_raises(gen: IFSFractalGenerator) -> None:
    """Invalid dimensions raises ValueError."""
    with pytest.raises(ValueError, match="dimensions must be one of"):
        gen.generate(params={"dimensions": "4d"})


def test_too_few_points_raises(gen: IFSFractalGenerator) -> None:
    """Too few points raises ValueError."""
    with pytest.raises(ValueError, match="num_points must be"):
        gen.generate(num_points=10)


def test_no_nan_in_output(gen: IFSFractalGenerator) -> None:
    """Output contains no NaN values."""
    obj = gen.generate(num_points=_TEST_NUM_POINTS)
    assert not np.any(np.isnan(obj.point_cloud.points))


def test_custom_missing_params_raises(gen: IFSFractalGenerator) -> None:
    """Custom preset without required params raises ValueError."""
    with pytest.raises(ValueError, match="Custom preset requires"):
        gen.generate(params={"preset": "custom"})


def test_custom_negative_probs_raises(gen: IFSFractalGenerator) -> None:
    """Custom preset with negative probabilities raises ValueError."""
    with pytest.raises(ValueError, match="non-negative"):
        gen.generate(params={
            "preset": "custom",
            "matrices": [np.eye(2), np.eye(2)],
            "offsets": [np.zeros(2), np.zeros(2)],
            "probabilities": [1.5, -0.5],
        })


def test_custom_invalid_matrix_shape_raises(gen: IFSFractalGenerator) -> None:
    """Custom preset with non-square matrix raises ValueError."""
    with pytest.raises(ValueError, match="shape"):
        gen.generate(params={
            "preset": "custom",
            "matrices": [np.ones((2, 3))],
            "offsets": [np.zeros(2)],
            "probabilities": [1.0],
        })


def test_custom_mixed_matrix_dims_raises(gen: IFSFractalGenerator) -> None:
    """Custom preset with mixed 2×2 and 3×3 matrices raises ValueError."""
    with pytest.raises(ValueError, match="differs"):
        gen.generate(params={
            "preset": "custom",
            "matrices": [np.eye(2), np.eye(3)],
            "offsets": [np.zeros(2), np.zeros(3)],
            "probabilities": [0.5, 0.5],
        })


def test_custom_3d_with_2d_extruded_raises(gen: IFSFractalGenerator) -> None:
    """3×3 custom matrices with 2d_extruded raises ValueError."""
    with pytest.raises(ValueError, match="incompatible"):
        gen.generate(params={
            "preset": "custom",
            "dimensions": "2d_extruded",
            "matrices": [np.eye(3), np.eye(3)],
            "offsets": [np.zeros(3), np.ones(3)],
            "probabilities": [0.5, 0.5],
        })
