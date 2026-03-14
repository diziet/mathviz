"""Tests for fractal generators, starting with mandelbrot_heightmap."""

from pathlib import Path

import numpy as np
import pytest
import trimesh

from mathviz.core.container import Container, PlacementPolicy
from mathviz.core.generator import clear_registry, get_generator, register
from mathviz.core.representation import RepresentationConfig, RepresentationType
from mathviz.generators.fractals.mandelbrot_heightmap import (
    MandelbrotHeightmapGenerator,
)
from mathviz.pipeline.runner import ExportConfig, run


@pytest.fixture(autouse=True)
def _clean_registry():
    """Reset registry and re-register mandelbrot_heightmap for each test."""
    clear_registry(suppress_discovery=True)
    register(MandelbrotHeightmapGenerator)
    yield
    clear_registry(suppress_discovery=True)


@pytest.fixture
def mandelbrot() -> MandelbrotHeightmapGenerator:
    """Return a MandelbrotHeightmapGenerator instance."""
    return MandelbrotHeightmapGenerator()


# Use low resolution for test speed
_TEST_PIXEL_RESOLUTION = 64


# ---------------------------------------------------------------------------
# Default parameters produce a non-empty scalar field
# ---------------------------------------------------------------------------


def test_default_produces_nonempty_scalar_field(
    mandelbrot: MandelbrotHeightmapGenerator,
) -> None:
    """Default parameters produce a non-empty scalar field of correct shape."""
    obj = mandelbrot.generate(pixel_resolution=_TEST_PIXEL_RESOLUTION)
    obj.validate_or_raise()

    assert obj.scalar_field is not None
    assert obj.scalar_field.shape == (
        _TEST_PIXEL_RESOLUTION,
        _TEST_PIXEL_RESOLUTION,
    )
    assert obj.scalar_field.dtype == np.float64
    assert not np.all(obj.scalar_field == 0)


# ---------------------------------------------------------------------------
# Smooth iteration count produces non-integer values
# ---------------------------------------------------------------------------


def test_smooth_iteration_produces_non_integer_values(
    mandelbrot: MandelbrotHeightmapGenerator,
) -> None:
    """Smoothing=True produces non-integer iteration counts for escaped points."""
    obj = mandelbrot.generate(
        params={"smoothing": True},
        pixel_resolution=_TEST_PIXEL_RESOLUTION,
    )
    assert obj.scalar_field is not None

    # Filter to escaped points (non-zero iteration count)
    escaped = obj.scalar_field[obj.scalar_field > 0]
    assert len(escaped) > 0

    # At least some escaped points should have fractional parts
    fractional_parts = escaped - np.floor(escaped)
    has_fractional = np.any(fractional_parts > 1e-10)
    assert has_fractional, "Smoothing should produce non-integer iteration counts"


def test_no_smoothing_produces_integer_values(
    mandelbrot: MandelbrotHeightmapGenerator,
) -> None:
    """Smoothing=False produces integer iteration counts."""
    obj = mandelbrot.generate(
        params={"smoothing": False},
        pixel_resolution=_TEST_PIXEL_RESOLUTION,
    )
    assert obj.scalar_field is not None

    escaped = obj.scalar_field[obj.scalar_field > 0]
    assert len(escaped) > 0

    fractional_parts = escaped - np.floor(escaped)
    np.testing.assert_allclose(
        fractional_parts, 0.0, atol=1e-10,
        err_msg="Without smoothing, all values should be integers",
    )


# ---------------------------------------------------------------------------
# Full pipeline with HEIGHTMAP_RELIEF produces a valid mesh
# ---------------------------------------------------------------------------


def test_full_pipeline_heightmap_relief(tmp_path: Path) -> None:
    """Full pipeline with HEIGHTMAP_RELIEF produces a valid STL."""
    out_path = tmp_path / "mandelbrot.stl"

    result = run(
        "mandelbrot_heightmap",
        resolution_kwargs={"pixel_resolution": _TEST_PIXEL_RESOLUTION},
        container=Container.with_uniform_margin(),
        placement=PlacementPolicy(),
        representation_config=RepresentationConfig(
            type=RepresentationType.HEIGHTMAP_RELIEF,
        ),
        export_config=ExportConfig(path=out_path, export_type="mesh"),
    )

    assert result.export_path == out_path
    assert out_path.exists()
    assert out_path.stat().st_size > 0

    reimported = trimesh.load(str(out_path), file_type="stl")
    assert len(reimported.faces) > 0
    assert len(reimported.vertices) > 0


# ---------------------------------------------------------------------------
# height_scale multiplier scales z-range proportionally
# ---------------------------------------------------------------------------


def test_height_scale_scales_z_range(
    mandelbrot: MandelbrotHeightmapGenerator,
) -> None:
    """height_scale multiplier scales the z-range of the scalar field."""
    obj_1x = mandelbrot.generate(
        params={"height_scale": 1.0},
        pixel_resolution=_TEST_PIXEL_RESOLUTION,
    )
    obj_3x = mandelbrot.generate(
        params={"height_scale": 3.0},
        pixel_resolution=_TEST_PIXEL_RESOLUTION,
    )

    assert obj_1x.scalar_field is not None
    assert obj_3x.scalar_field is not None

    range_1x = float(np.max(obj_1x.scalar_field) - np.min(obj_1x.scalar_field))
    range_3x = float(np.max(obj_3x.scalar_field) - np.min(obj_3x.scalar_field))

    assert range_1x > 0, "Field should have non-zero range"
    np.testing.assert_allclose(
        range_3x / range_1x, 3.0, rtol=1e-10,
        err_msg="height_scale=3 should triple the z-range",
    )


def test_height_scale_affects_mesh_z_range(tmp_path: Path) -> None:
    """height_scale multiplier scales z-range of the resulting mesh."""
    from mathviz.pipeline import representation_strategy

    gen = MandelbrotHeightmapGenerator()
    obj_1x = gen.generate(
        params={"height_scale": 1.0},
        pixel_resolution=32,
    )
    obj_2x = gen.generate(
        params={"height_scale": 2.0},
        pixel_resolution=32,
    )

    config = RepresentationConfig(type=RepresentationType.HEIGHTMAP_RELIEF)
    mesh_1x = representation_strategy.apply(obj_1x, config)
    mesh_2x = representation_strategy.apply(obj_2x, config)

    assert mesh_1x.mesh is not None and mesh_2x.mesh is not None
    z_vals_1x = mesh_1x.mesh.vertices[:, 2]
    z_vals_2x = mesh_2x.mesh.vertices[:, 2]
    z_range_1x = z_vals_1x.max() - z_vals_1x.min()
    z_range_2x = z_vals_2x.max() - z_vals_2x.min()

    assert z_range_1x > 0
    np.testing.assert_allclose(
        z_range_2x / z_range_1x, 2.0, rtol=1e-10,
    )


# ---------------------------------------------------------------------------
# Zooming in changes the detail pattern
# ---------------------------------------------------------------------------


def test_zoom_changes_scalar_field(
    mandelbrot: MandelbrotHeightmapGenerator,
) -> None:
    """Zooming in produces a different scalar field."""
    obj_default = mandelbrot.generate(
        params={"zoom": 1.0},
        pixel_resolution=_TEST_PIXEL_RESOLUTION,
    )
    obj_zoomed = mandelbrot.generate(
        params={"zoom": 10.0},
        pixel_resolution=_TEST_PIXEL_RESOLUTION,
    )

    assert obj_default.scalar_field is not None
    assert obj_zoomed.scalar_field is not None
    assert not np.array_equal(obj_default.scalar_field, obj_zoomed.scalar_field)


# ---------------------------------------------------------------------------
# Metadata recording
# ---------------------------------------------------------------------------


def test_metadata_recorded(
    mandelbrot: MandelbrotHeightmapGenerator,
) -> None:
    """Generator name, category, and parameters are recorded."""
    obj = mandelbrot.generate(pixel_resolution=_TEST_PIXEL_RESOLUTION)
    assert obj.generator_name == "mandelbrot_heightmap"
    assert obj.category == "fractals"
    assert obj.parameters["center_real"] == -0.5
    assert obj.parameters["center_imag"] == 0.0
    assert obj.parameters["zoom"] == 1.0
    assert obj.parameters["max_iterations"] == 256
    assert obj.parameters["pixel_resolution"] == _TEST_PIXEL_RESOLUTION


def test_seed_recorded(
    mandelbrot: MandelbrotHeightmapGenerator,
) -> None:
    """Seed is recorded in the MathObject."""
    obj = mandelbrot.generate(seed=999, pixel_resolution=_TEST_PIXEL_RESOLUTION)
    assert obj.seed == 999


# ---------------------------------------------------------------------------
# Registry: mandelbrot_heightmap is discoverable
# ---------------------------------------------------------------------------


def test_mandelbrot_registered() -> None:
    """mandelbrot_heightmap is discoverable via the registry."""
    gen_cls = get_generator("mandelbrot_heightmap")
    assert gen_cls is MandelbrotHeightmapGenerator


def test_default_representation_is_heightmap_relief(
    mandelbrot: MandelbrotHeightmapGenerator,
) -> None:
    """Default representation is HEIGHTMAP_RELIEF."""
    rep = mandelbrot.get_default_representation()
    assert rep.type == RepresentationType.HEIGHTMAP_RELIEF


# ---------------------------------------------------------------------------
# Determinism
# ---------------------------------------------------------------------------


def test_determinism(
    mandelbrot: MandelbrotHeightmapGenerator,
) -> None:
    """Same parameters produce identical scalar fields."""
    obj1 = mandelbrot.generate(seed=42, pixel_resolution=_TEST_PIXEL_RESOLUTION)
    obj2 = mandelbrot.generate(seed=42, pixel_resolution=_TEST_PIXEL_RESOLUTION)

    assert obj1.scalar_field is not None and obj2.scalar_field is not None
    np.testing.assert_array_equal(obj1.scalar_field, obj2.scalar_field)


# ---------------------------------------------------------------------------
# Input validation
# ---------------------------------------------------------------------------


def test_zero_zoom_raises(
    mandelbrot: MandelbrotHeightmapGenerator,
) -> None:
    """Zero zoom raises ValueError."""
    with pytest.raises(ValueError, match="zoom must be positive"):
        mandelbrot.generate(params={"zoom": 0.0})


def test_negative_zoom_raises(
    mandelbrot: MandelbrotHeightmapGenerator,
) -> None:
    """Negative zoom raises ValueError."""
    with pytest.raises(ValueError, match="zoom must be positive"):
        mandelbrot.generate(params={"zoom": -1.0})


def test_low_pixel_resolution_raises(
    mandelbrot: MandelbrotHeightmapGenerator,
) -> None:
    """pixel_resolution below minimum raises ValueError."""
    with pytest.raises(ValueError, match="pixel_resolution must be >= 4"):
        mandelbrot.generate(pixel_resolution=2)


def test_zero_max_iterations_raises(
    mandelbrot: MandelbrotHeightmapGenerator,
) -> None:
    """max_iterations=0 raises ValueError."""
    with pytest.raises(ValueError, match="max_iterations must be >= 1"):
        mandelbrot.generate(params={"max_iterations": 0})


def test_zero_height_scale_raises(
    mandelbrot: MandelbrotHeightmapGenerator,
) -> None:
    """height_scale=0 raises ValueError."""
    with pytest.raises(ValueError, match="height_scale must be positive"):
        mandelbrot.generate(params={"height_scale": 0.0})


def test_negative_height_scale_raises(
    mandelbrot: MandelbrotHeightmapGenerator,
) -> None:
    """Negative height_scale raises ValueError."""
    with pytest.raises(ValueError, match="height_scale must be positive"):
        mandelbrot.generate(params={"height_scale": -1.0})


# ---------------------------------------------------------------------------
# Smoothing clamp — no negative values
# ---------------------------------------------------------------------------


def test_smoothing_no_negative_values(
    mandelbrot: MandelbrotHeightmapGenerator,
) -> None:
    """Smoothed field has no negative values, even at low zoom."""
    obj = mandelbrot.generate(
        params={"zoom": 0.5, "smoothing": True},
        pixel_resolution=_TEST_PIXEL_RESOLUTION,
    )
    assert obj.scalar_field is not None
    assert np.all(obj.scalar_field >= 0.0), (
        f"Field min is {np.min(obj.scalar_field)}, expected >= 0"
    )


# ---------------------------------------------------------------------------
# Bounding box
# ---------------------------------------------------------------------------


def test_bounding_box_set(
    mandelbrot: MandelbrotHeightmapGenerator,
) -> None:
    """Bounding box is populated after generation."""
    obj = mandelbrot.generate(pixel_resolution=_TEST_PIXEL_RESOLUTION)
    assert obj.bounding_box is not None
    assert obj.bounding_box.min_corner[0] == 0.0
    assert obj.bounding_box.max_corner[0] == 1.0
