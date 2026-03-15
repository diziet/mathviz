"""Tests for the Burning Ship fractal heightmap generator."""

from pathlib import Path

import numpy as np
import pytest
import trimesh

from mathviz.core.container import Container, PlacementPolicy
from mathviz.core.generator import clear_registry, get_generator, register
from mathviz.core.representation import RepresentationConfig, RepresentationType
from mathviz.generators.fractals.burning_ship import BurningShipGenerator
from mathviz.generators.fractals.mandelbrot_heightmap import (
    MandelbrotHeightmapGenerator,
)
from mathviz.pipeline.runner import ExportConfig, run


@pytest.fixture(autouse=True)
def _clean_registry():
    """Reset registry and re-register generators for each test."""
    clear_registry(suppress_discovery=True)
    register(BurningShipGenerator)
    register(MandelbrotHeightmapGenerator)
    yield
    clear_registry(suppress_discovery=True)


@pytest.fixture
def burning_ship() -> BurningShipGenerator:
    """Return a BurningShipGenerator instance."""
    return BurningShipGenerator()


_TEST_PIXEL_RESOLUTION = 64


# ---------------------------------------------------------------------------
# Produces a valid scalar field
# ---------------------------------------------------------------------------


def test_produces_valid_scalar_field(
    burning_ship: BurningShipGenerator,
) -> None:
    """Default parameters produce a non-empty scalar field of correct shape."""
    obj = burning_ship.generate(pixel_resolution=_TEST_PIXEL_RESOLUTION)
    obj.validate_or_raise()

    assert obj.scalar_field is not None
    assert obj.scalar_field.shape == (
        _TEST_PIXEL_RESOLUTION,
        _TEST_PIXEL_RESOLUTION,
    )
    assert obj.scalar_field.dtype == np.float64
    assert not np.all(obj.scalar_field == 0)


# ---------------------------------------------------------------------------
# Output is distinct from Mandelbrot (asymmetric)
# ---------------------------------------------------------------------------


def test_output_distinct_from_mandelbrot() -> None:
    """Burning Ship produces a different field than Mandelbrot at same region."""
    shared_params_bs = {
        "center_x": -0.5,
        "center_y": 0.0,
        "zoom": 1.0,
        "max_iterations": 64,
        "height_scale": 1.0,
    }
    shared_params_mb = {
        "center_real": -0.5,
        "center_imag": 0.0,
        "zoom": 1.0,
        "max_iterations": 64,
        "height_scale": 1.0,
        "smoothing": False,
    }

    bs = BurningShipGenerator()
    mb = MandelbrotHeightmapGenerator()

    bs_obj = bs.generate(params=shared_params_bs, pixel_resolution=32)
    mb_obj = mb.generate(params=shared_params_mb, pixel_resolution=32)

    assert bs_obj.scalar_field is not None
    assert mb_obj.scalar_field is not None
    assert not np.array_equal(bs_obj.scalar_field, mb_obj.scalar_field)


def test_burning_ship_is_asymmetric(
    burning_ship: BurningShipGenerator,
) -> None:
    """Burning Ship fractal is not symmetric about the real axis.

    The Mandelbrot set is symmetric about Im=0, but the Burning Ship
    is not — this is its defining visual characteristic.
    """
    obj = burning_ship.generate(
        params={"center_x": -0.4, "center_y": 0.0, "zoom": 1.0},
        pixel_resolution=_TEST_PIXEL_RESOLUTION,
    )
    assert obj.scalar_field is not None
    field = obj.scalar_field

    # Flip vertically (about the real axis)
    flipped = field[::-1, :]
    assert not np.array_equal(field, flipped), (
        "Burning Ship should be asymmetric about the real axis"
    )


# ---------------------------------------------------------------------------
# Higher pixel_resolution produces a finer grid
# ---------------------------------------------------------------------------


def test_higher_resolution_produces_finer_grid(
    burning_ship: BurningShipGenerator,
) -> None:
    """Higher pixel_resolution produces a larger output grid."""
    obj_low = burning_ship.generate(pixel_resolution=32)
    obj_high = burning_ship.generate(pixel_resolution=64)

    assert obj_low.scalar_field is not None
    assert obj_high.scalar_field is not None
    assert obj_low.scalar_field.shape == (32, 32)
    assert obj_high.scalar_field.shape == (64, 64)


# ---------------------------------------------------------------------------
# Registers and renders successfully
# ---------------------------------------------------------------------------


def test_burning_ship_registered() -> None:
    """burning_ship is discoverable via the registry."""
    gen_cls = get_generator("burning_ship")
    assert gen_cls is BurningShipGenerator


def test_default_representation_is_heightmap_relief(
    burning_ship: BurningShipGenerator,
) -> None:
    """Default representation is HEIGHTMAP_RELIEF."""
    rep = burning_ship.get_default_representation()
    assert rep.type == RepresentationType.HEIGHTMAP_RELIEF


def test_full_pipeline_renders(tmp_path: Path) -> None:
    """Full pipeline with HEIGHTMAP_RELIEF produces a valid STL."""
    out_path = tmp_path / "burning_ship.stl"

    result = run(
        "burning_ship",
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
# Metadata and determinism
# ---------------------------------------------------------------------------


def test_metadata_recorded(
    burning_ship: BurningShipGenerator,
) -> None:
    """Generator name, category, and parameters are recorded."""
    obj = burning_ship.generate(pixel_resolution=_TEST_PIXEL_RESOLUTION)
    assert obj.generator_name == "burning_ship"
    assert obj.category == "fractals"
    assert obj.parameters["center_x"] == -0.4
    assert obj.parameters["center_y"] == -0.6
    assert obj.parameters["zoom"] == 3.0
    assert obj.parameters["max_iterations"] == 256
    assert obj.parameters["pixel_resolution"] == _TEST_PIXEL_RESOLUTION


def test_determinism(
    burning_ship: BurningShipGenerator,
) -> None:
    """Same parameters produce identical scalar fields."""
    obj1 = burning_ship.generate(seed=42, pixel_resolution=_TEST_PIXEL_RESOLUTION)
    obj2 = burning_ship.generate(seed=42, pixel_resolution=_TEST_PIXEL_RESOLUTION)

    assert obj1.scalar_field is not None and obj2.scalar_field is not None
    np.testing.assert_array_equal(obj1.scalar_field, obj2.scalar_field)


# ---------------------------------------------------------------------------
# Input validation
# ---------------------------------------------------------------------------


def test_zero_zoom_raises(burning_ship: BurningShipGenerator) -> None:
    """Zero zoom raises ValueError."""
    with pytest.raises(ValueError, match="zoom must be positive"):
        burning_ship.generate(params={"zoom": 0.0})


def test_negative_height_scale_raises(burning_ship: BurningShipGenerator) -> None:
    """Negative height_scale raises ValueError."""
    with pytest.raises(ValueError, match="height_scale must be positive"):
        burning_ship.generate(params={"height_scale": -1.0})


def test_low_pixel_resolution_raises(burning_ship: BurningShipGenerator) -> None:
    """pixel_resolution below minimum raises ValueError."""
    with pytest.raises(ValueError, match="pixel_resolution must be >= 4"):
        burning_ship.generate(pixel_resolution=2)
