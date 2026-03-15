"""Tests for data-driven generator demo/placeholder mode.

Verifies that soundwave, heightmap, and building_extrude generators
produce valid output when no input_file is provided, using built-in
demo data.
"""

import json
import wave
from pathlib import Path

import numpy as np
import pytest

from mathviz.core.generator import clear_registry, register
from mathviz.generators.data_driven.building_extrude import BuildingExtrudeGenerator
from mathviz.generators.data_driven.heightmap import HeightmapGenerator
from mathviz.generators.data_driven.soundwave import SoundwaveGenerator


@pytest.fixture(autouse=True)
def _clean_registry():
    """Reset registry and re-register data-driven generators for each test."""
    clear_registry(suppress_discovery=True)
    register(HeightmapGenerator)
    register(BuildingExtrudeGenerator)
    register(SoundwaveGenerator)
    yield
    clear_registry(suppress_discovery=True)


def _get_geometry(obj, attr: str):
    """Extract the primary geometry array from a MathObject."""
    if attr == "curves":
        return obj.curves[0].points
    if attr == "mesh":
        return obj.mesh.vertices
    return obj.scalar_field


# ---------------------------------------------------------------------------
# Soundwave demo tests
# ---------------------------------------------------------------------------


def test_soundwave_demo_generates_successfully() -> None:
    """Soundwave generates successfully with no input_file parameter."""
    gen = SoundwaveGenerator()
    obj = gen.generate(num_samples=64)
    obj.validate_or_raise()


def test_soundwave_demo_produces_valid_curve() -> None:
    """Soundwave demo produces a valid MathObject with non-empty geometry."""
    gen = SoundwaveGenerator()
    obj = gen.generate(num_samples=64)

    assert obj.curves is not None
    assert len(obj.curves) == 1
    curve = obj.curves[0]
    assert curve.points.shape == (64, 3)
    assert curve.points.dtype == np.float64
    assert not curve.closed

    # Envelope values should be non-negative
    assert np.all(curve.points[:, 1] >= 0)


# ---------------------------------------------------------------------------
# Heightmap demo tests
# ---------------------------------------------------------------------------


def test_heightmap_demo_generates_successfully() -> None:
    """Heightmap generates successfully with no input_file parameter."""
    gen = HeightmapGenerator()
    obj = gen.generate()
    obj.validate_or_raise()


def test_heightmap_demo_produces_valid_field() -> None:
    """Heightmap demo produces a valid MathObject with non-empty geometry."""
    gen = HeightmapGenerator()
    obj = gen.generate()

    assert obj.scalar_field is not None
    assert obj.scalar_field.ndim == 2
    assert obj.scalar_field.shape[0] >= 2
    assert obj.scalar_field.shape[1] >= 2
    assert obj.scalar_field.dtype == np.float64
    assert obj.bounding_box is not None


# ---------------------------------------------------------------------------
# Building extrude demo tests
# ---------------------------------------------------------------------------


def test_building_extrude_demo_generates_successfully() -> None:
    """Building_extrude generates successfully with no input_file parameter."""
    gen = BuildingExtrudeGenerator()
    obj = gen.generate()
    obj.validate_or_raise()


def test_building_extrude_demo_produces_valid_mesh() -> None:
    """Building_extrude demo produces a valid MathObject with non-empty geometry."""
    gen = BuildingExtrudeGenerator()
    obj = gen.generate()

    assert obj.mesh is not None
    assert len(obj.mesh.vertices) > 0
    assert len(obj.mesh.faces) > 0
    assert obj.mesh.vertices.dtype == np.float64
    assert obj.bounding_box is not None

    # Should have multiple buildings (6 default)
    # Each building has 8 vertices (4 bottom + 4 top)
    assert len(obj.mesh.vertices) >= 8


# ---------------------------------------------------------------------------
# Parametrized determinism and seed-variation tests
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("gen_cls,attr,gen_kwargs", [
    (SoundwaveGenerator, "curves", {"num_samples": 64}),
    (HeightmapGenerator, "scalar_field", {}),
    (BuildingExtrudeGenerator, "mesh", {}),
])
def test_demo_is_deterministic(gen_cls, attr: str, gen_kwargs: dict) -> None:
    """Demo output is deterministic: same seed produces identical geometry."""
    gen = gen_cls()
    obj1 = gen.generate(seed=123, **gen_kwargs)
    obj2 = gen.generate(seed=123, **gen_kwargs)

    np.testing.assert_array_equal(
        _get_geometry(obj1, attr), _get_geometry(obj2, attr),
    )


@pytest.mark.parametrize("gen_cls,attr,gen_kwargs", [
    (SoundwaveGenerator, "curves", {"num_samples": 64}),
    (HeightmapGenerator, "scalar_field", {}),
    (BuildingExtrudeGenerator, "mesh", {}),
])
def test_demo_different_seeds_differ(gen_cls, attr: str, gen_kwargs: dict) -> None:
    """Different seeds produce different demo geometry."""
    gen = gen_cls()
    obj1 = gen.generate(seed=1, **gen_kwargs)
    obj2 = gen.generate(seed=2, **gen_kwargs)

    assert not np.array_equal(
        _get_geometry(obj1, attr), _get_geometry(obj2, attr),
    )


# ---------------------------------------------------------------------------
# Regression: input_file still works
# ---------------------------------------------------------------------------


@pytest.fixture()
def test_wav(tmp_path: Path) -> Path:
    """Create a simple WAV file with a sine wave."""
    path = tmp_path / "test.wav"
    sample_rate = 8000
    duration = 0.1
    num_samples = int(sample_rate * duration)
    t = np.linspace(0, duration, num_samples, endpoint=False)
    signal = (np.sin(2 * np.pi * 440 * t) * 32767).astype(np.int16)

    with wave.open(str(path), "w") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(signal.tobytes())
    return path


@pytest.fixture()
def test_png(tmp_path: Path) -> Path:
    """Create a small test PNG."""
    from PIL import Image

    img = Image.new("L", (16, 16))
    for y in range(16):
        for x in range(16):
            img.putpixel((x, y), y * 16)
    path = tmp_path / "test.png"
    img.save(path)
    return path


@pytest.fixture()
def test_geojson(tmp_path: Path) -> Path:
    """Create a GeoJSON file with a rectangle."""
    geojson = {
        "type": "FeatureCollection",
        "features": [{
            "type": "Feature",
            "properties": {"height": 2.0},
            "geometry": {
                "type": "Polygon",
                "coordinates": [
                    [[0, 0], [1, 0], [1, 1], [0, 1], [0, 0]],
                ],
            },
        }],
    }
    path = tmp_path / "buildings.geojson"
    path.write_text(json.dumps(geojson))
    return path


def test_soundwave_with_input_file_still_works(test_wav: Path) -> None:
    """Providing an input_file still works as before (no regression)."""
    gen = SoundwaveGenerator()
    obj = gen.generate(params={"input_file": str(test_wav)}, num_samples=32)
    obj.validate_or_raise()
    assert obj.curves is not None
    assert "demo_mode" not in obj.parameters


def test_heightmap_with_input_file_still_works(test_png: Path) -> None:
    """Providing an input_file still works as before (no regression)."""
    gen = HeightmapGenerator()
    obj = gen.generate(params={"input_file": str(test_png)})
    obj.validate_or_raise()
    assert obj.scalar_field is not None
    assert "demo_mode" not in obj.parameters


def test_building_extrude_with_input_file_still_works(test_geojson: Path) -> None:
    """Providing an input_file still works as before (no regression)."""
    gen = BuildingExtrudeGenerator()
    obj = gen.generate(params={"input_file": str(test_geojson)})
    obj.validate_or_raise()
    assert obj.mesh is not None
    assert "demo_mode" not in obj.parameters
