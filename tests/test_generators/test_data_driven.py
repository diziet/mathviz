"""Tests for data-driven generators: heightmap, building_extrude, soundwave."""

import json
import wave
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from mathviz.core.generator import clear_registry, register
from mathviz.core.representation import RepresentationType
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


@pytest.fixture()
def test_png(tmp_path: Path) -> Path:
    """Create a small test PNG with a known gradient."""
    img = Image.new("L", (16, 16))
    for y in range(16):
        for x in range(16):
            img.putpixel((x, y), y * 16)  # gradient from 0 to 240
    path = tmp_path / "test.png"
    img.save(path)
    return path


@pytest.fixture()
def test_geojson_rect(tmp_path: Path) -> Path:
    """Create a GeoJSON file with a simple rectangle polygon."""
    geojson = {
        "type": "FeatureCollection",
        "features": [
            {
                "type": "Feature",
                "properties": {"height": 2.0},
                "geometry": {
                    "type": "Polygon",
                    "coordinates": [
                        [
                            [0.0, 0.0],
                            [1.0, 0.0],
                            [1.0, 1.0],
                            [0.0, 1.0],
                            [0.0, 0.0],
                        ]
                    ],
                },
            }
        ],
    }
    path = tmp_path / "buildings.geojson"
    path.write_text(json.dumps(geojson))
    return path


@pytest.fixture()
def test_wav(tmp_path: Path) -> Path:
    """Create a simple WAV file with a sine wave."""
    path = tmp_path / "test.wav"
    sample_rate = 8000
    duration = 0.1  # 100ms
    num_samples = int(sample_rate * duration)
    t = np.linspace(0, duration, num_samples, endpoint=False)
    signal = (np.sin(2 * np.pi * 440 * t) * 32767).astype(np.int16)

    with wave.open(str(path), "w") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(signal.tobytes())
    return path


# ---------------------------------------------------------------------------
# Heightmap tests
# ---------------------------------------------------------------------------


def test_heightmap_from_png_produces_mesh_with_z_range(test_png: Path) -> None:
    """Heightmap from a test PNG produces a mesh with z-range proportional to pixel values."""
    gen = HeightmapGenerator()
    obj = gen.generate(params={"input_file": str(test_png), "height_scale": 1.0})
    obj.validate_or_raise()

    assert obj.scalar_field is not None
    assert obj.scalar_field.shape == (16, 16)
    assert obj.scalar_field.dtype == np.float64

    # The gradient goes from 0 to 240, normalized to [0, 1]
    z_min = float(obj.scalar_field.min())
    z_max = float(obj.scalar_field.max())
    assert z_min == pytest.approx(0.0, abs=1e-6)
    assert z_max == pytest.approx(1.0, abs=1e-6)

    # Top rows should have higher values than bottom rows
    assert obj.scalar_field[-1, 0] > obj.scalar_field[0, 0]


def test_heightmap_metadata(test_png: Path) -> None:
    """Heightmap records correct metadata."""
    gen = HeightmapGenerator()
    obj = gen.generate(params={"input_file": str(test_png)}, seed=99)

    assert obj.generator_name == "heightmap"
    assert obj.category == "data_driven"
    assert obj.seed == 99
    assert obj.bounding_box is not None


def test_heightmap_representation() -> None:
    """Heightmap recommends HEIGHTMAP_RELIEF."""
    gen = HeightmapGenerator()
    rep = gen.get_default_representation()
    assert rep.type == RepresentationType.HEIGHTMAP_RELIEF


def test_heightmap_height_scale(test_png: Path) -> None:
    """Height scale parameter scales the z-range proportionally."""
    gen = HeightmapGenerator()
    obj1 = gen.generate(params={"input_file": str(test_png), "height_scale": 1.0})
    obj2 = gen.generate(params={"input_file": str(test_png), "height_scale": 2.0})

    assert obj1.scalar_field is not None
    assert obj2.scalar_field is not None
    np.testing.assert_allclose(
        obj2.scalar_field, obj1.scalar_field * 2.0, atol=1e-10,
    )


def test_heightmap_downsample(test_png: Path) -> None:
    """Downsample parameter reduces field resolution."""
    gen = HeightmapGenerator()
    obj = gen.generate(
        params={"input_file": str(test_png), "downsample": 2},
    )
    assert obj.scalar_field is not None
    assert obj.scalar_field.shape == (8, 8)


# ---------------------------------------------------------------------------
# Building extrude tests
# ---------------------------------------------------------------------------


def test_building_extrude_rectangle_produces_box(test_geojson_rect: Path) -> None:
    """Building extrude from a simple GeoJSON rectangle produces a box mesh."""
    gen = BuildingExtrudeGenerator()
    obj = gen.generate(params={"input_file": str(test_geojson_rect)})
    obj.validate_or_raise()

    assert obj.mesh is not None
    assert obj.mesh.vertices.dtype == np.float64
    assert obj.mesh.faces.dtype.kind in ("i", "u")

    # Rectangle has 4 vertices * 2 (top + bottom) = 8 vertices
    assert len(obj.mesh.vertices) == 8

    # Side faces: 4 sides * 2 triangles = 8
    # Top + bottom caps: 2 triangles each = 4
    assert len(obj.mesh.faces) == 12

    # Height should be 2.0 (from the feature property)
    z_values = obj.mesh.vertices[:, 2]
    assert float(z_values.min()) == pytest.approx(0.0)
    assert float(z_values.max()) == pytest.approx(2.0)

    # X and Y should span [0, 1]
    assert float(obj.mesh.vertices[:, 0].min()) == pytest.approx(0.0)
    assert float(obj.mesh.vertices[:, 0].max()) == pytest.approx(1.0)
    assert float(obj.mesh.vertices[:, 1].min()) == pytest.approx(0.0)
    assert float(obj.mesh.vertices[:, 1].max()) == pytest.approx(1.0)


def test_building_extrude_metadata(test_geojson_rect: Path) -> None:
    """Building extrude records correct metadata."""
    gen = BuildingExtrudeGenerator()
    obj = gen.generate(params={"input_file": str(test_geojson_rect)}, seed=7)

    assert obj.generator_name == "building_extrude"
    assert obj.category == "data_driven"
    assert obj.seed == 7
    assert obj.bounding_box is not None


def test_building_extrude_default_height(tmp_path: Path) -> None:
    """Default height is used when feature has no height property."""
    geojson = {
        "type": "FeatureCollection",
        "features": [
            {
                "type": "Feature",
                "properties": {},
                "geometry": {
                    "type": "Polygon",
                    "coordinates": [
                        [[0, 0], [1, 0], [1, 1], [0, 1], [0, 0]]
                    ],
                },
            }
        ],
    }
    path = tmp_path / "no_height.geojson"
    path.write_text(json.dumps(geojson))

    gen = BuildingExtrudeGenerator()
    obj = gen.generate(
        params={"input_file": str(path), "default_height": 5.0},
    )
    obj.validate_or_raise()
    assert float(obj.mesh.vertices[:, 2].max()) == pytest.approx(5.0)


# ---------------------------------------------------------------------------
# Soundwave tests
# ---------------------------------------------------------------------------


def test_soundwave_produces_curve(test_wav: Path) -> None:
    """Soundwave from a WAV file produces a valid curve."""
    gen = SoundwaveGenerator()
    obj = gen.generate(
        params={"input_file": str(test_wav)},
        num_samples=64,
    )
    obj.validate_or_raise()

    assert obj.curves is not None
    assert len(obj.curves) == 1
    curve = obj.curves[0]
    assert curve.points.shape == (64, 3)
    assert curve.points.dtype == np.float64
    assert not curve.closed

    # Amplitude should be non-negative (envelope is absolute values)
    assert np.all(curve.points[:, 1] >= 0)


def test_soundwave_metadata(test_wav: Path) -> None:
    """Soundwave records correct metadata."""
    gen = SoundwaveGenerator()
    obj = gen.generate(
        params={"input_file": str(test_wav)},
        seed=42,
        num_samples=32,
    )

    assert obj.generator_name == "soundwave"
    assert obj.category == "data_driven"
    assert obj.seed == 42
    assert obj.bounding_box is not None


def test_soundwave_representation() -> None:
    """Soundwave recommends TUBE representation."""
    gen = SoundwaveGenerator()
    rep = gen.get_default_representation()
    assert rep.type == RepresentationType.TUBE


# ---------------------------------------------------------------------------
# Error handling tests
# ---------------------------------------------------------------------------


def test_missing_input_file_raises_error() -> None:
    """Missing input file raises a clear error before any computation."""
    cases = [
        (HeightmapGenerator, "/nonexistent/file.png"),
        (BuildingExtrudeGenerator, "/nonexistent/file.geojson"),
        (SoundwaveGenerator, "/nonexistent/file.wav"),
    ]
    for gen_cls, path in cases:
        gen = gen_cls()
        with pytest.raises(FileNotFoundError, match="Input file not found"):
            gen.generate(params={"input_file": path})


def test_unsupported_format_raises_error(tmp_path: Path) -> None:
    """Unsupported file format raises a descriptive error."""
    bad_file = tmp_path / "data.xyz"
    bad_file.write_text("dummy")

    for gen_cls in [HeightmapGenerator, BuildingExtrudeGenerator, SoundwaveGenerator]:
        gen = gen_cls()
        with pytest.raises(ValueError, match="Unsupported file format"):
            gen.generate(params={"input_file": str(bad_file)})


def test_missing_input_file_param_raises_error() -> None:
    """Empty input_file parameter raises ValueError."""
    for gen_cls in [HeightmapGenerator, BuildingExtrudeGenerator, SoundwaveGenerator]:
        gen = gen_cls()
        with pytest.raises(ValueError, match="input_file parameter is required"):
            gen.generate()
