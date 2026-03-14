"""Tests for procedural generators: noise_surface, terrain, reaction_diffusion."""

import numpy as np
import pytest

from mathviz.core.generator import clear_registry, register
from mathviz.core.representation import RepresentationType
from mathviz.generators.procedural.noise_surface import NoiseSurfaceGenerator
from mathviz.generators.procedural.reaction_diffusion import (
    ReactionDiffusionGenerator,
)
from mathviz.generators.procedural.terrain import TerrainGenerator


@pytest.fixture(autouse=True)
def _clean_registry():
    """Reset registry and re-register procedural generators for each test."""
    clear_registry(suppress_discovery=True)
    register(NoiseSurfaceGenerator)
    register(TerrainGenerator)
    register(ReactionDiffusionGenerator)
    yield
    clear_registry(suppress_discovery=True)


_TEST_PIXEL_RESOLUTION = 32
_TEST_GRID_SIZE = 32
_TEST_TIMESTEPS = 500


# ---------------------------------------------------------------------------
# Noise surface: same seed produces identical mesh
# ---------------------------------------------------------------------------


def test_noise_surface_same_seed_identical() -> None:
    """Noise surface with same seed produces identical scalar field."""
    gen = NoiseSurfaceGenerator()
    obj1 = gen.generate(seed=123, pixel_resolution=_TEST_PIXEL_RESOLUTION)
    obj2 = gen.generate(seed=123, pixel_resolution=_TEST_PIXEL_RESOLUTION)

    obj1.validate_or_raise()
    obj2.validate_or_raise()

    assert obj1.scalar_field is not None
    assert obj2.scalar_field is not None
    np.testing.assert_array_equal(obj1.scalar_field, obj2.scalar_field)


def test_noise_surface_default_produces_valid_field() -> None:
    """Default parameters produce a valid non-empty scalar field."""
    gen = NoiseSurfaceGenerator()
    obj = gen.generate(pixel_resolution=_TEST_PIXEL_RESOLUTION)
    obj.validate_or_raise()

    assert obj.scalar_field is not None
    assert obj.scalar_field.shape == (_TEST_PIXEL_RESOLUTION, _TEST_PIXEL_RESOLUTION)
    assert obj.scalar_field.dtype == np.float64
    assert not np.all(obj.scalar_field == 0)


def test_noise_surface_metadata() -> None:
    """Noise surface records correct metadata."""
    gen = NoiseSurfaceGenerator()
    obj = gen.generate(seed=7, pixel_resolution=_TEST_PIXEL_RESOLUTION)

    assert obj.generator_name == "noise_surface"
    assert obj.category == "procedural"
    assert obj.seed == 7
    assert "frequency" in obj.parameters


def test_noise_surface_representation() -> None:
    """Noise surface recommends HEIGHTMAP_RELIEF."""
    gen = NoiseSurfaceGenerator()
    rep = gen.get_default_representation()
    assert rep.type == RepresentationType.HEIGHTMAP_RELIEF


# ---------------------------------------------------------------------------
# Terrain: plausible elevation range
# ---------------------------------------------------------------------------


def test_terrain_plausible_elevation_range() -> None:
    """Terrain heightmap has plausible range (not flat, not degenerate)."""
    gen = TerrainGenerator()
    obj = gen.generate(seed=42, pixel_resolution=_TEST_PIXEL_RESOLUTION)
    obj.validate_or_raise()

    assert obj.scalar_field is not None
    z_min = float(np.min(obj.scalar_field))
    z_max = float(np.max(obj.scalar_field))

    # Should not be flat
    assert z_max - z_min > 0.01, "Terrain is too flat"
    # Should not be degenerate (NaN or inf)
    assert np.all(np.isfinite(obj.scalar_field))
    # Normalized range should be within [0, height_scale]
    assert z_min >= -0.01
    assert z_max <= 1.01  # default height_scale=1.0


def test_terrain_same_seed_identical() -> None:
    """Terrain with same seed produces identical output."""
    gen = TerrainGenerator()
    obj1 = gen.generate(seed=99, pixel_resolution=_TEST_PIXEL_RESOLUTION)
    obj2 = gen.generate(seed=99, pixel_resolution=_TEST_PIXEL_RESOLUTION)

    assert obj1.scalar_field is not None
    assert obj2.scalar_field is not None
    np.testing.assert_array_equal(obj1.scalar_field, obj2.scalar_field)


def test_terrain_metadata() -> None:
    """Terrain records correct metadata."""
    gen = TerrainGenerator()
    obj = gen.generate(seed=1, pixel_resolution=_TEST_PIXEL_RESOLUTION)

    assert obj.generator_name == "terrain"
    assert obj.category == "procedural"
    assert obj.seed == 1
    assert "octaves" in obj.parameters


def test_terrain_custom_params() -> None:
    """Terrain with custom octaves produces valid output."""
    gen = TerrainGenerator()
    obj = gen.generate(
        params={"octaves": 2, "persistence": 0.3},
        seed=42,
        pixel_resolution=_TEST_PIXEL_RESOLUTION,
    )
    obj.validate_or_raise()
    assert obj.parameters["octaves"] == 2
    assert obj.parameters["persistence"] == 0.3


# ---------------------------------------------------------------------------
# Reaction-diffusion: non-trivial pattern
# ---------------------------------------------------------------------------


def test_reaction_diffusion_non_trivial_pattern() -> None:
    """Reaction-diffusion produces a non-uniform field (not all same value)."""
    gen = ReactionDiffusionGenerator()
    obj = gen.generate(
        seed=42,
        params={"timesteps": _TEST_TIMESTEPS},
        grid_size=_TEST_GRID_SIZE,
    )
    obj.validate_or_raise()

    assert obj.scalar_field is not None
    assert obj.scalar_field.dtype == np.float64
    # Not uniform
    std = float(np.std(obj.scalar_field))
    assert std > 1e-6, f"Field is uniform (std={std})"


def test_reaction_diffusion_same_seed_identical() -> None:
    """Reaction-diffusion with same seed produces identical output."""
    gen = ReactionDiffusionGenerator()
    kwargs = {"seed": 77, "params": {"timesteps": _TEST_TIMESTEPS}}
    obj1 = gen.generate(**kwargs, grid_size=_TEST_GRID_SIZE)
    obj2 = gen.generate(**kwargs, grid_size=_TEST_GRID_SIZE)

    assert obj1.scalar_field is not None
    assert obj2.scalar_field is not None
    np.testing.assert_array_equal(obj1.scalar_field, obj2.scalar_field)


def test_reaction_diffusion_metadata() -> None:
    """Reaction-diffusion records correct metadata."""
    gen = ReactionDiffusionGenerator()
    obj = gen.generate(
        seed=5,
        params={"timesteps": _TEST_TIMESTEPS},
        grid_size=_TEST_GRID_SIZE,
    )

    assert obj.generator_name == "reaction_diffusion"
    assert obj.category == "procedural"
    assert obj.seed == 5
    assert "feed_rate" in obj.parameters
    assert "kill_rate" in obj.parameters


def test_reaction_diffusion_values_in_range() -> None:
    """Reaction-diffusion values are in [0, height_scale]."""
    gen = ReactionDiffusionGenerator()
    obj = gen.generate(
        seed=42,
        params={"timesteps": _TEST_TIMESTEPS},
        grid_size=_TEST_GRID_SIZE,
    )

    assert obj.scalar_field is not None
    assert float(np.min(obj.scalar_field)) >= 0.0
    assert float(np.max(obj.scalar_field)) <= 1.0 + 1e-10


# ---------------------------------------------------------------------------
# Different seeds produce different outputs
# ---------------------------------------------------------------------------


def test_noise_surface_different_seeds_differ() -> None:
    """Different seeds produce different noise surfaces."""
    gen = NoiseSurfaceGenerator()
    obj1 = gen.generate(seed=1, pixel_resolution=_TEST_PIXEL_RESOLUTION)
    obj2 = gen.generate(seed=2, pixel_resolution=_TEST_PIXEL_RESOLUTION)

    assert obj1.scalar_field is not None
    assert obj2.scalar_field is not None
    assert not np.array_equal(obj1.scalar_field, obj2.scalar_field)


def test_terrain_different_seeds_differ() -> None:
    """Different seeds produce different terrain heightmaps."""
    gen = TerrainGenerator()
    obj1 = gen.generate(seed=10, pixel_resolution=_TEST_PIXEL_RESOLUTION)
    obj2 = gen.generate(seed=20, pixel_resolution=_TEST_PIXEL_RESOLUTION)

    assert obj1.scalar_field is not None
    assert obj2.scalar_field is not None
    assert not np.array_equal(obj1.scalar_field, obj2.scalar_field)


def test_reaction_diffusion_different_seeds_differ() -> None:
    """Different seeds produce different reaction-diffusion patterns."""
    gen = ReactionDiffusionGenerator()
    obj1 = gen.generate(
        seed=1,
        params={"timesteps": _TEST_TIMESTEPS},
        grid_size=_TEST_GRID_SIZE,
    )
    obj2 = gen.generate(
        seed=2,
        params={"timesteps": _TEST_TIMESTEPS},
        grid_size=_TEST_GRID_SIZE,
    )

    assert obj1.scalar_field is not None
    assert obj2.scalar_field is not None
    assert not np.array_equal(obj1.scalar_field, obj2.scalar_field)
