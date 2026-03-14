"""Terrain heightmap generator.

Multi-octave simplex noise heightmap simulating natural terrain. Uses fractal
Brownian motion (fBm) with configurable octaves, persistence, and lacunarity
for realistic terrain detail at multiple scales.
"""

import logging
from typing import Any

import numpy as np

from mathviz.core.generator import GeneratorBase, register
from mathviz.core.math_object import BoundingBox, MathObject
from mathviz.core.representation import RepresentationConfig, RepresentationType
from mathviz.generators.procedural.simplex import evaluate_2d_grid_octaves

logger = logging.getLogger(__name__)

_DEFAULT_OCTAVES = 6
_DEFAULT_PERSISTENCE = 0.5
_DEFAULT_LACUNARITY = 2.0
_DEFAULT_BASE_FREQUENCY = 3.0
_DEFAULT_HEIGHT_SCALE = 1.0
_DEFAULT_PIXEL_RESOLUTION = 256
_MIN_PIXEL_RESOLUTION = 4
_MIN_OCTAVES = 1
_MAX_OCTAVES = 12
_MIN_BASE_FREQUENCY = 0.01


def _validate_params(
    octaves: int,
    persistence: float,
    lacunarity: float,
    base_frequency: float,
    height_scale: float,
    pixel_resolution: int,
) -> None:
    """Validate terrain parameters."""
    if octaves < _MIN_OCTAVES:
        raise ValueError(f"octaves must be >= {_MIN_OCTAVES}, got {octaves}")
    if octaves > _MAX_OCTAVES:
        raise ValueError(f"octaves must be <= {_MAX_OCTAVES}, got {octaves}")
    if not 0 < persistence <= 1.0:
        raise ValueError(
            f"persistence must be in (0, 1], got {persistence}"
        )
    if lacunarity <= 0:
        raise ValueError(f"lacunarity must be positive, got {lacunarity}")
    if base_frequency < _MIN_BASE_FREQUENCY:
        raise ValueError(
            f"base_frequency must be >= {_MIN_BASE_FREQUENCY}, "
            f"got {base_frequency}"
        )
    if height_scale <= 0:
        raise ValueError(f"height_scale must be positive, got {height_scale}")
    if pixel_resolution < _MIN_PIXEL_RESOLUTION:
        raise ValueError(
            f"pixel_resolution must be >= {_MIN_PIXEL_RESOLUTION}, "
            f"got {pixel_resolution}"
        )


def _compute_terrain_field(
    octaves: int,
    persistence: float,
    lacunarity: float,
    base_frequency: float,
    height_scale: float,
    pixel_resolution: int,
    seed: int,
) -> np.ndarray:
    """Generate a multi-octave noise terrain heightmap."""
    extent = base_frequency
    field = evaluate_2d_grid_octaves(
        resolution=pixel_resolution,
        x_range=(0.0, extent),
        y_range=(0.0, extent),
        seed=seed,
        octaves=octaves,
        persistence=persistence,
        lacunarity=lacunarity,
    )
    # Shift to non-negative range and scale
    field = (field - field.min()) / (field.max() - field.min() + 1e-12)
    return field * height_scale


def _compute_bounding_box(field: np.ndarray) -> BoundingBox:
    """Compute bounding box for the terrain heightmap."""
    z_min = float(np.min(field))
    z_max = float(np.max(field))
    return BoundingBox(
        min_corner=(0.0, 0.0, z_min),
        max_corner=(1.0, 1.0, z_max),
    )


@register
class TerrainGenerator(GeneratorBase):
    """Multi-octave simplex noise terrain heightmap.

    Uses fractal Brownian motion with configurable octaves for realistic
    terrain surfaces. Output is a scalar field for HEIGHTMAP_RELIEF.
    """

    name = "terrain"
    category = "procedural"
    aliases = ("terrain_heightmap",)
    description = "Multi-octave simplex noise terrain heightmap"
    resolution_params = {
        "pixel_resolution": "Grid points per axis (N² cost)",
    }

    def get_default_params(self) -> dict[str, Any]:
        """Return default parameters for terrain generation."""
        return {
            "octaves": _DEFAULT_OCTAVES,
            "persistence": _DEFAULT_PERSISTENCE,
            "lacunarity": _DEFAULT_LACUNARITY,
            "base_frequency": _DEFAULT_BASE_FREQUENCY,
            "height_scale": _DEFAULT_HEIGHT_SCALE,
        }

    def generate(
        self,
        params: dict[str, Any] | None = None,
        seed: int = 42,
        **resolution_kwargs: Any,
    ) -> MathObject:
        """Generate a multi-octave terrain scalar field."""
        merged = self.get_default_params()
        if params:
            merged.update(params)

        octaves = int(merged["octaves"])
        persistence = float(merged["persistence"])
        lacunarity = float(merged["lacunarity"])
        base_frequency = float(merged["base_frequency"])
        height_scale = float(merged["height_scale"])
        pixel_resolution = int(
            resolution_kwargs.get("pixel_resolution", _DEFAULT_PIXEL_RESOLUTION)
        )

        _validate_params(
            octaves, persistence, lacunarity,
            base_frequency, height_scale, pixel_resolution,
        )

        merged["pixel_resolution"] = pixel_resolution

        field = _compute_terrain_field(
            octaves, persistence, lacunarity,
            base_frequency, height_scale, pixel_resolution, seed,
        )
        bbox = _compute_bounding_box(field)

        logger.info(
            "Generated terrain: octaves=%d, persistence=%.2f, "
            "base_freq=%.2f, pixel_res=%d, field_range=[%.4f, %.4f]",
            octaves, persistence, base_frequency, pixel_resolution,
            float(np.min(field)), float(np.max(field)),
        )

        return MathObject(
            scalar_field=field,
            generator_name=self.resolved_name or self.name,
            category=self.category,
            parameters=merged,
            seed=seed,
            bounding_box=bbox,
        )

    def get_default_representation(self) -> RepresentationConfig:
        """Return HEIGHTMAP_RELIEF as default representation."""
        return RepresentationConfig(type=RepresentationType.HEIGHTMAP_RELIEF)
