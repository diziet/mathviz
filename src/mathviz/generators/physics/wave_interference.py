"""Wave interference pattern generator.

Generates 3D standing wave interference patterns from multiple point sources.
Computes the superposition of spherical waves sin(k*r - ωt) / r at each
point in a 3D voxel grid, then extracts an isosurface at a threshold
amplitude via marching cubes.
"""

import logging
from typing import Any

import numpy as np
from numpy.random import default_rng

from mathviz.core.generator import GeneratorBase, register
from mathviz.core.math_object import BoundingBox, MathObject
from mathviz.core.representation import RepresentationConfig, RepresentationType
from mathviz.shared.marching_cubes import SpatialBounds, extract_mesh

logger = logging.getLogger(__name__)

_DEFAULT_NUM_SOURCES = 3
_DEFAULT_WAVELENGTH = 0.5
_DEFAULT_SOURCE_SPACING = 1.0
_DEFAULT_VOXEL_RESOLUTION = 128
_DEFAULT_ISO_LEVEL = 0.5
_DEFAULT_TIME = 0.0

_MIN_NUM_SOURCES = 1
_MAX_NUM_SOURCES = 20
_MIN_VOXEL_RESOLUTION = 16
_MIN_WAVELENGTH = 0.01
_SOFTENING = 1e-6


def _validate_params(
    num_sources: int,
    wavelength: float,
    source_spacing: float,
    voxel_resolution: int,
    iso_level: float,
) -> None:
    """Validate wave interference parameters."""
    if num_sources < _MIN_NUM_SOURCES:
        raise ValueError(
            f"num_sources must be >= {_MIN_NUM_SOURCES}, got {num_sources}"
        )
    if num_sources > _MAX_NUM_SOURCES:
        raise ValueError(
            f"num_sources must be <= {_MAX_NUM_SOURCES}, got {num_sources}"
        )
    if wavelength < _MIN_WAVELENGTH:
        raise ValueError(
            f"wavelength must be >= {_MIN_WAVELENGTH}, got {wavelength}"
        )
    if source_spacing <= 0:
        raise ValueError(
            f"source_spacing must be > 0, got {source_spacing}"
        )
    if voxel_resolution < _MIN_VOXEL_RESOLUTION:
        raise ValueError(
            f"voxel_resolution must be >= {_MIN_VOXEL_RESOLUTION}, "
            f"got {voxel_resolution}"
        )
    if iso_level <= 0:
        raise ValueError(f"iso_level must be > 0, got {iso_level}")
    if iso_level >= 1.0:
        raise ValueError(
            f"iso_level must be < 1.0 (field is normalized to [-1, 1]), "
            f"got {iso_level}"
        )


def _place_sources(
    num_sources: int,
    source_spacing: float,
    rng: np.random.Generator,
) -> np.ndarray:
    """Place point sources on the z=0 plane with jitter from seed.

    Sources are arranged in a regular polygon pattern centered at the
    origin, with small random jitter controlled by the RNG.
    """
    if num_sources == 1:
        base = np.array([[0.0, 0.0, 0.0]])
    else:
        angles = np.linspace(0, 2 * np.pi, num_sources, endpoint=False)
        radius = source_spacing * num_sources / (2 * np.pi)
        base = np.column_stack([
            radius * np.cos(angles),
            radius * np.sin(angles),
            np.zeros(num_sources),
        ])

    jitter = rng.uniform(-0.1 * source_spacing, 0.1 * source_spacing,
                         size=base.shape)
    jitter[:, 2] = 0.0  # keep sources on z=0 plane
    return base + jitter


def _compute_wave_field(
    sources: np.ndarray,
    wavelength: float,
    time: float,
    voxel_resolution: int,
    grid_extent: float,
) -> tuple[np.ndarray, SpatialBounds]:
    """Compute superposition of spherical waves on a 3D grid."""
    k = 2 * np.pi / wavelength
    omega = 2 * np.pi / wavelength  # wave speed = 1

    coords = np.linspace(-grid_extent, grid_extent, voxel_resolution)
    x, y, z = np.meshgrid(coords, coords, coords, indexing="ij")

    field = np.zeros_like(x)
    r = np.empty_like(x)
    tmp = np.empty_like(x)
    for source in sources:
        np.subtract(x, source[0], out=r)
        np.multiply(r, r, out=r)
        np.subtract(y, source[1], out=tmp)
        r += tmp * tmp
        np.subtract(z, source[2], out=tmp)
        r += tmp * tmp
        r += _SOFTENING
        np.sqrt(r, out=r)
        # sin(k*r - omega*time) / r
        np.multiply(r, k, out=tmp)
        tmp -= omega * time
        np.sin(tmp, out=tmp)
        tmp /= r
        field += tmp

    bounds = SpatialBounds(
        min_corner=(-grid_extent, -grid_extent, -grid_extent),
        max_corner=(grid_extent, grid_extent, grid_extent),
    )
    return field, bounds


@register
class WaveInterferenceGenerator(GeneratorBase):
    """3D standing wave interference pattern from multiple point sources."""

    name = "wave_interference"
    category = "physics"
    aliases = ("wave_pattern", "interference")
    description = (
        "3D standing wave interference pattern from multiple "
        "point sources with isosurface extraction"
    )
    resolution_params = {
        "voxel_resolution": "Voxels per axis (N³ cost)",
    }
    _resolution_defaults = {"voxel_resolution": _DEFAULT_VOXEL_RESOLUTION}

    def get_default_params(self) -> dict[str, Any]:
        """Return default wave interference parameters."""
        return {
            "num_sources": _DEFAULT_NUM_SOURCES,
            "wavelength": _DEFAULT_WAVELENGTH,
            "source_spacing": _DEFAULT_SOURCE_SPACING,
            "iso_level": _DEFAULT_ISO_LEVEL,
            "time": _DEFAULT_TIME,
        }

    def generate(
        self,
        params: dict[str, Any] | None = None,
        seed: int = 42,
        **resolution_kwargs: Any,
    ) -> MathObject:
        """Generate a wave interference isosurface mesh."""
        merged = self.get_default_params()
        if params:
            merged.update(params)

        num_sources = int(merged["num_sources"])
        wavelength = float(merged["wavelength"])
        source_spacing = float(merged["source_spacing"])
        iso_level = float(merged["iso_level"])
        time = float(merged["time"])
        voxel_resolution = int(
            resolution_kwargs.get(
                "voxel_resolution",
                self._resolution_defaults["voxel_resolution"],
            )
        )

        _validate_params(
            num_sources, wavelength, source_spacing,
            voxel_resolution, iso_level,
        )
        merged["voxel_resolution"] = voxel_resolution

        rng = default_rng(seed)
        sources = _place_sources(num_sources, source_spacing, rng)

        # Grid extent based on source arrangement plus extra wavelengths
        max_source_dist = float(np.max(np.abs(sources))) if num_sources > 1 else 0.0
        grid_extent = max(max_source_dist + 3 * wavelength, 2.0 * wavelength)

        field, bounds = _compute_wave_field(
            sources, wavelength, time, voxel_resolution, grid_extent,
        )

        # Normalize field for consistent iso_level interpretation
        max_abs = max(float(field.max()), float(-field.min()))
        if max_abs > 0:
            field /= max_abs

        mesh = extract_mesh(field, bounds, isolevel=iso_level)
        bbox = BoundingBox(
            min_corner=bounds.min_corner,
            max_corner=bounds.max_corner,
        )

        logger.info(
            "Generated wave_interference: sources=%d, wavelength=%.3f, "
            "voxel_res=%d, vertices=%d, faces=%d",
            num_sources, wavelength, voxel_resolution,
            len(mesh.vertices), len(mesh.faces),
        )

        return MathObject(
            mesh=mesh,
            generator_name=self.resolved_name or self.name,
            category=self.category,
            parameters=merged,
            seed=seed,
            bounding_box=bbox,
        )

    def get_default_representation(self) -> RepresentationConfig:
        """Return SURFACE_SHELL as default representation."""
        return RepresentationConfig(type=RepresentationType.SURFACE_SHELL)
