"""Genus-2 implicit surface generator.

Constructs a genus-2 surface (double torus) by smoothly blending two
overlapping tori via a soft-minimum of their implicit distance fields.
Each torus is defined by (sqrt(x² + z²) - R)² + y² - r² = 0, centered
at (±d, 0, 0) in the xz-plane. When the tori overlap (d < 2R), the
smooth blend produces a closed surface with two handles (genus 2).

The soft-minimum uses the log-sum-exp formulation for numerical stability:
  softmin(a, b) = m - (1/k) * ln(exp(-k(a-m)) + exp(-k(b-m)))
where m = min(a, b) and k controls blend sharpness.

Note: field evaluation is O(N³) where N is voxel_resolution.
"""

import logging
from typing import Any

import numpy as np

from mathviz.core.generator import GeneratorBase, register
from mathviz.core.math_object import MathObject
from mathviz.core.representation import RepresentationConfig, RepresentationType
from mathviz.shared.marching_cubes import SpatialBounds, bounds_to_bbox, extract_mesh

logger = logging.getLogger(__name__)

_DEFAULT_VOXEL_RESOLUTION = 128
_MIN_VOXEL_RESOLUTION = 16
_DEFAULT_SEPARATION = 1.15
_DEFAULT_MAJOR_RADIUS = 1.0
_DEFAULT_TUBE_RADIUS = 0.4
_DEFAULT_BLEND_SHARPNESS = 15.0


def _torus_field(
    xx: np.ndarray, yy: np.ndarray, zz: np.ndarray,
    center_x: float, major_r: float, tube_r: float,
) -> np.ndarray:
    """Evaluate torus implicit field centered at (center_x, 0, 0) in xz-plane."""
    r_xz = np.sqrt((xx - center_x) ** 2 + zz**2)
    return (r_xz - major_r) ** 2 + yy**2 - tube_r**2


def _softmin(a: np.ndarray, b: np.ndarray, k: float) -> np.ndarray:
    """Numerically stable soft-minimum via log-sum-exp."""
    m = np.minimum(a, b)
    exp_a = np.exp(-k * (a - m))
    exp_b = np.exp(-k * (b - m))
    return m - np.log(exp_a + exp_b) / k


def _evaluate_genus2_field(
    voxel_resolution: int,
    bounds: SpatialBounds,
    separation: float,
    major_radius: float,
    tube_radius: float,
    blend_sharpness: float,
) -> np.ndarray:
    """Evaluate the genus-2 implicit field on an N³ voxel grid."""
    n = voxel_resolution
    x = np.linspace(bounds.min_corner[0], bounds.max_corner[0], n)
    y = np.linspace(bounds.min_corner[1], bounds.max_corner[1], n)
    z = np.linspace(bounds.min_corner[2], bounds.max_corner[2], n)
    xx, yy, zz = np.meshgrid(x, y, z, indexing="ij")

    f1 = _torus_field(xx, yy, zz, separation, major_radius, tube_radius)
    f2 = _torus_field(xx, yy, zz, -separation, major_radius, tube_radius)

    return _softmin(f1, f2, blend_sharpness)


def _compute_bounds(
    separation: float, major_radius: float, tube_radius: float,
) -> SpatialBounds:
    """Compute spatial bounds to fully contain both tori."""
    margin = 0.5
    x_extent = separation + major_radius + tube_radius + margin
    yz_extent = major_radius + tube_radius + margin
    return SpatialBounds(
        min_corner=(-x_extent, -yz_extent, -yz_extent),
        max_corner=(x_extent, yz_extent, yz_extent),
    )


def _validate_params(
    voxel_resolution: int,
    separation: float,
    major_radius: float,
    tube_radius: float,
    blend_sharpness: float,
) -> None:
    """Validate genus-2 surface parameters."""
    if voxel_resolution < _MIN_VOXEL_RESOLUTION:
        raise ValueError(
            f"voxel_resolution must be >= {_MIN_VOXEL_RESOLUTION}, "
            f"got {voxel_resolution}"
        )
    if separation < 0:
        raise ValueError(f"separation must be >= 0, got {separation}")
    if major_radius <= 0:
        raise ValueError(f"major_radius must be positive, got {major_radius}")
    if tube_radius <= 0:
        raise ValueError(f"tube_radius must be positive, got {tube_radius}")
    if tube_radius >= major_radius:
        raise ValueError(
            f"tube_radius ({tube_radius}) must be < major_radius ({major_radius})"
        )
    if blend_sharpness <= 0:
        raise ValueError(
            f"blend_sharpness must be positive, got {blend_sharpness}"
        )


@register
class Genus2SurfaceGenerator(GeneratorBase):
    """Genus-2 implicit surface generator via smooth torus blending.

    Creates a double-torus (genus-2) surface by blending two overlapping
    tori with a smooth minimum of their implicit distance fields. The
    resulting isosurface is closed and has Euler characteristic -2,
    confirming genus 2.

    Seed has no effect; the surface is fully deterministic for given parameters.
    """

    name = "genus2_surface"
    category = "implicit"
    aliases = ()
    description = "Genus-2 surface via smooth blending of two overlapping tori"
    resolution_params = {
        "voxel_resolution": "Number of voxels per axis (N³ cost)",
    }

    def get_default_params(self) -> dict[str, Any]:
        """Return default parameters for the genus-2 surface generator."""
        return {
            "separation": _DEFAULT_SEPARATION,
            "major_radius": _DEFAULT_MAJOR_RADIUS,
            "tube_radius": _DEFAULT_TUBE_RADIUS,
            "blend_sharpness": _DEFAULT_BLEND_SHARPNESS,
        }

    def generate(
        self,
        params: dict[str, Any] | None = None,
        seed: int = 42,
        **resolution_kwargs: Any,
    ) -> MathObject:
        """Generate a genus-2 surface mesh via marching cubes."""
        merged = self.get_default_params()
        if params:
            merged.update(params)

        separation = float(merged["separation"])
        major_radius = float(merged["major_radius"])
        tube_radius = float(merged["tube_radius"])
        blend_sharpness = float(merged["blend_sharpness"])
        voxel_resolution = int(
            resolution_kwargs.get("voxel_resolution", _DEFAULT_VOXEL_RESOLUTION)
        )

        _validate_params(
            voxel_resolution, separation, major_radius,
            tube_radius, blend_sharpness,
        )

        merged["voxel_resolution"] = voxel_resolution

        bounds = _compute_bounds(separation, major_radius, tube_radius)
        field = _evaluate_genus2_field(
            voxel_resolution, bounds,
            separation, major_radius, tube_radius, blend_sharpness,
        )
        mesh = extract_mesh(field, bounds, isolevel=0.0)
        bbox = bounds_to_bbox(bounds)

        logger.info(
            "Generated genus2_surface: separation=%.3f, R=%.3f, r=%.3f, "
            "voxel_resolution=%d, vertices=%d, faces=%d",
            separation, major_radius, tube_radius, voxel_resolution,
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
        """Return the recommended representation for the genus-2 surface."""
        return RepresentationConfig(type=RepresentationType.SURFACE_SHELL)
