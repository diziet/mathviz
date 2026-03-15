"""Cross-cap parametric surface generator.

The cross-cap is a non-orientable surface that models the real projective
plane immersed in three-dimensional Euclidean space. It features a single
point of self-intersection along a line segment.
"""

import logging
from typing import Any

import numpy as np

from mathviz.core.generator import GeneratorBase, register
from mathviz.core.math_object import BoundingBox, MathObject, Mesh
from mathviz.core.representation import RepresentationConfig, RepresentationType
from mathviz.generators.parametric._mesh_utils import build_open_grid_faces

logger = logging.getLogger(__name__)

_DEFAULT_SCALE = 1.0
_DEFAULT_GRID_RESOLUTION = 128
_MIN_GRID_RESOLUTION = 3


def _evaluate_cross_cap(
    u: np.ndarray, v: np.ndarray, scale: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Evaluate the cross-cap immersion f(u, v) -> (x, y, z).

    u, v are 2D arrays from meshgrid, both in [0, pi].
    """
    sin_u = np.sin(u)
    sin_2v = np.sin(2.0 * v)
    sin_2u = np.sin(2.0 * u)
    cos_2u = np.cos(2.0 * u)
    cos_v_sq = np.cos(v) ** 2

    x = scale * sin_u * sin_2v / 2.0
    y = scale * sin_2u * cos_v_sq
    z = scale * cos_2u * cos_v_sq
    return x, y, z


def _validate_params(scale: float, grid_resolution: int) -> None:
    """Validate cross-cap parameters."""
    if scale <= 0:
        raise ValueError(f"scale must be positive, got {scale}")
    if grid_resolution < _MIN_GRID_RESOLUTION:
        raise ValueError(
            f"grid_resolution must be >= {_MIN_GRID_RESOLUTION}, "
            f"got {grid_resolution}"
        )


def _compute_bounding_box(scale: float) -> BoundingBox:
    """Compute axis-aligned bounding box for the cross-cap.

    The x-component has max |x| = scale/2 (from sin(u)*sin(2v)/2),
    while y and z reach scale. We use per-axis extents with margin.
    """
    x_extent = scale * 0.55  # max |x| = scale/2, with 10% margin
    yz_extent = scale * 1.05
    return BoundingBox(
        min_corner=(-x_extent, -yz_extent, -yz_extent),
        max_corner=(x_extent, yz_extent, yz_extent),
    )


def _generate_cross_cap_mesh(scale: float, grid_resolution: int) -> Mesh:
    """Build triangle mesh for the cross-cap surface."""
    n = grid_resolution
    u_vals = np.linspace(0, np.pi, n, endpoint=True)
    v_vals = np.linspace(0, np.pi, n, endpoint=True)
    uu, vv = np.meshgrid(u_vals, v_vals, indexing="ij")

    x, y, z = _evaluate_cross_cap(uu, vv, scale)
    vertices = np.column_stack([x.ravel(), y.ravel(), z.ravel()])
    vertices = vertices.astype(np.float64)
    faces = build_open_grid_faces(n, n)
    return Mesh(vertices=vertices, faces=faces)


@register
class CrossCapGenerator(GeneratorBase):
    """Parametric cross-cap surface — real projective plane immersed in R³."""

    name = "cross_cap"
    category = "parametric"
    aliases = ("crosscap",)
    description = "Non-orientable cross-cap immersion of the real projective plane"
    resolution_params = {"grid_resolution": "Number of grid divisions per axis"}
    _resolution_defaults = {"grid_resolution": _DEFAULT_GRID_RESOLUTION}

    def get_default_params(self) -> dict[str, Any]:
        """Return default parameters for the cross-cap."""
        return {"scale": _DEFAULT_SCALE}

    def generate(
        self,
        params: dict[str, Any] | None = None,
        seed: int = 42,
        **resolution_kwargs: Any,
    ) -> MathObject:
        """Generate a cross-cap surface mesh."""
        merged = self.get_default_params()
        if params:
            merged.update(params)

        scale = float(merged["scale"])
        grid_resolution = int(
            resolution_kwargs.get(
                "grid_resolution",
                self._resolution_defaults["grid_resolution"],
            )
        )

        _validate_params(scale, grid_resolution)

        mesh = _generate_cross_cap_mesh(scale, grid_resolution)
        bbox = _compute_bounding_box(scale)

        merged["grid_resolution"] = grid_resolution

        logger.info(
            "Generated cross_cap: scale=%.3f, grid=%d, vertices=%d, faces=%d",
            scale, grid_resolution, len(mesh.vertices), len(mesh.faces),
        )

        return MathObject(
            mesh=mesh,
            generator_name=self.name,
            category=self.category,
            parameters=merged,
            seed=seed,
            bounding_box=bbox,
        )

    def get_default_representation(self) -> RepresentationConfig:
        """Return the recommended representation for the cross-cap."""
        return RepresentationConfig(type=RepresentationType.SURFACE_SHELL)
