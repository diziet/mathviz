"""Möbius strip parametric surface generator.

The Möbius strip is a non-orientable surface with a single boundary curve.
It wraps in the u direction (with a half-twist) but is open in v, so the
resulting mesh is NOT watertight.
"""

import logging
from typing import Any

import numpy as np

from mathviz.core.generator import GeneratorBase, register
from mathviz.core.math_object import BoundingBox, MathObject, Mesh
from mathviz.core.representation import RepresentationConfig, RepresentationType

logger = logging.getLogger(__name__)

_DEFAULT_RADIUS = 1.0
_DEFAULT_HALF_WIDTH = 0.4
_DEFAULT_GRID_RESOLUTION = 128
_MIN_GRID_RESOLUTION = 3


def _evaluate_mobius(
    u: np.ndarray, v: np.ndarray, radius: float, half_width: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Evaluate the Möbius strip parametric surface f(u, v) -> (x, y, z).

    u in [0, 2*pi), v in [-half_width, half_width].
    """
    half_u = u / 2.0
    x = (radius + v * np.cos(half_u)) * np.cos(u)
    y = (radius + v * np.cos(half_u)) * np.sin(u)
    z = v * np.sin(half_u)
    return x, y, z


def _build_u_wrapped_grid_faces(n_u: int, n_v: int) -> np.ndarray:
    """Build triangle faces wrapping in u but open in v."""
    rows = np.arange(n_u)
    cols = np.arange(n_v - 1)
    rr, cc = np.meshgrid(rows, cols, indexing="ij")
    rr, cc = rr.ravel(), cc.ravel()

    i00 = rr * n_v + cc
    i10 = ((rr + 1) % n_u) * n_v + cc
    i01 = rr * n_v + (cc + 1)
    i11 = ((rr + 1) % n_u) * n_v + (cc + 1)

    tri1 = np.stack([i00, i10, i11], axis=-1)
    tri2 = np.stack([i00, i11, i01], axis=-1)
    return np.concatenate([tri1, tri2], axis=0).astype(np.int64)


def _validate_params(
    radius: float, half_width: float, grid_resolution: int,
) -> None:
    """Validate Möbius strip parameters."""
    if radius <= 0:
        raise ValueError(f"radius must be positive, got {radius}")
    if half_width <= 0:
        raise ValueError(f"half_width must be positive, got {half_width}")
    if grid_resolution < _MIN_GRID_RESOLUTION:
        raise ValueError(
            f"grid_resolution must be >= {_MIN_GRID_RESOLUTION}, "
            f"got {grid_resolution}"
        )


def _compute_bounding_box(radius: float, half_width: float) -> BoundingBox:
    """Compute axis-aligned bounding box for the Möbius strip."""
    xy_extent = radius + half_width
    z_extent = half_width
    return BoundingBox(
        min_corner=(-xy_extent, -xy_extent, -z_extent),
        max_corner=(xy_extent, xy_extent, z_extent),
    )


def _generate_mobius_mesh(
    radius: float, half_width: float, grid_resolution: int,
) -> Mesh:
    """Build triangle mesh for the Möbius strip."""
    n = grid_resolution
    u_vals = np.linspace(0, 2 * np.pi, n, endpoint=False)
    v_vals = np.linspace(-half_width, half_width, n)
    uu, vv = np.meshgrid(u_vals, v_vals, indexing="ij")

    x, y, z = _evaluate_mobius(uu, vv, radius, half_width)
    vertices = np.column_stack([x.ravel(), y.ravel(), z.ravel()])
    vertices = vertices.astype(np.float64)
    faces = _build_u_wrapped_grid_faces(n, n)
    return Mesh(vertices=vertices, faces=faces)


@register
class MobiusStripGenerator(GeneratorBase):
    """Parametric Möbius strip surface generator."""

    name = "mobius_strip"
    category = "parametric"
    aliases = ()
    description = "Möbius strip with configurable radius and width"
    resolution_params = {"grid_resolution": "Number of grid divisions per axis"}

    def get_default_params(self) -> dict[str, Any]:
        """Return default parameters for the Möbius strip."""
        return {
            "radius": _DEFAULT_RADIUS,
            "half_width": _DEFAULT_HALF_WIDTH,
        }

    def get_default_resolution(self) -> dict[str, Any]:
        """Return default values for resolution parameters."""
        return {"grid_resolution": _DEFAULT_GRID_RESOLUTION}

    def generate(
        self,
        params: dict[str, Any] | None = None,
        seed: int = 42,
        **resolution_kwargs: Any,
    ) -> MathObject:
        """Generate a Möbius strip mesh."""
        merged = self.get_default_params()
        if params:
            merged.update(params)

        radius = float(merged["radius"])
        half_width = float(merged["half_width"])
        grid_resolution = int(
            resolution_kwargs.get("grid_resolution", _DEFAULT_GRID_RESOLUTION)
        )

        _validate_params(radius, half_width, grid_resolution)

        mesh = _generate_mobius_mesh(radius, half_width, grid_resolution)
        bbox = _compute_bounding_box(radius, half_width)

        logger.info(
            "Generated mobius_strip: R=%.3f, w=%.3f, grid=%d, "
            "vertices=%d, faces=%d",
            radius, half_width, grid_resolution,
            len(mesh.vertices), len(mesh.faces),
        )

        merged["grid_resolution"] = grid_resolution

        return MathObject(
            mesh=mesh,
            generator_name=self.name,
            category=self.category,
            parameters=merged,
            seed=seed,
            bounding_box=bbox,
        )

    def get_default_representation(self) -> RepresentationConfig:
        """Return the recommended representation for the Möbius strip."""
        return RepresentationConfig(type=RepresentationType.SURFACE_SHELL)
