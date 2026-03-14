"""Torus parametric surface generator.

The torus is periodic in both u and v, so the grid wraps at boundaries.
"""

import logging
from typing import Any

import numpy as np

from mathviz.core.generator import GeneratorBase, register
from mathviz.core.math_object import BoundingBox, MathObject, Mesh
from mathviz.core.representation import RepresentationConfig, RepresentationType

logger = logging.getLogger(__name__)

_DEFAULT_MAJOR_RADIUS = 1.0
_DEFAULT_MINOR_RADIUS = 0.4
_DEFAULT_GRID_RESOLUTION = 128
_MIN_GRID_RESOLUTION = 3


def _evaluate_torus(
    u: np.ndarray,
    v: np.ndarray,
    major_radius: float,
    minor_radius: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Evaluate the torus parametric surface f(u, v) -> (x, y, z).

    u, v are 2D arrays from meshgrid, both in [0, 2*pi).
    """
    x = (major_radius + minor_radius * np.cos(v)) * np.cos(u)
    y = (major_radius + minor_radius * np.cos(v)) * np.sin(u)
    z = minor_radius * np.sin(v)
    return x, y, z


def _build_wrapped_grid_faces(resolution: int) -> np.ndarray:
    """Build triangle faces for a periodic grid that wraps in both u and v.

    For an N×N grid of vertices (no duplicated boundary), each cell produces
    two triangles, wrapping indices with modulo.
    """
    n = resolution
    row = np.arange(n)
    col = np.arange(n)
    rr, cc = np.meshgrid(row, col, indexing="ij")
    rr = rr.ravel()
    cc = cc.ravel()

    # Four corners of each quad, wrapping with modulo
    i00 = rr * n + cc
    i10 = ((rr + 1) % n) * n + cc
    i01 = rr * n + ((cc + 1) % n)
    i11 = ((rr + 1) % n) * n + ((cc + 1) % n)

    tri1 = np.stack([i00, i10, i11], axis=-1)
    tri2 = np.stack([i00, i11, i01], axis=-1)
    faces = np.concatenate([tri1, tri2], axis=0)
    return faces.astype(np.int64)


def _compute_bounding_box(
    major_radius: float, minor_radius: float,
) -> BoundingBox:
    """Compute the axis-aligned bounding box for a torus."""
    xy_extent = major_radius + minor_radius
    z_extent = minor_radius
    return BoundingBox(
        min_corner=(-xy_extent, -xy_extent, -z_extent),
        max_corner=(xy_extent, xy_extent, z_extent),
    )


@register
class TorusGenerator(GeneratorBase):
    """Parametric torus surface generator."""

    name = "torus"
    category = "parametric"
    aliases = ()
    description = "Parametric torus surface with configurable radii"
    resolution_params = {"grid_resolution": "Number of grid divisions per axis"}

    def get_default_params(self) -> dict[str, Any]:
        """Return default parameters for the torus generator."""
        return {
            "major_radius": _DEFAULT_MAJOR_RADIUS,
            "minor_radius": _DEFAULT_MINOR_RADIUS,
        }

    def generate(
        self,
        params: dict[str, Any] | None = None,
        seed: int = 42,
        **resolution_kwargs: Any,
    ) -> MathObject:
        """Generate a torus mesh."""
        merged = self.get_default_params()
        if params:
            merged.update(params)

        major_radius = float(merged["major_radius"])
        minor_radius = float(merged["minor_radius"])
        grid_resolution = int(
            resolution_kwargs.get("grid_resolution", _DEFAULT_GRID_RESOLUTION)
        )

        _validate_params(major_radius, minor_radius, grid_resolution)

        mesh = _generate_torus_mesh(major_radius, minor_radius, grid_resolution)
        bbox = _compute_bounding_box(major_radius, minor_radius)

        logger.info(
            "Generated torus: R=%.3f, r=%.3f, grid=%d, vertices=%d, faces=%d",
            major_radius, minor_radius, grid_resolution,
            len(mesh.vertices), len(mesh.faces),
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
        """Return the recommended representation for the torus."""
        return RepresentationConfig(type=RepresentationType.SURFACE_SHELL)


def _validate_params(
    major_radius: float, minor_radius: float, grid_resolution: int,
) -> None:
    """Validate torus parameters, raising ValueError for invalid inputs."""
    if major_radius <= 0:
        raise ValueError(
            f"major_radius must be positive, got {major_radius}"
        )
    if minor_radius <= 0:
        raise ValueError(
            f"minor_radius must be positive, got {minor_radius}"
        )
    if minor_radius >= major_radius:
        logger.warning(
            "minor_radius (%.3f) >= major_radius (%.3f): "
            "torus will self-intersect",
            minor_radius, major_radius,
        )
    if grid_resolution < _MIN_GRID_RESOLUTION:
        raise ValueError(
            f"grid_resolution must be >= {_MIN_GRID_RESOLUTION}, "
            f"got {grid_resolution}"
        )


def _generate_torus_mesh(
    major_radius: float, minor_radius: float, grid_resolution: int,
) -> Mesh:
    """Build a triangle mesh for a torus from parametric evaluation."""
    n = grid_resolution
    u_vals = np.linspace(0, 2 * np.pi, n, endpoint=False)
    v_vals = np.linspace(0, 2 * np.pi, n, endpoint=False)
    uu, vv = np.meshgrid(u_vals, v_vals, indexing="ij")

    x, y, z = _evaluate_torus(uu, vv, major_radius, minor_radius)

    vertices = np.column_stack([x.ravel(), y.ravel(), z.ravel()])
    vertices = vertices.astype(np.float64)

    faces = _build_wrapped_grid_faces(n)

    return Mesh(vertices=vertices, faces=faces)
